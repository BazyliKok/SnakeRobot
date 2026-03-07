from replaybuffer import EnvReplayBuffer
from rlkit.data_management.replay_buffer import ReplayBuffer
import numpy as np
import torch

class CoadaptReplayBuffer(ReplayBuffer):
    def __init__(
            self,
            max_replay_buffer_size_species,
            max_replay_buffer_size_population,
            env,
            env_info_sizes=None
    ):
        self._env = env
        self._max_replay_buffer_size_species = max_replay_buffer_size_species
        self._max_replay_buffer_size_population = max_replay_buffer_size_population
        
        if env_info_sizes is None:
            env_info_sizes = {'terrain_id': 1}
        elif 'terrain_id' not in env_info_sizes:
            env_info_sizes = dict(env_info_sizes)
            env_info_sizes['terrain_id'] = 1

        self._env_info_sizes = env_info_sizes

        # default mode 
        self._mode = "species"

        self._ep_counter = 0
        self._expect_init_state = True # LOOK AT THIS VARIABLE?
      
        # init replay buffers
        self._individual_buffer = EnvReplayBuffer(
            env=self._env,
            max_replay_buffer_size=self._max_replay_buffer_size_species,
            env_info_sizes=self._env_info_sizes,
        )
        self._population_buffer = EnvReplayBuffer(
            env=self._env,
            max_replay_buffer_size=self._max_replay_buffer_size_population,
            env_info_sizes=self._env_info_sizes,
        )
        self._init_state_buffer = EnvReplayBuffer(
            env=self._env,
            max_replay_buffer_size=self._max_replay_buffer_size_population,
            env_info_sizes=self._env_info_sizes,
        )
    
    # def __getstate__(self):
    #     state = self.__dict__.copy()
    #     # remove env from state to make it picklable
    #     if 'env' in state:
    #         state['env'] = None
    #     if '_env' in state:
    #         state['_env'] = None
    #     return state

    # def __setstate__(self, state):
    #     self.__dict__.update(state)
    #     # reset of env
    #     self.env = None
    #     self._env = None

    def dump(self, filepath: str):
        """Safely save the replay buffer to a file, excluding non-pickleable items."""
        safe_dict = {}

        # only copy safe items
        for k, v in self.__dict__.items():
            if "env" in k or "lock" in k or "socket" in str(type(v)):
                continue
            try:
                torch.save(v, filepath + ".tmp")  # test if it's savable
                safe_dict[k] = v
            except Exception as e:
                print(f"skipping key '{k}' in replay buffer (unsavable): {e}")

        torch.save({'buffer': safe_dict}, filepath)
        print(f"replay buffer saved to {filepath}")

    @classmethod
    def load(cls, filepath: str, env=None):
        """Load the replay buffer and reattach the environment."""
        saved = torch.load(filepath)
        buffer = cls.__new__(cls)
        buffer.__dict__.update(saved['buffer'])

        # restore env manually
        buffer.env = env
        buffer._env = env
        print(f"replay buffer loaded from {filepath}")
        return buffer


    def add_sample(self, observation, action, reward, terminal,
                   next_observation, env_info=None, **kwargs):
        if env_info is None:
            env_info = {}
        terrain_id = int(env_info.get('terrain_id', -1))
        env_info = dict(env_info)
        env_info['terrain_id'] = np.array([terrain_id], dtype=np.float32)

        self._individual_buffer.add_sample(observation=observation, action=action, reward=reward, terminal=terminal, next_observation=next_observation, env_info=env_info, **kwargs)
        self._population_buffer.add_sample(observation=observation, action=action, reward=reward, terminal=terminal, next_observation=next_observation, env_info=env_info, **kwargs)

        # TODO: What is the point of an intitial state replay buffer?
        if self._expect_init_state:
            self._init_state_buffer.add_sample(observation=observation, action=action, reward=reward, terminal=terminal, next_observation=next_observation, env_info=env_info, **kwargs)
            self._init_state_buffer.terminate_episode() # right now terminate episode is a pass but could change
            self._expect_init_state = False

    def _random_batch_from_indices(self, buffer, indices):
        batch = dict(
            observations=buffer._observations[indices],
            actions=buffer._actions[indices],
            rewards=buffer._rewards[indices],
            terminals=buffer._terminals[indices],
            next_observations=buffer._next_obs[indices],
        )
        for key in buffer._env_info_keys:
            batch[key] = buffer._env_infos[key][indices]
        return batch

    def _balanced_random_batch(self, buffer, batch_size):
        if buffer._size <= 0:
            return buffer.random_batch(batch_size)

        if 'terrain_id' not in buffer._env_info_keys:
            return buffer.random_batch(batch_size)

        terrain_ids = buffer._env_infos['terrain_id'][:buffer._size].reshape(-1).astype(int)
        valid_terrain_ids = sorted([tid for tid in np.unique(terrain_ids) if tid >= 0])
        if not valid_terrain_ids:
            return buffer.random_batch(batch_size)

        n_terrains = len(valid_terrain_ids)
        base = batch_size // n_terrains
        remainder = batch_size % n_terrains

        sampled_indices = []
        for i, terrain_id in enumerate(valid_terrain_ids):
            n_take = base + (1 if i < remainder else 0)
            if n_take == 0:
                continue
            candidate = np.where(terrain_ids == terrain_id)[0]
            if len(candidate) == 0:
                continue
            replace = len(candidate) < n_take
            sampled = np.random.choice(candidate, size=n_take, replace=replace)
            sampled_indices.append(sampled)

        if not sampled_indices:
            return buffer.random_batch(batch_size)

        indices = np.concatenate(sampled_indices, axis=0)
        if len(indices) < batch_size:
            # Top up from all data to preserve requested batch size.
            extra = np.random.choice(buffer._size, size=batch_size - len(indices), replace=buffer._size < (batch_size - len(indices)))
            indices = np.concatenate([indices, extra], axis=0)

        np.random.shuffle(indices)
        return self._random_batch_from_indices(buffer, indices)

    def _terrain_random_batch(self, buffer, batch_size, terrain_id):
        if buffer._size <= 0:
            return buffer.random_batch(batch_size)

        if 'terrain_id' not in buffer._env_info_keys:
            return buffer.random_batch(batch_size)

        terrain_ids = buffer._env_infos['terrain_id'][:buffer._size].reshape(-1).astype(int)
        candidate = np.where(terrain_ids == int(terrain_id))[0]
        if len(candidate) == 0:
            return self._balanced_random_batch(buffer, batch_size)

        replace = len(candidate) < batch_size
        indices = np.random.choice(candidate, size=batch_size, replace=replace)
        return self._random_batch_from_indices(buffer, indices)

    def random_start_batch_for_terrain(self, batch_size, terrain_id):
        """Sample a start-state batch from one terrain when available."""
        return self._terrain_random_batch(
            buffer=self._init_state_buffer,
            batch_size=batch_size,
            terrain_id=terrain_id,
        )

    def terminate_episode(self):
        """
        :return: # of unique items that can be sampled.
        """
        
        #if self._mode == "species": # double check why we should check this??

        self._individual_buffer.terminate_episode()
        self._population_buffer.terminate_episode()
        self._ep_counter += 1
        self._expect_init_state = True



    def num_steps_can_sample(self, **kwargs):

        if self._mode == "species":
            return self._individual_buffer.num_steps_can_sample(**kwargs)
        elif self._mode == "population":
            return self._population_buffer.num_steps_can_sample(**kwargs)
        else:
            raise ValueError(f"Unknown replay buffer mode: {self._mode}")

    def random_batch(self, batch_size):
        """
        Return a batch of size `batch_size`.
        :param batch_size:
        :return:
        """
        if self._mode == "species":
            # TODO: Figure out what to put here
            ind_batch_size = int(np.floor(batch_size * 0.9))
            pop_batch_size = int(np.ceil(batch_size * 0.1))
            pop = self._balanced_random_batch(self._population_buffer, pop_batch_size)
            spec = self._balanced_random_batch(self._individual_buffer, ind_batch_size)
            for key, item in pop.items():
                pop[key] = np.concatenate([pop[key], spec[key]], axis=0)
            return pop
 
    
        elif self._mode == "population":
            return self._balanced_random_batch(self._population_buffer, batch_size)
        
        elif self._mode == "start":
            return self._balanced_random_batch(self._init_state_buffer, batch_size)
        
        else:
            raise ValueError(f"Unknown replay buffer mode: {self._mode}")

    
    def set_mode(self, mode):
        if mode == "species": # TODO: change to "individual"
            self._mode = mode
        elif mode == "population":
            self._mode = mode
        elif mode == "start":
            self._mode = mode
        else:
            raise ValueError(f"No known mode: {mode}")

    
    def reset_individual_buffer(self):
        self._individual_buffer = EnvReplayBuffer(
            env=self._env,
            max_replay_buffer_size=self._max_replay_buffer_size_species,
            env_info_sizes=self._env_info_sizes,
        )
        self._ep_counter = 0 # reset number of episodes for next design


