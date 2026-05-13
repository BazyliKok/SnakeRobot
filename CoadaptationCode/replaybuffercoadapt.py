from replaybuffer import EnvReplayBuffer
from rlkit.data_management.replay_buffer import ReplayBuffer
import numpy as np
import os
import torch

class CoadaptReplayBuffer(ReplayBuffer):
    def __init__(
            self,
            max_replay_buffer_size_species,
            max_replay_buffer_size_population,
            env,
            env_info_sizes=None,
            terrain_model_mode='separate',
            terrain_name_to_id=None,
            terrain_names=None,
    ):
        self._env = env
        self._max_replay_buffer_size_species = max_replay_buffer_size_species
        self._max_replay_buffer_size_population = max_replay_buffer_size_population
        self._terrain_model_mode = str(terrain_model_mode).strip().lower()
        if self._terrain_model_mode not in ('separate', 'shared'):
            raise ValueError(
                "terrain_model_mode must be either 'separate' or 'shared'."
            )
        self._terrain_name_to_id = dict(terrain_name_to_id or {})
        self._terrain_id_to_name = {
            int(terrain_id): terrain_name
            for terrain_name, terrain_id in self._terrain_name_to_id.items()
        }
        self._terrain_names = list(terrain_names or self._terrain_name_to_id.keys())
        self._active_terrain_name = None
        self._active_terrain_id = None
        
        default_env_info_sizes = {
            'terrain_id': 1,
            'episode_return': 1,
            'episode_progress_cm': 1,
            'episode_positive_reward_fraction': 1,
            'episode_mean_action_delta': 1,
            'episode_score': 1,
        }
        if env_info_sizes is None:
            env_info_sizes = default_env_info_sizes
        else:
            env_info_sizes = dict(env_info_sizes)
            for key, size in default_env_info_sizes.items():
                env_info_sizes.setdefault(key, size)

        self._env_info_sizes = env_info_sizes
        self._individual_batch_fraction = float(
            np.clip(
                float(os.getenv('SNAKE_INDIVIDUAL_BATCH_FRACTION', '0.8')),
                0.0,
                1.0,
            )
        )
        self._reward_biased_batch_fraction = float(
            np.clip(
                float(os.getenv('SNAKE_REWARD_BIASED_BATCH_FRACTION', '0.0')),
                0.0,
                1.0,
            )
        )
        self._reward_bias_temperature = max(
            1e-6,
            float(os.getenv('SNAKE_REWARD_BIAS_TEMPERATURE', '0.5')),
        )
        self._reward_bias_step_weight = float(os.getenv('SNAKE_REWARD_BIAS_STEP_WEIGHT', '1.0'))
        self._reward_bias_episode_weight = float(os.getenv('SNAKE_REWARD_BIAS_EPISODE_WEIGHT', '1.0'))
        self._active_terrain_ids = None

        # default mode 
        self._mode = "species"

        self._ep_counter = 0
        self._expect_init_state = True # LOOK AT THIS VARIABLE?
      
        self._individual_buffers = {}
        self._population_buffers = {}
        self._init_state_buffers = {}

        if self._terrain_model_mode == 'separate':
            for terrain_name in self._terrain_names:
                self._ensure_terrain_buffers(terrain_name)
            if self._terrain_names:
                self.set_active_terrain(self._terrain_names[0])
            else:
                self._individual_buffer = self._new_individual_buffer()
                self._population_buffer = self._new_population_buffer()
                self._init_state_buffer = self._new_population_buffer()
        else:
            self._individual_buffer = self._new_individual_buffer()
            self._population_buffer = self._new_population_buffer()
            self._init_state_buffer = self._new_population_buffer()
            self._individual_buffers = {'shared': self._individual_buffer}
            self._population_buffers = {'shared': self._population_buffer}
            self._init_state_buffers = {'shared': self._init_state_buffer}
    
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
        if not hasattr(buffer, '_active_terrain_ids'):
            buffer._active_terrain_ids = None
        if not hasattr(buffer, '_reward_biased_batch_fraction'):
            buffer._reward_biased_batch_fraction = 0.0
        if not hasattr(buffer, '_reward_bias_temperature'):
            buffer._reward_bias_temperature = 0.5
        if not hasattr(buffer, '_reward_bias_step_weight'):
            buffer._reward_bias_step_weight = 1.0
        if not hasattr(buffer, '_reward_bias_episode_weight'):
            buffer._reward_bias_episode_weight = 1.0
        print(f"replay buffer loaded from {filepath}")
        return buffer

    def _new_individual_buffer(self):
        return EnvReplayBuffer(
            env=self._env,
            max_replay_buffer_size=self._max_replay_buffer_size_species,
            env_info_sizes=self._env_info_sizes,
        )

    def _new_population_buffer(self):
        return EnvReplayBuffer(
            env=self._env,
            max_replay_buffer_size=self._max_replay_buffer_size_population,
            env_info_sizes=self._env_info_sizes,
        )

    def configure_terrains(self, terrain_name_to_id=None, terrain_names=None):
        if terrain_name_to_id is not None:
            self._terrain_name_to_id = dict(terrain_name_to_id)
            self._terrain_id_to_name = {
                int(terrain_id): terrain_name
                for terrain_name, terrain_id in self._terrain_name_to_id.items()
            }
        if terrain_names is not None:
            self._terrain_names = list(terrain_names)
        if self._terrain_model_mode == 'separate':
            for terrain_name in self._terrain_names:
                self._ensure_terrain_buffers(terrain_name)

    def _terrain_name_from_id(self, terrain_id):
        terrain_id = int(terrain_id)
        if terrain_id in self._terrain_id_to_name:
            return self._terrain_id_to_name[terrain_id]
        return f'terrain_{terrain_id}'

    def _terrain_id_from_name(self, terrain_name):
        if terrain_name is None:
            return None
        if terrain_name in self._terrain_name_to_id:
            return int(self._terrain_name_to_id[terrain_name])
        return None

    def _coerce_terrain_name(self, terrain_name=None, terrain_id=None):
        if self._terrain_model_mode == 'shared':
            return 'shared'
        if terrain_name is None and terrain_id is not None:
            terrain_name = self._terrain_name_from_id(terrain_id)
        if terrain_name is None:
            terrain_name = self._active_terrain_name
        if terrain_name is None:
            terrain_name = self._terrain_names[0] if self._terrain_names else 'terrain_unknown'
        return str(terrain_name)

    def _ensure_terrain_buffers(self, terrain_name=None, terrain_id=None):
        terrain_name = self._coerce_terrain_name(terrain_name, terrain_id)
        if terrain_name not in self._individual_buffers:
            self._individual_buffers[terrain_name] = self._new_individual_buffer()
        if terrain_name not in self._population_buffers:
            self._population_buffers[terrain_name] = self._new_population_buffer()
        if terrain_name not in self._init_state_buffers:
            self._init_state_buffers[terrain_name] = self._new_population_buffer()
        if terrain_name not in self._terrain_names and terrain_name != 'shared':
            self._terrain_names.append(terrain_name)
        return terrain_name

    def _bind_active_terrain_buffers(self, terrain_name):
        terrain_name = self._ensure_terrain_buffers(terrain_name)
        self._active_terrain_name = terrain_name
        self._active_terrain_id = self._terrain_id_from_name(terrain_name)
        self._individual_buffer = self._individual_buffers[terrain_name]
        self._population_buffer = self._population_buffers[terrain_name]
        self._init_state_buffer = self._init_state_buffers[terrain_name]

    def set_active_terrain(self, terrain_name=None, terrain_id=None):
        terrain_name = self._coerce_terrain_name(terrain_name, terrain_id)
        if self._terrain_model_mode == 'separate':
            self._bind_active_terrain_buffers(terrain_name)
        else:
            self._active_terrain_name = 'shared'
            self._active_terrain_id = terrain_id
        return terrain_name

    def configure_sampling(
            self,
            reward_biased_batch_fraction=None,
            reward_bias_temperature=None,
            reward_bias_step_weight=None,
            reward_bias_episode_weight=None,
    ):
        if reward_biased_batch_fraction is not None:
            self._reward_biased_batch_fraction = float(
                np.clip(float(reward_biased_batch_fraction), 0.0, 1.0)
            )
        if reward_bias_temperature is not None:
            self._reward_bias_temperature = max(1e-6, float(reward_bias_temperature))
        if reward_bias_step_weight is not None:
            self._reward_bias_step_weight = float(reward_bias_step_weight)
        if reward_bias_episode_weight is not None:
            self._reward_bias_episode_weight = float(reward_bias_episode_weight)

    def set_active_terrain_ids(self, terrain_ids):
        if terrain_ids is None:
            self._active_terrain_ids = None
            return
        terrain_ids = list(terrain_ids)
        self._active_terrain_ids = set(map(int, terrain_ids)) if terrain_ids else None


    def add_sample(self, observation, action, reward, terminal,
                   next_observation, env_info=None, **kwargs):
        if env_info is None:
            env_info = {}
        terrain_id = int(env_info.get('terrain_id', -1))
        env_info = dict(env_info)
        env_info['terrain_id'] = np.array([terrain_id], dtype=np.float32)
        for key in self._env_info_sizes:
            if key == 'terrain_id':
                continue
            env_info.setdefault(key, np.zeros(self._env_info_sizes[key], dtype=np.float32))
            env_info[key] = np.asarray(env_info[key], dtype=np.float32).reshape(-1)[:self._env_info_sizes[key]]

        if self._terrain_model_mode == 'separate':
            terrain_name = self._ensure_terrain_buffers(terrain_id=terrain_id)
            individual_buffer = self._individual_buffers[terrain_name]
            population_buffer = self._population_buffers[terrain_name]
            init_state_buffer = self._init_state_buffers[terrain_name]
        else:
            individual_buffer = self._individual_buffer
            population_buffer = self._population_buffer
            init_state_buffer = self._init_state_buffer

        individual_buffer.add_sample(observation=observation, action=action, reward=reward, terminal=terminal, next_observation=next_observation, env_info=env_info, **kwargs)
        population_buffer.add_sample(observation=observation, action=action, reward=reward, terminal=terminal, next_observation=next_observation, env_info=env_info, **kwargs)

        # TODO: What is the point of an intitial state replay buffer?
        if self._expect_init_state:
            init_state_buffer.add_sample(observation=observation, action=action, reward=reward, terminal=terminal, next_observation=next_observation, env_info=env_info, **kwargs)
            init_state_buffer.terminate_episode() # right now terminate episode is a pass but could change
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

    def _sample_candidate_indices(self, buffer, candidate, batch_size):
        candidate = np.asarray(candidate, dtype=np.int64)
        if len(candidate) == 0:
            return candidate

        biased_count = int(round(batch_size * self._reward_biased_batch_fraction))
        biased_count = min(max(biased_count, 0), batch_size)
        uniform_count = batch_size - biased_count

        sampled_parts = []
        if biased_count > 0:
            scores = (
                self._reward_bias_step_weight
                * buffer._rewards[candidate].reshape(-1).astype(np.float64)
            )
            if 'episode_score' in buffer._env_info_keys:
                scores = scores + (
                    self._reward_bias_episode_weight
                    * buffer._env_infos['episode_score'][candidate].reshape(-1).astype(np.float64)
                )
            logits = (scores - np.max(scores)) / self._reward_bias_temperature
            logits = np.clip(logits, -50.0, 0.0)
            weights = np.exp(logits)
            weight_sum = np.sum(weights)
            if not np.isfinite(weight_sum) or weight_sum <= 0.0:
                probabilities = None
            else:
                probabilities = weights / weight_sum
            sampled_parts.append(
                np.random.choice(
                    candidate,
                    size=biased_count,
                    replace=len(candidate) < biased_count,
                    p=probabilities,
                )
            )

        if uniform_count > 0:
            sampled_parts.append(
                np.random.choice(
                    candidate,
                    size=uniform_count,
                    replace=len(candidate) < uniform_count,
                )
            )

        sampled = np.concatenate(sampled_parts) if sampled_parts else np.array([], dtype=np.int64)
        np.random.shuffle(sampled)
        return sampled

    def _balanced_random_batch(self, buffer, batch_size):
        if buffer._size <= 0:
            return buffer.random_batch(batch_size)

        if 'terrain_id' not in buffer._env_info_keys:
            return buffer.random_batch(batch_size)

        terrain_ids = buffer._env_infos['terrain_id'][:buffer._size].reshape(-1).astype(int)
        active_mask = np.ones(buffer._size, dtype=bool)
        if self._active_terrain_ids is not None:
            active_values = np.asarray(sorted(self._active_terrain_ids), dtype=int)
            active_mask = np.isin(terrain_ids, active_values)

        eligible_indices = np.where(active_mask)[0]
        if len(eligible_indices) == 0:
            raise ValueError(
                "No replay samples are available for the active terrain filter."
            )

        valid_terrain_ids = sorted([
            tid for tid in np.unique(terrain_ids[eligible_indices])
            if tid >= 0
        ])
        if not valid_terrain_ids:
            replace = len(eligible_indices) < batch_size
            indices = np.random.choice(eligible_indices, size=batch_size, replace=replace)
            return self._random_batch_from_indices(buffer, indices)

        n_terrains = len(valid_terrain_ids)
        base = batch_size // n_terrains
        remainder = batch_size % n_terrains

        sampled_indices = []
        for i, terrain_id in enumerate(valid_terrain_ids):
            n_take = base + (1 if i < remainder else 0)
            if n_take == 0:
                continue
            candidate = np.where(active_mask & (terrain_ids == terrain_id))[0]
            if len(candidate) == 0:
                continue
            sampled = self._sample_candidate_indices(buffer, candidate, n_take)
            sampled_indices.append(sampled)

        if not sampled_indices:
            replace = len(eligible_indices) < batch_size
            indices = np.random.choice(eligible_indices, size=batch_size, replace=replace)
            return self._random_batch_from_indices(buffer, indices)

        indices = np.concatenate(sampled_indices, axis=0)
        if len(indices) < batch_size:
            # Top up from active terrains only to preserve the requested batch size.
            extra_needed = batch_size - len(indices)
            extra = self._sample_candidate_indices(buffer, eligible_indices, extra_needed)
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

        indices = self._sample_candidate_indices(buffer, candidate, batch_size)
        return self._random_batch_from_indices(buffer, indices)

    def random_start_batch_for_terrain(self, batch_size, terrain_id):
        """Sample a start-state batch from one terrain when available."""
        if self._terrain_model_mode == 'separate':
            terrain_name = self._ensure_terrain_buffers(terrain_id=terrain_id)
            return self._balanced_random_batch(
                self._init_state_buffers[terrain_name],
                batch_size,
            )
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
        elif self._mode == "start":
            return self._init_state_buffer.num_steps_can_sample(**kwargs)
        else:
            raise ValueError(f"Unknown replay buffer mode: {self._mode}")

    def random_batch(self, batch_size):
        """
        Return a batch of size `batch_size`.
        :param batch_size:
        :return:
        """
        if self._mode == "species":
            ind_batch_size = int(round(batch_size * self._individual_batch_fraction))
            ind_batch_size = min(max(ind_batch_size, 0), batch_size)
            pop_batch_size = batch_size - ind_batch_size

            batches = []
            if pop_batch_size > 0:
                batches.append(self._balanced_random_batch(self._population_buffer, pop_batch_size))
            if ind_batch_size > 0:
                batches.append(self._balanced_random_batch(self._individual_buffer, ind_batch_size))

            if len(batches) == 1:
                return batches[0]

            batch = {}
            for key in batches[0]:
                batch[key] = np.concatenate([item[key] for item in batches], axis=0)
            return batch
 
    
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

    
    def reset_individual_buffer(self, terrain_name=None, terrain_id=None):
        if self._terrain_model_mode == 'separate':
            terrain_name = self._coerce_terrain_name(terrain_name, terrain_id)
            self._individual_buffers[terrain_name] = self._new_individual_buffer()
            if terrain_name == self._active_terrain_name:
                self._individual_buffer = self._individual_buffers[terrain_name]
        else:
            self._individual_buffer = self._new_individual_buffer()
            self._individual_buffers['shared'] = self._individual_buffer
        self._ep_counter = 0 # reset number of episodes for next design


