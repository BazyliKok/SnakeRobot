from gym.spaces import Discrete

from simple_replay_buffer import SimpleReplayBuffer
import numpy as np
from gymnasium.spaces import Box, Discrete, Tuple


class EnvReplayBuffer(SimpleReplayBuffer):
    def __init__(
            self,
            max_replay_buffer_size,
            env,
            env_info_sizes=None
    ):
        """
        :param max_replay_buffer_size:
        :param env:
        """
        self.env = env
        self._ob_space = env.observation_space
        self._action_space = env.action_space
      
        if env_info_sizes is None:
            if hasattr(env, 'info_sizes'):
                env_info_sizes = env.info_sizes
            else:
                env_info_sizes = dict()



        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            observation_dim=self.get_dim(self._ob_space),
            action_dim=self.get_dim(self._action_space),
            env_info_sizes=env_info_sizes
        )

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        if isinstance(self._action_space, Discrete):
            new_action = np.zeros(self._action_dim)
            new_action[action] = 1
        else:
            new_action = action
        return super().add_sample(
            observation=observation,
            action=new_action,
            reward=reward,
            next_observation=next_observation,
            terminal=terminal,
            **kwargs
        )
    
    def get_dim(self,space):
        if isinstance(space, Box):
            return space.low.size
        elif isinstance(space, Discrete):
            return space.n
        elif isinstance(space, Tuple):
            return sum(self.get_dim(subspace) for subspace in space.spaces)
        elif hasattr(space, 'flat_dim'):
            return space.flat_dim
        else:
            raise TypeError("Unknown space: {}".format(space))
