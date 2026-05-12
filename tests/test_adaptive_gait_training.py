import sys
import types
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "CoadaptationCode"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from adaptive_gait_utils import (
    action_delta_mean_and_penalty,
    deterministic_rollout_probability_for_episode,
    episode_replay_score,
    scheduled_target_entropy_for_episode,
)


def test_rollout_schedule_stochastic_then_ramps_to_deterministic():
    assert deterministic_rollout_probability_for_episode(0, 10, 20) == 0.0
    assert deterministic_rollout_probability_for_episode(10, 10, 20) == 0.0
    assert deterministic_rollout_probability_for_episode(20, 10, 20) == 0.5
    assert deterministic_rollout_probability_for_episode(30, 10, 20) == 1.0
    assert deterministic_rollout_probability_for_episode(99, 10, 20) == 1.0


def test_target_entropy_anneals_toward_convergence_value():
    assert scheduled_target_entropy_for_episode(0, -7.0, -2.0, 30) == -7.0
    assert scheduled_target_entropy_for_episode(15, -7.0, -2.0, 30) == -4.5
    assert scheduled_target_entropy_for_episode(30, -7.0, -2.0, 30) == -2.0
    assert scheduled_target_entropy_for_episode(50, -7.0, -2.0, 30) == -2.0


def test_action_delta_penalty_is_zero_first_step_then_scaled():
    action = np.array([0.5, -0.5, 0.0], dtype=np.float32)
    assert action_delta_mean_and_penalty(action, None, 0.03) == (0.0, 0.0)

    previous = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    mean_delta, penalty = action_delta_mean_and_penalty(action, previous, 0.03)
    assert np.isclose(mean_delta, (0.5 + 0.5 + 0.0) / 3.0)
    assert np.isclose(penalty, mean_delta * 0.03)


def test_episode_replay_score_prefers_good_episodes_without_smoothness_by_default():
    good = episode_replay_score(
        mean_reward=0.05,
        positive_reward_fraction=0.70,
        mean_action_delta=0.35,
    )
    twitchy = episode_replay_score(
        mean_reward=0.05,
        positive_reward_fraction=0.70,
        mean_action_delta=0.75,
    )
    bad = episode_replay_score(
        mean_reward=-0.10,
        positive_reward_fraction=0.40,
        mean_action_delta=0.55,
    )
    assert good == twitchy
    assert good > bad

    good_smooth_weighted = episode_replay_score(
        mean_reward=0.05,
        positive_reward_fraction=0.70,
        mean_action_delta=0.35,
        action_delta_weight=1.0,
    )
    twitchy_weighted = episode_replay_score(
        mean_reward=0.05,
        positive_reward_fraction=0.70,
        mean_action_delta=0.75,
        action_delta_weight=1.0,
    )
    assert good_smooth_weighted > twitchy_weighted


def test_reward_biased_replay_sampling_prefers_high_episode_score(monkeypatch):
    rlkit = types.ModuleType("rlkit")
    data_management = types.ModuleType("rlkit.data_management")
    replay_buffer = types.ModuleType("rlkit.data_management.replay_buffer")

    class ReplayBuffer:
        pass

    replay_buffer.ReplayBuffer = ReplayBuffer
    monkeypatch.setitem(sys.modules, "rlkit", rlkit)
    monkeypatch.setitem(sys.modules, "rlkit.data_management", data_management)
    monkeypatch.setitem(sys.modules, "rlkit.data_management.replay_buffer", replay_buffer)
    torch = types.ModuleType("torch")
    torch.save = lambda *args, **kwargs: None
    torch.load = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "torch", torch)

    gymnasium = types.ModuleType("gymnasium")
    spaces = types.ModuleType("gymnasium.spaces")

    class Box:
        def __init__(self, low, high, shape, dtype):
            self.low = np.full(shape, low, dtype=dtype)
            self.high = np.full(shape, high, dtype=dtype)
            self.shape = shape
            self.dtype = dtype

    class Discrete:
        def __init__(self, n):
            self.n = n

    class Tuple:
        def __init__(self, spaces_):
            self.spaces = spaces_

    class Dict:
        def __init__(self, spaces_):
            self.spaces = spaces_

    spaces.Box = Box
    spaces.Discrete = Discrete
    spaces.Tuple = Tuple
    spaces.Dict = Dict
    gymnasium.spaces = spaces
    monkeypatch.setitem(sys.modules, "gymnasium", gymnasium)
    monkeypatch.setitem(sys.modules, "gymnasium.spaces", spaces)

    from replaybuffercoadapt import CoadaptReplayBuffer

    class FakeEnv:
        observation_space = Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        action_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    np.random.seed(7)
    replay = CoadaptReplayBuffer(1000, 1000, FakeEnv())
    replay.configure_sampling(
        reward_biased_batch_fraction=1.0,
        reward_bias_temperature=0.1,
        reward_bias_step_weight=1.0,
        reward_bias_episode_weight=1.0,
    )

    for _ in range(50):
        replay.add_sample(
            observation=np.zeros(3),
            action=np.zeros(1),
            reward=np.array([-0.2]),
            terminal=np.array([False]),
            next_observation=np.zeros(3),
            env_info={'terrain_id': 0, 'episode_score': -0.8},
        )
    for _ in range(5):
        replay.add_sample(
            observation=np.ones(3),
            action=np.ones(1),
            reward=np.array([0.8]),
            terminal=np.array([False]),
            next_observation=np.ones(3),
            env_info={'terrain_id': 0, 'episode_score': 0.8},
        )

    replay.set_mode('population')
    batch = replay.random_batch(64)
    sampled_scores = batch['episode_score'].reshape(-1)
    assert sampled_scores.mean() > 0.5
