import numpy as np


def deterministic_rollout_probability_for_episode(
        episode_in_block,
        start_episode=10,
        ramp_episodes=20,
):
    episode_in_block = int(max(0, episode_in_block))
    start_episode = int(max(0, start_episode))
    ramp_episodes = int(max(0, ramp_episodes))
    if episode_in_block <= start_episode:
        return 0.0
    if ramp_episodes == 0:
        return 1.0
    return float(np.clip((episode_in_block - start_episode) / ramp_episodes, 0.0, 1.0))


def scheduled_target_entropy_for_episode(
        episode_in_block,
        start_entropy=-7.0,
        end_entropy=-2.0,
        anneal_episodes=30,
):
    episode_in_block = int(max(0, episode_in_block))
    anneal_episodes = max(1, int(anneal_episodes))
    fraction = float(np.clip(episode_in_block / anneal_episodes, 0.0, 1.0))
    return float(start_entropy + fraction * (end_entropy - start_entropy))


def action_delta_mean_and_penalty(action, previous_action, penalty_scale):
    if previous_action is None:
        return 0.0, 0.0
    action = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)
    previous_action = np.clip(
        np.asarray(previous_action, dtype=np.float32).reshape(-1)[:len(action)],
        -1.0,
        1.0,
    )
    action_delta_mean = float(np.mean(np.abs(action - previous_action)))
    return action_delta_mean, float(max(0.0, penalty_scale) * action_delta_mean)


def episode_replay_score(mean_reward, positive_reward_fraction, mean_action_delta, action_delta_weight=0.0):
    return float(mean_reward + positive_reward_fraction - (action_delta_weight * mean_action_delta))
