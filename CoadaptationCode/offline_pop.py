import os
import torch
from SACTrainer import SACTrainer
from soft_actor_critic_coadapt import SoftActorCriticCoadapt
from replaybuffercoadapt import CoadaptReplayBuffer
from snakeenv_thread_coadapt import SnakeEnv
import rlkit.torch.pytorch_util as ptu
import numpy as np


class WrappedSnakeEnv(SnakeEnv):
    def __init__(self, design_dim=None):
        super().__init__()
        self._design_dim = (
            len(SnakeEnv.design_parameter_bounds)
            if design_dim is None
            else design_dim
        )


RESULT_TAGS = ("mixed_terrain", *SnakeEnv.terrains, "carton")

def _torch_load(path, **kwargs):
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError as exc:
        if 'weights_only' not in str(exc):
            raise
        return torch.load(path, **kwargs)


def _candidate_replay_paths(path):
    yield path

    basename = os.path.basename(path)
    stem, ext = os.path.splitext(basename)
    matched_tag = next((tag for tag in RESULT_TAGS if stem.endswith(f"_{tag}")), None)
    base_stem = stem[:-(len(matched_tag) + 1)] if matched_tag else stem

    for folder in ("results_bazyli", "replay"):
        yield os.path.join(folder, basename)
        if matched_tag is not None:
            for tag in RESULT_TAGS:
                yield os.path.join(folder, f"{base_stem}_{tag}{ext}")


def _resolve_replay_path(path):
    seen = set()
    for candidate in _candidate_replay_paths(path):
        normalized = os.path.normpath(candidate)
        if normalized in seen:
            continue
        seen.add(normalized)
        if os.path.exists(normalized):
            return normalized
    raise FileNotFoundError(f"Replay file not found for '{path}'. Tried: {sorted(seen)}")


def _load_replay_buffer(path, preferred_buffer="population_buffer"):
    resolved_path = _resolve_replay_path(path)
    payload = _torch_load(resolved_path, map_location=torch.device("cpu"))
    if isinstance(payload, dict) and "buffer" in payload:
        payload = payload["buffer"]
    if preferred_buffer in payload:
        payload = payload[preferred_buffer]
    elif "individual_buffer" in payload:
        payload = payload["individual_buffer"]
    return resolved_path, payload


def _terrain_env_info(buffer_payload, index):
    env_infos = buffer_payload.get("env_infos", {})
    terrain_ids = env_infos.get("terrain_id")
    if terrain_ids is None:
        return {}
    terrain_id = int(np.asarray(terrain_ids[index]).reshape(-1)[0])
    return {"terrain_id": terrain_id}


ptu.set_gpu_mode(False)

REPLAY_PATHS = [
    "replay/replay_2025_06_03_Design0_carpet.pt",
    "replay/replay_2025_06_02_Design2_carton.pt",
    "replay/replay_2025_06_02_Design0_foam.pt",
    "replay/replay_2025_06_02_Design0_carton.pt",
    "replay/replay_2025_05_30_Design1_carton.pt",
    "replay/replay_2025_05_30_Design1_carpet.pt",
    "replay/replay_2025_05_26_Design2_foam.pt",
    "replay/replay_2025_05_26_Design1_carton.pt",
    "replay/replay_2025_05_26_Design1_carton.pt",
    "replay/replay_2025_06_18_Design4_carpet.pt",
    "replay/replay_2025_06_17_Design4_carton.pt",
    "replay/replay_2025_06_12_Design4_foam.pt",
    "replay/replay_2025_06_16_Design5_carton.pt",
    "replay/replay_2025_06_18_Design5_carpet.pt",
    "replay/replay_2025_06_16_Design5_foam.pt",
]

env = WrappedSnakeEnv()

networks = SoftActorCriticCoadapt.create_networks(env)

pop_replay = CoadaptReplayBuffer(
    max_replay_buffer_size_species=int(1e6),
    max_replay_buffer_size_population=int(1e7),
    env=env,
    env_info_sizes=None
)

episode_length = 175
design_dim = len(SnakeEnv.design_parameter_bounds)
expected_obs_dim = env.observation_space.low.size
expected_action_dim = env.action_space.low.size
loaded_samples = 0

for path in REPLAY_PATHS:
    resolved_path, replay_payload = _load_replay_buffer(path)
    print(f"Loading replay: {resolved_path}")
    num_samples = int(replay_payload.get('_size', len(replay_payload['observations'])))
    observations = replay_payload['observations']
    actions = replay_payload['actions']
    rewards = replay_payload['rewards']
    next_obs = replay_payload['next_observations']
    terminals = replay_payload['terminals']

    replay_obs_dim = observations.shape[1] if observations.ndim > 1 else 0
    replay_action_dim = actions.shape[1] if actions.ndim > 1 else 0
    if replay_obs_dim != expected_obs_dim or replay_action_dim != expected_action_dim:
        print(
            "Skipping incompatible replay {}: obs/action dims {}/{} do not match current {}/{}.".format(
                resolved_path,
                replay_obs_dim,
                replay_action_dim,
                expected_obs_dim,
                expected_action_dim,
            )
        )
        continue

    for ep in range(17, 31):
        start = ep * episode_length
        end = min((ep + 1) * episode_length, num_samples)
        for i in range(start, end):
            pop_replay.add_sample(
                observation=observations[i],
                action=actions[i],
                reward=rewards[i],
                next_observation=next_obs[i],
                terminal=terminals[i],
                env_info=_terrain_env_info(replay_payload, i)
            )
            loaded_samples += 1

if loaded_samples == 0:
    raise RuntimeError(
        "No compatible replay samples were loaded. Use replay generated with the current "
        "scale-parameter observation/action spaces before running offline_pop.py."
    )

trainer = SoftActorCriticCoadapt(
    env=env,
    replay=pop_replay,
    networks=networks,
)

trainer._replay.set_mode("population")
trainer._nmbr_pop_updates = 500

max_epochs = 500

for epoch in range(max_epochs):
    print(f"\nEpoch {epoch + 1}")
    _, _, _, popQ1, popQ2, popPol = trainer.single_train_step(train_ind=False, train_pop=True)

    # total_loss = abs(popQ1[0]) + abs(popQ2[0]) + abs(popPol[0])
    print(f"Policy loss: {abs(popPol[0])}")

#save models
pop_policy = networks["population"]["policy"]
pop_qf1 = networks["population"]["qf1"]

torch.save(pop_policy.state_dict(), "pop_policy.pt")
torch.save(pop_qf1.state_dict(), "pop_qf1_epoch.pt")

print("models saved")
