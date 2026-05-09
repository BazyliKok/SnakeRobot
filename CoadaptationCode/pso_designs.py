import os
import sys
import torch
import numpy as np
from pso_batch import PSO_batch
from snakeenv_thread_coadapt import SnakeEnv
from replaybuffercoadapt import CoadaptReplayBuffer
from soft_actor_critic_coadapt import SoftActorCriticCoadapt
import rlkit.torch.pytorch_util as ptu

ptu.set_gpu_mode(False)

RESULT_TAGS = ("mixed_terrain", *SnakeEnv.terrains, "carton")

def _torch_load(path, **kwargs):
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError as exc:
        if 'weights_only' not in str(exc):
            raise
        return torch.load(path, **kwargs)


def identity(x):
    return x


# monkey patch for older rlkit compatibility
import rlkit.torch.networks
rlkit.torch.networks.identity = identity


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


env = SnakeEnv()
obs_dim = env.observation_space.low.size
action_dim = env.action_space.low.size

pop_replay = CoadaptReplayBuffer(
    max_replay_buffer_size_species=int(1e6),
    max_replay_buffer_size_population=int(1e7),
    env=env,
    env_info_sizes=None
)

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

print("Loading selected episodes into population buffer...")
episode_length = 175
expected_obs_dim = env.observation_space.low.size
expected_action_dim = env.action_space.low.size
loaded_samples = 0
for path in REPLAY_PATHS:
    resolved_path, replay_payload = _load_replay_buffer(path)
    print(f"Loading replay: {resolved_path}")
    num_samples = int(replay_payload.get('_size', len(replay_payload['observations'])))
    observations = replay_payload['observations']
    actions = replay_payload['actions']
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

    for ep in range(15, 31):
        start_idx = ep * episode_length
        end_idx = min((ep + 1) * episode_length, num_samples)
        for i in range(start_idx, end_idx):
            pop_replay.add_sample(
                observation=observations[i],
                action=actions[i],
                reward=replay_payload['rewards'][i],
                next_observation=replay_payload['next_observations'][i],
                terminal=replay_payload['terminals'][i],
                env_info=_terrain_env_info(replay_payload, i)
            )
            loaded_samples += 1
print("Done loading selected episodes.")

if loaded_samples == 0:
    raise RuntimeError(
        "No compatible replay samples were loaded. Use replay generated with the current "
        "scale-parameter observation/action spaces before running pso_designs.py."
    )

design_dim = len(SnakeEnv.design_parameter_bounds)
obs_dim = env.observation_space.low.size
action_dim = env.action_space.low.size
design_mode = os.getenv("SNAKE_SCALE_DESIGN_MODE", "heterogeneous").strip().lower()
active_terrains = [
    terrain.strip()
    for terrain in os.getenv("SNAKE_ACTIVE_TERRAINS", "carpet,foam").split(",")
    if terrain.strip()
]
pop_replay.set_active_terrain_ids([SnakeEnv.get_terrain_id(terrain) for terrain in active_terrains])

networks = SoftActorCriticCoadapt.create_networks(env)
q_network = networks["population"]["qf1"]
policy_network = networks["population"]["policy"]

print("env base obs dim:", env.observation_space.low.size)
print("design dim:", design_dim)
print("scale design mode:", design_mode)
print("active terrains:", active_terrains)
print("scale parameter bounds:", SnakeEnv.design_parameter_bounds)
print("total obs dim (used):", obs_dim)
print("action dim:", action_dim)
print("Q input dim:", obs_dim + action_dim)

q_state_path = "pop_qf1_epoch.pt"
policy_state_path = "pop_policy.pt"
if not os.path.exists(q_state_path) or not os.path.exists(policy_state_path):
    raise FileNotFoundError(
        "Expected pop_qf1_epoch.pt and pop_policy.pt from offline_pop.py. "
        "Run offline_pop.py with compatible replay first, or point this script at current checkpoints."
    )

q_network.load_state_dict(_torch_load(q_state_path, map_location=ptu.device))
policy_network.load_state_dict(_torch_load(policy_state_path, map_location=ptu.device))

#run pso
pso = PSO_batch(pop_replay, env)
init_design = SnakeEnv.get_default_design()

print("Running PSO...")
cost, best_design = pso.optimize_design(
    design=init_design,
    q_network=q_network,
    policy_network=policy_network,
    active_terrains=active_terrains,
    design_mode=design_mode,
)

print("Best morphology parameters:", best_design)
print("Final cost:", cost)
