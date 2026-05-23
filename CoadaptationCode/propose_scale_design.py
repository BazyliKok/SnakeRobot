import argparse
import csv
import os
import sys
from datetime import datetime

import numpy as np
import torch


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results_bazyli")
DESIGN_VALUE_MODEL_DIR = os.path.join(RESULTS_DIR, "design_value_models")

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

try:
    import rlkit.torch.networks as rlkit_networks

    def identity(x):
        return x

    rlkit_networks.identity = identity
    if not hasattr(rlkit_networks, "FlattenMlp") and hasattr(rlkit_networks, "ConcatMlp"):
        rlkit_networks.FlattenMlp = rlkit_networks.ConcatMlp
except ImportError:
    pass

from snakeenv_thread_coadapt import SnakeEnv


TERRAINS = ["cardboard", "carpet"]
ROBUSTNESS_LAMBDA = 0.5
CHECKPOINT_DESIGN_PARAMETER_BOUNDS = [
    (0.45, 0.90),
    (0.0, 30.0),
    (0.45, 0.90),
    (0.0, 30.0),
]

INITIAL_DESIGN_SOURCES = [
    {
        "index": 0,
        "design": [0.63, 0.0, 0.63, 0.0],
        "checkpoint": "2026_05_18_DesignCycle0_ep76",
        "tag": "scale_ab_carpet_cardboard",
        "replay": "replay_2026_05_18_DesignCycle0_scale_ab_carpet_cardboard.pt",
    },
    {
        "index": 1,
        "design": [0.63, 30.0, 0.63, 30.0],
        "checkpoint": "2026_05_20_DesignCycle1_ep60",
        "tag": "scale_ab_carpet_cardboard_design1_30ep",
        "replay": "replay_2026_05_20_DesignCycle1_scale_ab_carpet_cardboard_design1_30ep.pt",
    },
    {
        "index": 2,
        "design": [0.90, 30.0, 0.90, 30.0],
        "checkpoint": "2026_05_21_DesignCycle2_ep60",
        "tag": "scale_ab_carpet_cardboard_design2_30ep",
        "replay": "replay_2026_05_21_DesignCycle2_scale_ab_carpet_cardboard_design2_30ep.pt",
    },
    {
        "index": 3,
        "design": [0.90, 0.0, 0.90, 0.0],
        "checkpoint": "2026_05_21_DesignCycle3_ep60",
        "tag": "scale_ab_carpet_cardboard_design3_30ep",
        "replay": "replay_2026_05_21_DesignCycle3_scale_ab_carpet_cardboard_design3_30ep.pt",
    },
]


HETEROGENEOUS_DESIGN_SOURCES = [
    {
        "index": 0,
        "design": [0.63, 0.0, 0.90, 30.0],
        "checkpoint": "2026_05_22_DesignCycle0_ep60",
        "tag": "scale_ab_carpet_cardboard_heterogeneous_design0_30ep",
        "replay": "replay_2026_05_22_DesignCycle0_scale_ab_carpet_cardboard_heterogeneous_design0_30ep.pt",
        "reason": "heterogeneous A=(0.63, 0 deg), B=(0.90, 30 deg)",
    },
    {
        "index": 1,
        "design": [0.90, 30.0, 0.63, 0.0],
        "checkpoint": "2026_05_23_DesignCycle1_ep59",
        "tag": "scale_ab_carpet_cardboard_heterogeneous_design1_30ep",
        "replay": "replay_2026_05_23_DesignCycle1_scale_ab_carpet_cardboard_heterogeneous_design1_30ep.pt",
        "reason": "heterogeneous A=(0.90, 30 deg), B=(0.63, 0 deg); resumed run, filtered by design vector",
    },
    {
        "index": 2,
        "design": [0.63, 30.0, 0.90, 0.0],
        "checkpoint": "2026_05_23_DesignCycle2_ep60",
        "tag": "scale_ab_carpet_cardboard_heterogeneous_design2_30ep",
        "replay": "replay_2026_05_23_DesignCycle2_scale_ab_carpet_cardboard_heterogeneous_design2_30ep.pt",
        "reason": "heterogeneous A=(0.63, 30 deg), B=(0.90, 0 deg)",
    },
]


def design_sources_for_mode(design_mode):
    if design_mode == "homogeneous":
        return INITIAL_DESIGN_SOURCES
    if design_mode == "heterogeneous":
        return HETEROGENEOUS_DESIGN_SOURCES
    raise ValueError(f"Unsupported design mode: {design_mode}")


def load_trusted_checkpoint(path, map_location="cpu"):
    try:
        return torch.load(path, weights_only=False, map_location=map_location)
    except TypeError as exc:
        if "weights_only" not in str(exc):
            raise
        return torch.load(path, map_location=map_location)


def load_replay(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return load_trusted_checkpoint(path)


def coerce_design_vector(design, bounds=None):
    if bounds is None:
        bounds = SnakeEnv.design_parameter_bounds
    values = np.asarray(design, dtype=float).reshape(-1).tolist()
    if len(values) < len(bounds):
        values += list(SnakeEnv.current_design)[len(values):len(bounds)]
    return [float(np.clip(value, low, high)) for value, (low, high) in zip(values[:len(bounds)], bounds)]


def design_key(design, design_bounds=None):
    full_design = coerce_design_vector(design, design_bounds)
    return tuple(round(float(value), 3 if i % 2 == 0 else 2) for i, value in enumerate(full_design))


def actual_width(width_ratio):
    return SnakeEnv.actual_width_from_ratio(float(width_ratio))


def robust_score(values, robustness_lambda=ROBUSTNESS_LAMBDA):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.nan
    return float(values.mean() - robustness_lambda * values.std())


def active_indices(buffer):
    size = int(buffer.get("_size", 0))
    if size <= 0:
        return np.array([], dtype=np.int64)
    max_size = int(buffer.get("_max_replay_buffer_size", size))
    top = int(buffer.get("_top", size))
    if size < max_size and top == size:
        return np.arange(size, dtype=np.int64)
    return np.concatenate([np.arange(top, max_size), np.arange(0, top)])[:size].astype(np.int64)


def decode_design_from_observation(observation, design_bounds=None):
    if design_bounds is None:
        design_bounds = SnakeEnv.design_parameter_bounds
    observation = np.asarray(observation, dtype=np.float32).reshape(-1)
    design_dims = [int(i) for i in SnakeEnv.get_design_dimensions() if 0 <= int(i) < len(observation)]
    feature_count = len(SnakeEnv.config_numpy)
    if len(design_dims) != feature_count:
        design_dims = list(range(len(observation) - feature_count, len(observation)))

    decoded = []
    for value, (low, high) in zip(observation[design_dims][:len(design_bounds)], design_bounds):
        clipped = float(np.clip(value, -1.0, 1.0))
        decoded.append(float(low + 0.5 * (clipped + 1.0) * (high - low)))
    return coerce_design_vector(decoded, design_bounds)


def finish_rollout(rollouts, current):
    if current["length"] <= 0 or current["design"] is None:
        return
    rollouts.append({
        "terrain": current["terrain"],
        "design": current["design"],
        "design_key": design_key(current["design"], current["design_bounds"]),
        "return": float(current["return"]),
        "length": int(current["length"]),
    })
    current.update({"return": 0.0, "length": 0, "terrain": None, "design": None, "design_key": None})


def rollouts_from_buffer(buffer, default_terrain, design_bounds=None):
    indices = active_indices(buffer)
    if len(indices) == 0:
        return []

    terrain_info = None
    if "terrain_id" in buffer.get("env_info_keys", []):
        terrain_info = np.asarray(buffer.get("env_infos", {})["terrain_id"]).reshape(-1)

    current = {
        "return": 0.0,
        "length": 0,
        "terrain": None,
        "design": None,
        "design_key": None,
        "design_bounds": design_bounds,
    }
    rollouts = []

    for idx in indices:
        terrain = default_terrain
        if terrain_info is not None:
            terrain = SnakeEnv.terrain_id_to_name.get(int(terrain_info[idx]), default_terrain)

        design = decode_design_from_observation(buffer["observations"][idx], design_bounds)
        key = design_key(design, design_bounds)

        if current["length"] > 0 and (current["terrain"] != terrain or current["design_key"] != key):
            finish_rollout(rollouts, current)

        if current["length"] == 0:
            current.update({"terrain": terrain, "design": design, "design_key": key})

        current["return"] += float(np.asarray(buffer["rewards"][idx]).reshape(-1)[0])
        current["length"] += 1

        if bool(np.asarray(buffer["terminals"][idx]).reshape(-1)[0]):
            finish_rollout(rollouts, current)

    finish_rollout(rollouts, current)
    return rollouts


def optimizer_parameter_names(design_mode):
    if design_mode == "homogeneous":
        return ["width_ratio", "attack_angle_deg"]
    if design_mode == "heterogeneous":
        return ["A_width_ratio", "A_attack_angle_deg", "B_width_ratio", "B_attack_angle_deg"]
    raise ValueError(f"Unsupported design mode: {design_mode}")


def checkpoint_optimization_bounds(design_mode):
    if design_mode == "homogeneous":
        return CHECKPOINT_DESIGN_PARAMETER_BOUNDS[:2]
    if design_mode == "heterogeneous":
        return CHECKPOINT_DESIGN_PARAMETER_BOUNDS
    raise ValueError(f"Unsupported design mode: {design_mode}")


def search_bounds(design_mode, proposal_mode):
    if proposal_mode == "safe":
        return checkpoint_optimization_bounds(design_mode)
    return [(float(low), float(high)) for low, high in SnakeEnv.get_optimization_bounds(design_mode)]


def expand_design_candidate(candidate, design_mode):
    candidate = np.asarray(candidate, dtype=np.float64).reshape(-1)
    if design_mode == "homogeneous":
        if candidate.size < 2:
            raise ValueError("Homogeneous candidates require [width_ratio, attack_angle_deg].")
        try:
            return np.asarray(SnakeEnv.expand_optimization_design(candidate[:2], "homogeneous"), dtype=np.float64)
        except AttributeError:
            return np.asarray([candidate[0], candidate[1], candidate[0], candidate[1]], dtype=np.float64)
    if design_mode == "heterogeneous":
        if candidate.size != 4:
            raise ValueError("Heterogeneous candidates require [A_width, A_angle, B_width, B_angle].")
        return np.asarray(coerce_design_vector(candidate, SnakeEnv.get_optimization_bounds("heterogeneous")), dtype=np.float64)
    raise ValueError(f"Unsupported design mode: {design_mode}")


def candidate_from_full_design(full_design, design_mode):
    full_design = np.asarray(full_design, dtype=np.float64).reshape(-1)
    if design_mode == "homogeneous":
        return np.asarray([full_design[0], full_design[1]], dtype=np.float64)
    return np.asarray(full_design[:4], dtype=np.float64)


def normalize_candidate(candidate, bounds):
    candidate = np.asarray(candidate, dtype=np.float64).reshape(-1)
    lows = np.asarray([b[0] for b in bounds], dtype=np.float64)
    highs = np.asarray([b[1] for b in bounds], dtype=np.float64)
    return np.clip((candidate - lows) / np.maximum(highs - lows, 1e-9), 0.0, 1.0)


def nearest_known_distance(candidate, known_candidates, bounds):
    if not known_candidates:
        return np.inf
    x = normalize_candidate(candidate, bounds)
    scale = np.sqrt(max(len(x), 1))
    return float(min(np.linalg.norm(x - normalize_candidate(k, bounds)) / scale for k in known_candidates))


def measured_space_distance(candidate, known_candidates, bounds):
    if not known_candidates:
        return 0.0
    x = normalize_candidate(candidate, bounds)
    known = np.asarray([normalize_candidate(k, bounds) for k in known_candidates], dtype=np.float64)
    below = np.maximum(known.min(axis=0) - x, 0.0)
    above = np.maximum(x - known.max(axis=0), 0.0)
    return float(np.linalg.norm(below + above) / np.sqrt(max(len(x), 1)))


def boundary_score(candidate, bounds, margin):
    x = normalize_candidate(candidate, bounds)
    low = np.maximum(float(margin) - x, 0.0)
    high = np.maximum(x - (1.0 - float(margin)), 0.0)
    return float(np.max((low + high) / max(float(margin), 1e-9)))


def source_rows(robustness_lambda, design_mode):
    rows = []
    for source in design_sources_for_mode(design_mode):
        replay = load_replay(os.path.join(RESULTS_DIR, source["replay"]))
        design_bounds = source.get("design_bounds", CHECKPOINT_DESIGN_PARAMETER_BOUNDS)
        full_design = coerce_design_vector(source["design"], design_bounds)
        target_key = design_key(full_design, design_bounds)
        returns = {terrain: [] for terrain in TERRAINS}

        terrain_buffers = replay.get("population_buffers_by_terrain")
        if terrain_buffers is None:
            terrain_buffers = {TERRAINS[0]: replay["population_buffer"]}

        for default_terrain, buffer in terrain_buffers.items():
            if default_terrain not in TERRAINS:
                continue
            for rollout in rollouts_from_buffer(buffer, default_terrain, design_bounds):
                if rollout["design_key"] == target_key and rollout["terrain"] in TERRAINS:
                    returns[rollout["terrain"]].append(rollout["return"])

        terrain_means = {t: float(np.mean(v)) if v else np.nan for t, v in returns.items()}
        terrain_values = [terrain_means[t] for t in TERRAINS if np.isfinite(terrain_means[t])]
        rows.append({
            "design_index": int(source["index"]),
            "candidate": candidate_from_full_design(full_design, design_mode),
            "full_design": np.asarray(full_design, dtype=np.float64),
            "terrain_means": terrain_means,
            "robustness": robust_score(terrain_values, robustness_lambda),
        })
    return rows


def resolve_model_path(model_spec):
    if model_spec != "latest":
        return os.path.abspath(model_spec)
    if not os.path.isdir(DESIGN_VALUE_MODEL_DIR):
        raise FileNotFoundError(DESIGN_VALUE_MODEL_DIR)
    candidates = [
        os.path.join(DESIGN_VALUE_MODEL_DIR, name)
        for name in os.listdir(DESIGN_VALUE_MODEL_DIR)
        if name.endswith("_design_value_model.joblib")
    ]
    if not candidates:
        raise FileNotFoundError(f"No *_design_value_model.joblib files in {DESIGN_VALUE_MODEL_DIR}")
    return max(candidates, key=os.path.getmtime)


def load_design_value_model(model_spec):
    import joblib

    path = resolve_model_path(model_spec)
    bundle = joblib.load(path)
    if "model" not in bundle or "feature_bounds" not in bundle:
        raise RuntimeError(f"Invalid design-value model bundle: {path}")
    bundle["path"] = path
    return bundle


def encode_features(candidate, terrain, bundle):
    bounds = bundle["feature_bounds"]
    x = normalize_candidate(candidate, bounds).tolist()
    x += [1.0 if terrain == item else 0.0 for item in bundle.get("terrains", TERRAINS)]
    return x


def model_predict(model, x):
    mean, std = model.predict(np.asarray(x, dtype=np.float64), return_std=True)
    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    if bool(getattr(model, "_snake_target_normalized", False)):
        target_mean = float(getattr(model, "_snake_target_mean", 0.0))
        target_std = float(getattr(model, "_snake_target_std", 1.0))
        mean = mean * target_std + target_mean
        std = std * target_std
    return mean, std


def predict_candidates(candidates, bundle, robustness_lambda):
    candidates = np.asarray(candidates, dtype=np.float64).reshape(-1, len(bundle["feature_bounds"]))
    model = bundle["model"]
    terrain_means = {}
    terrain_stds = {}

    for terrain in bundle.get("terrains", TERRAINS):
        x = [encode_features(candidate, terrain, bundle) for candidate in candidates]
        mean, std = model_predict(model, x)
        terrain_means[terrain] = mean
        terrain_stds[terrain] = std

    mean_matrix = np.vstack([terrain_means[t] for t in TERRAINS])
    std_matrix = np.vstack([terrain_stds[t] for t in TERRAINS])
    robustness = mean_matrix.mean(axis=0) - float(robustness_lambda) * mean_matrix.std(axis=0)
    uncertainty = np.sqrt(np.mean(std_matrix ** 2, axis=0) + np.std(mean_matrix, axis=0) ** 2)
    return terrain_means, terrain_stds, robustness, uncertainty


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def acquisition_score(candidate, bundle, known_candidates, args):
    candidate = np.asarray(candidate, dtype=np.float64).reshape(1, -1)
    terrain_means, terrain_stds, robustness, uncertainty = predict_candidates(
        candidate,
        bundle,
        args.robustness_lambda,
    )
    candidate = candidate[0]
    bounds = search_bounds(args.design_mode, args.proposal_mode)
    distance = nearest_known_distance(candidate, known_candidates, bounds)
    target = np.exp(-0.5 * ((distance - args.target_novelty_distance) / max(args.target_novelty_width, 1e-9)) ** 2)
    gate = sigmoid((robustness[0] - args.value_floor) / max(args.gate_temperature, 1e-9))
    b_penalty = boundary_score(candidate, bounds, args.boundary_margin)
    e_penalty = measured_space_distance(candidate, known_candidates, bounds)

    score = (
        args.uncertainty_weight * uncertainty[0] * gate
        + args.mean_weight * robustness[0]
        + args.target_novelty_weight * target
        - args.boundary_penalty * b_penalty
        - args.extrapolation_penalty * e_penalty
    )
    return float(score), {
        "design_value_robustness": float(robustness[0]),
        "design_value_uncertainty": float(uncertainty[0]),
        "target_novelty_score": float(target),
        "novelty_distance": float(distance),
        "performance_gate": float(gate),
        "boundary_score": float(b_penalty),
        "extrapolation_distance": float(e_penalty),
        **{f"{terrain}_mean": float(terrain_means[terrain][0]) for terrain in TERRAINS},
        **{f"{terrain}_std": float(terrain_stds[terrain][0]) for terrain in TERRAINS},
    }


def run_pso(objective, bounds, args):
    rng = np.random.default_rng(args.seed)
    lows = np.asarray([b[0] for b in bounds], dtype=np.float64)
    highs = np.asarray([b[1] for b in bounds], dtype=np.float64)
    span = highs - lows
    dim = len(bounds)
    velocity_limit = float(args.velocity_clamp) * span

    positions = rng.uniform(lows, highs, size=(args.particles, dim))
    velocities = rng.uniform(-velocity_limit, velocity_limit, size=(args.particles, dim))
    scores = np.asarray([objective(position)[0] for position in positions], dtype=np.float64)

    personal_best = positions.copy()
    personal_scores = scores.copy()
    best_idx = int(np.argmax(scores))
    global_best = positions[best_idx].copy()
    global_score = float(scores[best_idx])
    best_path = [global_best.copy()]
    history = [global_score]
    particle_history = [positions.copy()]
    score_history = [scores.copy()]

    for _ in range(args.iters):
        r1 = rng.random(size=(args.particles, dim))
        r2 = rng.random(size=(args.particles, dim))
        velocities = (
            args.inertia * velocities
            + args.c1 * r1 * (personal_best - positions)
            + args.c2 * r2 * (global_best - positions)
        )
        velocities = np.clip(velocities, -velocity_limit, velocity_limit)
        positions = np.clip(positions + velocities, lows, highs)
        scores = np.asarray([objective(position)[0] for position in positions], dtype=np.float64)

        improved = scores > personal_scores
        personal_best[improved] = positions[improved]
        personal_scores[improved] = scores[improved]

        best_idx = int(np.argmax(scores))
        if scores[best_idx] > global_score:
            global_best = positions[best_idx].copy()
            global_score = float(scores[best_idx])
        best_path.append(global_best.copy())
        history.append(global_score)
        particle_history.append(positions.copy())
        score_history.append(scores.copy())

    return {
        "best_position": global_best,
        "best_score": global_score,
        "best_path": np.asarray(best_path),
        "history": np.asarray(history),
        "particle_history": np.asarray(particle_history),
        "score_history": np.asarray(score_history),
    }


def proposal_row(candidate, score, details, args):
    full_design = expand_design_candidate(candidate, args.design_mode)
    row = {
        "created_at": datetime.now().strftime("%Y_%m_%d_%H%M%S"),
        "design_mode": args.design_mode,
        "proposal_objective": "design_value_gated_uncertainty",
        "acquisition_score": float(score),
        **details,
        "A_width_ratio": float(full_design[0]),
        "A_attack_angle_deg": float(full_design[1]),
        "A_actual_width": actual_width(full_design[0]),
        "B_width_ratio": float(full_design[2]),
        "B_attack_angle_deg": float(full_design[3]),
        "B_actual_width": actual_width(full_design[2]),
    }
    for name, value in zip(optimizer_parameter_names(args.design_mode), candidate):
        row[f"optimizer_{name}"] = float(value)
    return row


def save_csv(row, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{row['created_at']}_scale_design_proposal.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    return path


def plot_homogeneous(bundle, known_rows, pso_result, selected, args, output_dir, timestamp):
    if args.no_plot or args.design_mode != "homogeneous":
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    bounds = search_bounds(args.design_mode, args.proposal_mode)
    w = np.linspace(bounds[0][0], bounds[0][1], 80)
    a = np.linspace(bounds[1][0], bounds[1][1], 80)
    mesh_w, mesh_a = np.meshgrid(w, a)
    candidates = np.column_stack([mesh_w.reshape(-1), mesh_a.reshape(-1)])
    known = [row["candidate"] for row in known_rows]

    terrain_means, terrain_stds, robustness, uncertainty = predict_candidates(candidates, bundle, args.robustness_lambda)
    scores = np.asarray([acquisition_score(candidate, bundle, known, args)[0] for candidate in candidates])

    robustness = robustness.reshape(mesh_w.shape)
    uncertainty = uncertainty.reshape(mesh_w.shape)
    scores = scores.reshape(mesh_w.shape)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    axes = axes.reshape(-1)

    flat_particles = pso_result["particle_history"].reshape(-1, 2)
    flat_scores = pso_result["score_history"].reshape(-1)
    axes[0].scatter(flat_particles[:, 0], flat_particles[:, 1], c=flat_scores, s=7, alpha=0.35)
    axes[0].plot(pso_result["best_path"][:, 0], pso_result["best_path"][:, 1], color="black", lw=1.0)
    axes[0].set_title("PSO particles")

    score_plot = axes[1].contourf(mesh_w, mesh_a, scores, levels=24, cmap="viridis")
    fig.colorbar(score_plot, ax=axes[1], label="Acquisition score")
    axes[1].set_title("Gated uncertainty acquisition")

    unc_plot = axes[2].contourf(mesh_w, mesh_a, uncertainty, levels=24, cmap="magma")
    fig.colorbar(unc_plot, ax=axes[2], label="Predictive uncertainty")
    axes[2].set_title("Predictive uncertainty")

    axes[3].plot(pso_result["history"], color="crimson")
    axes[3].set_title("Best acquisition score history")
    axes[3].set_xlabel("Iteration")
    axes[3].set_ylabel("Best score")
    axes[3].grid(True, alpha=0.25)

    best_measured = max(known_rows, key=lambda row: row["robustness"])
    for axis in axes[:3]:
        for row in known_rows:
            x, y = row["candidate"][:2]
            axis.scatter([x], [y], s=80, marker="o", facecolors="none", edgecolors="black", linewidths=1.2)
            axis.text(x, y, f" {row['design_index']}", fontsize=8)
        axis.scatter([best_measured["candidate"][0]], [best_measured["candidate"][1]], marker="D", s=70, color="tab:blue")
        axis.scatter([selected[0]], [selected[1]], marker="*", s=180, color="orange", edgecolors="black", linewidths=0.7)
        axis.set_xlabel("Width ratio")
        axis.set_ylabel("Attack angle (deg)")
        axis.grid(True, alpha=0.25)

    output_path = os.path.join(output_dir, f"{timestamp}_scale_design_pso.png")
    fig.suptitle(
        f"Scale design proposal | score={pso_result['best_score']:.3f} | "
        f"selected=({selected[0]:.3f}, {selected[1]:.2f})",
        fontsize=11,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _mark_projection(axis, known_rows, selected, dims, x_label, y_label):
    best_measured = max(known_rows, key=lambda row: row["robustness"])
    for row in known_rows:
        x, y = row["candidate"][dims[0]], row["candidate"][dims[1]]
        axis.scatter([x], [y], s=80, marker="o", facecolors="none", edgecolors="black", linewidths=1.2)
        axis.text(x, y, f" {row['design_index']}", fontsize=8)
    axis.scatter(
        [best_measured["candidate"][dims[0]]],
        [best_measured["candidate"][dims[1]]],
        marker="D",
        s=70,
        color="tab:blue",
        label="best measured",
    )
    axis.scatter(
        [selected[dims[0]]],
        [selected[dims[1]]],
        marker="*",
        s=180,
        color="orange",
        edgecolors="black",
        linewidths=0.7,
        label="selected",
    )
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.grid(True, alpha=0.25)


def plot_heterogeneous(bundle, known_rows, pso_result, selected, args, output_dir, timestamp):
    if args.no_plot or args.design_mode != "heterogeneous":
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    bounds = search_bounds(args.design_mode, args.proposal_mode)
    known = [row["candidate"] for row in known_rows]
    flat_particles = pso_result["particle_history"].reshape(-1, 4)
    flat_scores = pso_result["score_history"].reshape(-1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    axes = axes.reshape(-1)

    particle_plot = axes[0].scatter(
        flat_particles[:, 0],
        flat_particles[:, 1],
        c=flat_scores,
        s=8,
        alpha=0.35,
        cmap="viridis",
    )
    axes[0].plot(pso_result["best_path"][:, 0], pso_result["best_path"][:, 1], color="black", lw=1.0)
    _mark_projection(axes[0], known_rows, selected, (0, 1), "A width ratio", "A attack angle (deg)")
    axes[0].set_title("PSO particles projected to Scale A")
    fig.colorbar(particle_plot, ax=axes[0], label="Acquisition score")

    particle_plot_b = axes[1].scatter(
        flat_particles[:, 2],
        flat_particles[:, 3],
        c=flat_scores,
        s=8,
        alpha=0.35,
        cmap="viridis",
    )
    axes[1].plot(pso_result["best_path"][:, 2], pso_result["best_path"][:, 3], color="black", lw=1.0)
    _mark_projection(axes[1], known_rows, selected, (2, 3), "B width ratio", "B attack angle (deg)")
    axes[1].set_title("PSO particles projected to Scale B")
    fig.colorbar(particle_plot_b, ax=axes[1], label="Acquisition score")

    a_width = np.linspace(bounds[0][0], bounds[0][1], 70)
    a_angle = np.linspace(bounds[1][0], bounds[1][1], 70)
    mesh_aw, mesh_aa = np.meshgrid(a_width, a_angle)
    a_slice_candidates = np.column_stack([
        mesh_aw.reshape(-1),
        mesh_aa.reshape(-1),
        np.full(mesh_aw.size, selected[2]),
        np.full(mesh_aw.size, selected[3]),
    ])
    a_scores = np.asarray([acquisition_score(candidate, bundle, known, args)[0] for candidate in a_slice_candidates])
    a_scores = a_scores.reshape(mesh_aw.shape)
    a_plot = axes[2].contourf(mesh_aw, mesh_aa, a_scores, levels=24, cmap="viridis")
    _mark_projection(axes[2], known_rows, selected, (0, 1), "A width ratio", "A attack angle (deg)")
    axes[2].set_title(
        f"Acquisition slice: vary A, fix B=({selected[2]:.3f}, {selected[3]:.1f})"
    )
    fig.colorbar(a_plot, ax=axes[2], label="Acquisition score")

    b_width = np.linspace(bounds[2][0], bounds[2][1], 70)
    b_angle = np.linspace(bounds[3][0], bounds[3][1], 70)
    mesh_bw, mesh_ba = np.meshgrid(b_width, b_angle)
    b_slice_candidates = np.column_stack([
        np.full(mesh_bw.size, selected[0]),
        np.full(mesh_bw.size, selected[1]),
        mesh_bw.reshape(-1),
        mesh_ba.reshape(-1),
    ])
    b_scores = np.asarray([acquisition_score(candidate, bundle, known, args)[0] for candidate in b_slice_candidates])
    b_scores = b_scores.reshape(mesh_bw.shape)
    b_plot = axes[3].contourf(mesh_bw, mesh_ba, b_scores, levels=24, cmap="viridis")
    _mark_projection(axes[3], known_rows, selected, (2, 3), "B width ratio", "B attack angle (deg)")
    axes[3].set_title(
        f"Acquisition slice: vary B, fix A=({selected[0]:.3f}, {selected[1]:.1f})"
    )
    fig.colorbar(b_plot, ax=axes[3], label="Acquisition score")

    output_path = os.path.join(output_dir, f"{timestamp}_scale_design_pso_heterogeneous.png")
    fig.suptitle(
        "Heterogeneous 4D PSO visualization | "
        f"selected A=({selected[0]:.3f}, {selected[1]:.1f}), "
        f"B=({selected[2]:.3f}, {selected[3]:.1f}), "
        f"score={pso_result['best_score']:.3f}",
        fontsize=11,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-mode", choices=["homogeneous", "heterogeneous"], default="homogeneous")
    parser.add_argument("--proposal-mode", choices=["safe", "exploratory"], default="exploratory")
    parser.add_argument("--proposal-objective", choices=["design_value_gated_uncertainty"], default="design_value_gated_uncertainty")
    parser.add_argument("--design-value-model", default="latest")
    parser.add_argument("--particles", type=int, default=80)
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument("--c1", type=float, default=2.0)
    parser.add_argument("--c2", type=float, default=1.0)
    parser.add_argument("--inertia", type=float, default=0.85)
    parser.add_argument("--velocity-clamp", type=float, default=0.25)
    parser.add_argument("--target-novelty-distance", type=float, default=0.25)
    parser.add_argument("--target-novelty-width", type=float, default=0.10)
    parser.add_argument("--target-novelty-weight", type=float, default=5.0)
    parser.add_argument("--uncertainty-weight", type=float, default=1.0)
    parser.add_argument("--mean-weight", type=float, default=0.25)
    parser.add_argument("--value-floor", type=float, default=8.0)
    parser.add_argument("--gate-temperature", type=float, default=3.0)
    parser.add_argument("--boundary-penalty", type=float, default=2.0)
    parser.add_argument("--boundary-margin", type=float, default=0.08)
    parser.add_argument("--extrapolation-penalty", type=float, default=5.0)
    parser.add_argument("--robustness-lambda", type=float, default=ROBUSTNESS_LAMBDA)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--selection-mode", choices=["new", "best"], default="new")
    parser.add_argument("--min-proposal-distance", type=float, default=0.0)
    parser.add_argument("--save-csv", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = DESIGN_VALUE_MODEL_DIR
    os.makedirs(output_dir, exist_ok=True)

    bundle = load_design_value_model(args.design_value_model)
    model_mode = bundle.get("feature_schema", {}).get("design_mode", args.design_mode)
    if model_mode != args.design_mode:
        raise ValueError(f"Model was trained for {model_mode}, but proposal uses {args.design_mode}.")

    bounds = search_bounds(args.design_mode, args.proposal_mode)
    known_rows = source_rows(args.robustness_lambda, args.design_mode)
    known_candidates = [row["candidate"] for row in known_rows]

    objective = lambda candidate: acquisition_score(candidate, bundle, known_candidates, args)
    pso_result = run_pso(objective, bounds, args)
    selected = pso_result["best_position"]
    score, details = objective(selected)
    row = proposal_row(selected, score, details, args)

    timestamp = row["created_at"]
    csv_path = save_csv(row, output_dir) if args.save_csv else None
    if args.design_mode == "heterogeneous":
        plot_path = plot_heterogeneous(bundle, known_rows, pso_result, selected, args, output_dir, timestamp)
    else:
        plot_path = plot_homogeneous(bundle, known_rows, pso_result, selected, args, output_dir, timestamp)

    print(f"Loaded model: {bundle['path']}")
    print(f"Selected {args.design_mode} proposal: {selected}")
    print(f"Full design: {[row['A_width_ratio'], row['A_attack_angle_deg'], row['B_width_ratio'], row['B_attack_angle_deg']]}")
    print(f"Score={score:.4f}, robustness={row['design_value_robustness']:.4f}, uncertainty={row['design_value_uncertainty']:.4f}, novelty={row['novelty_distance']:.4f}")
    if csv_path:
        print(f"Saved CSV: {csv_path}")
    if plot_path:
        print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
