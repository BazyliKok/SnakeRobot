import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from goal_research_training import (  # noqa: E402
    ACTION_DIM,
    CONDITION_DIM,
    CONDITIONED_OBSERVATION_DIM,
    HIDDEN_SIZES,
    SCALE_ANGLE_BOUNDS,
    SCALE_WIDTH_BOUNDS,
    TERRAIN_LABELS,
    GoalReplayBuffer,
    GoalSACAgent,
    build_condition_vector,
    load_torch_payload,
    morphology_from_pso_position,
    set_global_seeds,
)


SEARCH_WIDTH_BOUNDS: Tuple[float, float] = (float(SCALE_WIDTH_BOUNDS[0]), float(SCALE_WIDTH_BOUNDS[1]))
SEARCH_ANGLE_BOUNDS: Tuple[float, float] = (float(SCALE_ANGLE_BOUNDS[0]), float(SCALE_ANGLE_BOUNDS[1]))


def set_search_bounds(width_min: float, width_max: float, angle_min: float, angle_max: float) -> None:
    global SEARCH_WIDTH_BOUNDS, SEARCH_ANGLE_BOUNDS
    width_min = float(width_min)
    width_max = float(width_max)
    angle_min = float(angle_min)
    angle_max = float(angle_max)
    if width_max <= width_min:
        raise ValueError("--width-max must be greater than --width-min.")
    if angle_max <= angle_min:
        raise ValueError("--angle-max must be greater than --angle-min.")
    if width_min < SCALE_WIDTH_BOUNDS[0] or width_max > SCALE_WIDTH_BOUNDS[1]:
        raise ValueError(f"Width bounds must stay within trained encoding bounds {SCALE_WIDTH_BOUNDS}.")
    if angle_min < SCALE_ANGLE_BOUNDS[0] or angle_max > SCALE_ANGLE_BOUNDS[1]:
        raise ValueError(f"Angle bounds must stay within trained encoding bounds {SCALE_ANGLE_BOUNDS}.")
    SEARCH_WIDTH_BOUNDS = (width_min, width_max)
    SEARCH_ANGLE_BOUNDS = (angle_min, angle_max)


class PopulationCriticEnsemble:
    def __init__(
        self,
        checkpoints: Sequence[Path],
        replay_path: Path,
        batch_size: int,
        seed: int,
    ):
        if not checkpoints:
            raise ValueError("At least one population critic checkpoint is required.")
        self.checkpoints = [Path(path) for path in checkpoints]
        self.replay_path = Path(replay_path)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.replay = GoalReplayBuffer(capacity=1)
        self.replay.load_npz(self.replay_path)
        set_global_seeds(self.seed)
        self.robot_observations = self.replay.sample_robot_observations(self.batch_size)
        self.agents = [self._load_agent(path) for path in self.checkpoints]

    @staticmethod
    def _load_agent(path: Path) -> GoalSACAgent:
        checkpoint = load_torch_payload(path)
        payload = checkpoint["population_agent"]
        hparams = payload.get("hyperparameters", {}) if isinstance(payload, dict) else {}
        agent = GoalSACAgent(
            obs_dim=int(hparams.get("obs_dim", CONDITIONED_OBSERVATION_DIM)),
            action_dim=int(hparams.get("action_dim", ACTION_DIM)),
            hidden_sizes=hparams.get("hidden_sizes", HIDDEN_SIZES),
            lr=float(hparams.get("lr", 1e-3)),
            gamma=float(hparams.get("gamma", 0.99)),
            tau=float(hparams.get("tau", 0.01)),
            alpha_init=float(hparams.get("alpha_init", 0.01)),
            target_entropy=float(hparams.get("target_entropy_internal", -float(ACTION_DIM))),
            grad_clip_value=float(hparams.get("grad_clip_value", 1.0)),
        )
        agent.load_state_dict_payload(payload, load_optimizers=False)
        return agent

    def score_one_agent(self, agent: GoalSACAgent, morphology, terrain: str) -> Dict[str, float]:
        condition = build_condition_vector(morphology, terrain)
        conditions = np.repeat(condition.reshape(1, -1), self.robot_observations.shape[0], axis=0)
        observations = np.concatenate([self.robot_observations, conditions], axis=1).astype(np.float32)
        actions = agent.act_batch(observations, deterministic=True)
        q1, q2 = agent.q_values(observations, actions)
        q1 = np.asarray(q1, dtype=np.float64).reshape(-1)
        q2 = np.asarray(q2, dtype=np.float64).reshape(-1)
        min_q = np.minimum(q1, q2)
        return {
            f"q_{terrain}": float(min_q.mean()),
            f"q1_{terrain}": float(q1.mean()),
            f"q2_{terrain}": float(q2.mean()),
            f"q_disagreement_{terrain}": float(np.abs(q1 - q2).mean()),
            f"q_state_std_{terrain}": float(min_q.std()),
        }

    def score_candidate(self, morphology, uncertainty_weight: float) -> Dict[str, object]:
        per_member: List[Dict[str, float]] = []
        robust_values: List[float] = []
        carpet_values: List[float] = []
        cardboard_values: List[float] = []
        q_disagreements: List[float] = []

        for member_idx, agent in enumerate(self.agents):
            member_payload: Dict[str, float] = {"member_index": float(member_idx)}
            for terrain in TERRAIN_LABELS:
                member_payload.update(self.score_one_agent(agent, morphology, terrain))
            carpet_q = float(member_payload["q_carpet"])
            cardboard_q = float(member_payload["q_cardboard"])
            robust_q = float(min(carpet_q, cardboard_q))
            member_payload["member_robust_q"] = robust_q
            member_payload["member_mean_q"] = float((carpet_q + cardboard_q) / 2.0)
            member_payload["member_terrain_q_std"] = float(np.std([carpet_q, cardboard_q]))
            per_member.append(member_payload)
            robust_values.append(robust_q)
            carpet_values.append(carpet_q)
            cardboard_values.append(cardboard_q)
            q_disagreements.extend(
                [
                    float(member_payload["q_disagreement_carpet"]),
                    float(member_payload["q_disagreement_cardboard"]),
                ]
            )

        robust_array = np.asarray(robust_values, dtype=np.float64)
        carpet_array = np.asarray(carpet_values, dtype=np.float64)
        cardboard_array = np.asarray(cardboard_values, dtype=np.float64)
        ensemble_mean_robust = float(robust_array.mean())
        ensemble_std_robust = float(robust_array.std())
        ensemble_lcb_robust = float(ensemble_mean_robust - float(uncertainty_weight) * ensemble_std_robust)
        carpet_mean = float(carpet_array.mean())
        cardboard_mean = float(cardboard_array.mean())
        terrain_q_std = float(np.std([carpet_mean, cardboard_mean]))
        return {
            "q_carpet": carpet_mean,
            "q_cardboard": cardboard_mean,
            "q_carpet_std_across_critics": float(carpet_array.std()),
            "q_cardboard_std_across_critics": float(cardboard_array.std()),
            "mean_q": float((carpet_mean + cardboard_mean) / 2.0),
            "worst_q": float(min(carpet_mean, cardboard_mean)),
            "terrain_q_std": terrain_q_std,
            "ensemble_mean_robust_q": ensemble_mean_robust,
            "ensemble_std_robust_q": ensemble_std_robust,
            "ensemble_lcb_robust_q": ensemble_lcb_robust,
            "ensemble_uncertainty_weight": float(uncertainty_weight),
            "mean_q_uncertainty": float(np.mean(q_disagreements)) if q_disagreements else 0.0,
            "base_robust_q": ensemble_mean_robust,
            "robust_q": ensemble_lcb_robust,
            "population_objective": "new_experiment_utility",
            "critic_count": len(self.agents),
            "member_scores": per_member,
        }


def layout_search_bounds(layout: str) -> Tuple[np.ndarray, np.ndarray]:
    if layout == "homogeneous":
        low = np.asarray([SEARCH_WIDTH_BOUNDS[0], SEARCH_ANGLE_BOUNDS[0]], dtype=np.float64)
        high = np.asarray([SEARCH_WIDTH_BOUNDS[1], SEARCH_ANGLE_BOUNDS[1]], dtype=np.float64)
        return low, high
    if layout == "heterogeneous_ab":
        low = np.asarray(
            [SEARCH_WIDTH_BOUNDS[0], SEARCH_ANGLE_BOUNDS[0], SEARCH_WIDTH_BOUNDS[0], SEARCH_ANGLE_BOUNDS[0]],
            dtype=np.float64,
        )
        high = np.asarray(
            [SEARCH_WIDTH_BOUNDS[1], SEARCH_ANGLE_BOUNDS[1], SEARCH_WIDTH_BOUNDS[1], SEARCH_ANGLE_BOUNDS[1]],
            dtype=np.float64,
        )
        return low, high
    raise ValueError("layout must be 'homogeneous' or 'heterogeneous_ab'.")


def denormalize_from_unit(value: float, bounds: Tuple[float, float]) -> float:
    low, high = float(bounds[0]), float(bounds[1])
    return float(low + 0.5 * (np.clip(float(value), -1.0, 1.0) + 1.0) * (high - low))


def condition_to_position(layout: str, condition: Sequence[float]) -> np.ndarray:
    condition_array = np.asarray(condition, dtype=np.float64).reshape(CONDITION_DIM)
    a_width = denormalize_from_unit(condition_array[0], SCALE_WIDTH_BOUNDS)
    a_angle = denormalize_from_unit(condition_array[1], SCALE_ANGLE_BOUNDS)
    b_width = denormalize_from_unit(condition_array[2], SCALE_WIDTH_BOUNDS)
    b_angle = denormalize_from_unit(condition_array[3], SCALE_ANGLE_BOUNDS)
    if layout == "homogeneous":
        return np.asarray([a_width, a_angle], dtype=np.float64)
    return np.asarray([a_width, a_angle, b_width, b_angle], dtype=np.float64)


def unique_known_positions_from_replay(replay: GoalReplayBuffer, layout: str) -> List[np.ndarray]:
    if len(replay) <= 0:
        return []
    indices = np.arange(replay.size if replay.size < replay.capacity else replay.capacity, dtype=np.int64)
    positions: List[np.ndarray] = []
    seen = set()
    for idx in indices:
        position = condition_to_position(layout, replay.conditions[idx])
        key = tuple(round(float(value), 3 if axis % 2 == 0 else 2) for axis, value in enumerate(position))
        if key in seen:
            continue
        seen.add(key)
        positions.append(position)
    return positions


def nearest_position_distance(position: Sequence[float], known_positions: Sequence[Sequence[float]], layout: str) -> float:
    if not known_positions:
        return float("inf")
    low, high = layout_search_bounds(layout)
    span = np.maximum(high - low, 1e-9)
    points = np.asarray([np.asarray(item, dtype=np.float64) for item in known_positions], dtype=np.float64)
    query = np.asarray(position, dtype=np.float64)
    return float(np.min(np.linalg.norm((points - query) / span, axis=1)))


def heterogeneity_distance(position: Sequence[float], layout: str) -> float:
    if layout != "heterogeneous_ab":
        return float("inf")
    low, high = layout_search_bounds(layout)
    span = np.maximum(high - low, 1e-9)
    pos = np.asarray(position, dtype=np.float64)
    return float(np.linalg.norm((pos[:2] - pos[2:]) / span[:2]))


def heterogeneity_utility(position: Sequence[float], layout: str, heterogeneity_scale: float) -> float:
    if layout != "heterogeneous_ab":
        return 1.0
    distance = heterogeneity_distance(position, layout)
    return float(min(1.0, distance / max(float(heterogeneity_scale), 1e-9)))


def summary_position(row: Dict[str, object], layout: str) -> np.ndarray:
    if layout == "homogeneous":
        return np.asarray([float(row["a_width"]), float(row["a_angle"])], dtype=np.float64)
    return np.asarray(
        [float(row["a_width"]), float(row["a_angle"]), float(row["b_width"]), float(row["b_angle"])],
        dtype=np.float64,
    )


def run_ensemble_pso(
    scorer,
    layout: str,
    particles: int,
    iterations: int,
    top_k: int,
    seed: int,
    known_positions: Sequence[Sequence[float]],
    min_candidate_distance: float,
    novelty_scale: float,
    heterogeneity_scale: float,
    selection_grid_size: int = 0,
) -> List[Dict[str, object]]:
    rng = np.random.default_rng(seed)
    low, high = layout_search_bounds(layout)
    span = high - low
    positions = low + rng.random((particles, low.size)) * span
    velocities = rng.uniform(-0.25, 0.25, size=positions.shape) * span
    personal_best = positions.copy()

    def score_position_for_pso(pos: Sequence[float]) -> Tuple[Dict[str, object], float, float, float]:
        morphology = morphology_from_pso_position(layout, pos)
        payload = scorer(morphology)
        raw_score = float(payload["robust_q"])
        nearest = nearest_position_distance(pos, known_positions, layout)
        novelty = 1.0 if not np.isfinite(nearest) else nearest / (nearest + max(float(novelty_scale), 1e-9))
        hetero_distance = heterogeneity_distance(pos, layout)
        
        if layout == "heterogeneous_ab":
            width_diff = abs(pos[0] - pos[2])
            angle_diff = abs(pos[1] - pos[3])
            if width_diff < 0.15 and angle_diff < 10.0:
                score = -1.0 
                payload["pso_score"] = score 
                return payload, raw_score, score, float(nearest)
        hetero_utility = heterogeneity_utility(pos, layout, heterogeneity_scale)
        score = raw_score * novelty * hetero_utility
        
        payload["novelty_utility"] = float(novelty)
        payload["novelty_scale"] = float(novelty_scale)
        payload["heterogeneity_distance"] = float(hetero_distance)
        payload["heterogeneity_utility"] = float(hetero_utility)
        payload["heterogeneity_scale"] = float(heterogeneity_scale)
        return payload, raw_score, float(score), float(nearest)

    def candidate_row_from_position(pos: Sequence[float], source: str) -> Dict[str, object]:
        payload, raw_score, score, nearest = score_position_for_pso(pos)
        row = morphology_from_pso_position(layout, pos).metadata()
        row.update(payload)
        row["predicted_score"] = float(raw_score)
        row["pso_score"] = float(score)
        row["nearest_existing_distance"] = float(nearest)
        row["proposal_source"] = source
        return row

    initial_scores = [score_position_for_pso(pos) for pos in positions]
    personal_payloads = [item[0] for item in initial_scores]
    personal_raw_scores = np.asarray([item[1] for item in initial_scores], dtype=np.float64)
    personal_scores = np.asarray([item[2] for item in initial_scores], dtype=np.float64)
    personal_distances = np.asarray([item[3] for item in initial_scores], dtype=np.float64)
    global_best = personal_best[int(np.argmax(personal_scores))].copy()
    global_score = float(np.max(personal_scores))
    candidates: List[Dict[str, object]] = []

    for _ in range(int(iterations)):
        r1 = rng.random(positions.shape)
        r2 = rng.random(positions.shape)
        velocities = 0.65 * velocities + 1.4 * r1 * (personal_best - positions) + 1.4 * r2 * (global_best - positions)
        velocities = np.clip(velocities, -0.25 * span, 0.25 * span)
        positions = np.clip(positions + velocities, low, high)
        for particle_idx, pos in enumerate(positions):
            payload, raw_score, score, nearest = score_position_for_pso(pos)
            if score > float(personal_scores[particle_idx]):
                personal_best[particle_idx] = pos
                personal_raw_scores[particle_idx] = raw_score
                personal_scores[particle_idx] = score
                personal_distances[particle_idx] = nearest
                personal_payloads[particle_idx] = payload
            if score > global_score:
                global_score = score
                global_best = pos.copy()
            row = morphology_from_pso_position(layout, pos).metadata()
            row.update(payload)
            row["predicted_score"] = float(raw_score)
            row["pso_score"] = float(score)
            row["nearest_existing_distance"] = float(nearest)
            row["proposal_source"] = "pso"
            candidates.append(row)

    for pos, raw_score, score, nearest, payload in zip(
        personal_best,
        personal_raw_scores,
        personal_scores,
        personal_distances,
        personal_payloads,
    ):
        row = morphology_from_pso_position(layout, pos).metadata()
        row.update(payload)
        row["predicted_score"] = float(raw_score)
        row["pso_score"] = float(score)
        row["nearest_existing_distance"] = float(nearest)
        row["proposal_source"] = "pso_personal_best"
        candidates.append(row)

    if layout == "homogeneous" and int(selection_grid_size) > 1:
        widths = np.linspace(low[0], high[0], int(selection_grid_size), dtype=np.float64)
        angles = np.linspace(low[1], high[1], int(selection_grid_size), dtype=np.float64)
        for width in widths:
            for angle in angles:
                candidates.append(candidate_row_from_position([float(width), float(angle)], "selection_grid"))

    candidates.sort(key=lambda item: float(item["pso_score"]), reverse=True)
    selected: List[Dict[str, object]] = []
    for candidate in candidates:
        position = summary_position(candidate, layout)
        if any(np.linalg.norm((position - summary_position(row, layout)) / span) < min_candidate_distance for row in selected):
            continue
        selected.append(candidate)
        if len(selected) >= int(top_k):
            break
    return selected


def read_manifest(path: Optional[Path]) -> Tuple[List[Path], Optional[Path]]:
    if path is None:
        return [], None
    payload = json.loads(path.read_text(encoding="utf-8"))
    members = payload.get("members", [])
    checkpoints = [Path(member["checkpoint"]) for member in members if member.get("checkpoint")]
    replay = None
    for member in members:
        if member.get("replay"):
            replay = Path(member["replay"])
            break
    return checkpoints, replay


def collect_checkpoints(args: argparse.Namespace) -> Tuple[List[Path], Path]:
    manifest_checkpoints, manifest_replay = read_manifest(args.manifest)
    checkpoints = list(manifest_checkpoints)
    checkpoints.extend(Path(path) for path in args.population_checkpoint)
    if not checkpoints:
        raise ValueError("Provide --manifest or at least one --population-checkpoint.")
    replay = Path(args.population_replay_npz) if args.population_replay_npz else manifest_replay
    if replay is None:
        raise ValueError("Provide --population-replay-npz or a manifest containing a replay path.")
    return checkpoints, replay


def merge_known_positions(position_groups: Sequence[Sequence[np.ndarray]], layout: str) -> List[np.ndarray]:
    merged: List[np.ndarray] = []
    seen = set()
    for positions in position_groups:
        for position in positions:
            array = np.asarray(position, dtype=np.float64)
            key = tuple(round(float(value), 3 if axis % 2 == 0 else 2) for axis, value in enumerate(array))
            if key in seen:
                continue
            seen.add(key)
            merged.append(array)
    return merged


def known_positions_from_replay_path(path: Path, layout: str) -> List[np.ndarray]:
    replay = GoalReplayBuffer(capacity=1)
    replay.load_npz(path)
    return unique_known_positions_from_replay(replay, layout)


def collect_known_positions(args: argparse.Namespace, ensemble: PopulationCriticEnsemble, layout: str) -> List[np.ndarray]:
    groups: List[Sequence[np.ndarray]] = [unique_known_positions_from_replay(ensemble.replay, layout)]
    for replay_path in args.extra_novelty_replay_npz:
        groups.append(known_positions_from_replay_path(Path(replay_path), layout))
    for manifest_path in args.extra_novelty_manifest:
        _, replay_path = read_manifest(Path(manifest_path))
        if replay_path is not None:
            groups.append(known_positions_from_replay_path(replay_path, layout))
    return merge_known_positions(groups, layout)


def candidate_position(candidate: Optional[Dict[str, object]], layout: str, known_positions: Sequence[np.ndarray]) -> np.ndarray:
    low, high = layout_search_bounds(layout)
    if candidate is not None:
        return summary_position(candidate, layout)
    if known_positions:
        return np.asarray(known_positions, dtype=np.float64).mean(axis=0)
    return (low + high) / 2.0


def score_position(
    ensemble: PopulationCriticEnsemble,
    layout: str,
    position: Sequence[float],
    uncertainty_weight: float,
    novelty_scale: float,
    heterogeneity_scale: float,
    known_positions: Sequence[Sequence[float]],
) -> Tuple[Dict[str, object], float]:
    morphology = morphology_from_pso_position(layout, position)
    score = ensemble.score_candidate(morphology, uncertainty_weight=uncertainty_weight)
    nearest = nearest_position_distance(position, known_positions, layout)
    novelty = 1.0 if not np.isfinite(nearest) else nearest / (nearest + max(float(novelty_scale), 1e-9))
    hetero = heterogeneity_utility(position, layout, heterogeneity_scale)
    return score, float(score["robust_q"]) * float(novelty) * float(hetero)


def add_homogeneous_plot(
    output_png: Path,
    ensemble: PopulationCriticEnsemble,
    proposal: Optional[Dict[str, object]],
    uncertainty_weight: float,
    novelty_scale: float,
    heterogeneity_scale: float,
    known_positions: Sequence[np.ndarray],
    grid_size: int,
    dpi: int,
) -> None:
    widths = np.linspace(SEARCH_WIDTH_BOUNDS[0], SEARCH_WIDTH_BOUNDS[1], int(grid_size))
    angles = np.linspace(SEARCH_ANGLE_BOUNDS[0], SEARCH_ANGLE_BOUNDS[1], int(grid_size))
    mesh_width, mesh_angle = np.meshgrid(widths, angles)

    carpet = np.zeros_like(mesh_width)
    cardboard = np.zeros_like(mesh_width)
    robust_lcb = np.zeros_like(mesh_width)
    utility = np.zeros_like(mesh_width)

    for row_idx, angle in enumerate(angles):
        for col_idx, width in enumerate(widths):
            score, utility_value = score_position(
                ensemble=ensemble,
                layout="homogeneous",
                position=[float(width), float(angle)],
                uncertainty_weight=uncertainty_weight,
                novelty_scale=novelty_scale,
                heterogeneity_scale=heterogeneity_scale,
                known_positions=known_positions,
            )
            carpet[row_idx, col_idx] = float(score["q_carpet"])
            cardboard[row_idx, col_idx] = float(score["q_cardboard"])
            robust_lcb[row_idx, col_idx] = float(score["robust_q"])
            utility[row_idx, col_idx] = utility_value

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6), constrained_layout=True)
    panels = [
        (carpet, "Mean carpet value"),
        (cardboard, "Mean cardboard value"),
        (robust_lcb, "Robust ensemble value"),
        (utility, "PSO utility"),
    ]
    known_widths = [float(position[0]) for position in known_positions]
    known_angles = [float(position[1]) for position in known_positions]
    for axis, (values, title) in zip(axes, panels):
        contour = axis.contourf(mesh_width, mesh_angle, values, levels=24, cmap="viridis")
        fig.colorbar(contour, ax=axis)
        if known_positions:
            axis.scatter(
                known_widths,
                known_angles,
                c="white",
                edgecolors="black",
                linewidths=0.8,
                s=42,
                label="tested designs",
            )
        if proposal is not None:
            axis.scatter(
                float(proposal["a_width"]),
                float(proposal["a_angle"]),
                c="red",
                edgecolors="black",
                marker="*",
                s=190,
                label="ensemble PSO proposal",
            )
        axis.set_title(title)
        axis.set_xlabel("scale width")
        axis.set_ylabel("attack angle")
        axis.set_xlim(SEARCH_WIDTH_BOUNDS[0], SEARCH_WIDTH_BOUNDS[1])
        axis.set_ylim(SEARCH_ANGLE_BOUNDS[0], SEARCH_ANGLE_BOUNDS[1])
    axes[0].legend(loc="best")
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=int(dpi))
    plt.close(fig)


def add_heterogeneous_plot(
    output_png: Path,
    ensemble: PopulationCriticEnsemble,
    proposal: Optional[Dict[str, object]],
    uncertainty_weight: float,
    novelty_scale: float,
    heterogeneity_scale: float,
    known_positions: Sequence[np.ndarray],
    grid_size: int,
    dpi: int,
) -> None:
    layout = "heterogeneous_ab"
    base = candidate_position(proposal, layout, known_positions)
    widths = np.linspace(SEARCH_WIDTH_BOUNDS[0], SEARCH_WIDTH_BOUNDS[1], int(grid_size))
    angles = np.linspace(SEARCH_ANGLE_BOUNDS[0], SEARCH_ANGLE_BOUNDS[1], int(grid_size))
    mesh_width, mesh_angle = np.meshgrid(widths, angles)

    fig, axes = plt.subplots(2, 4, figsize=(18, 8.4), constrained_layout=True)
    row_specs = [
        ("A varied, B fixed", 0, 1, float(base[2]), float(base[3])),
        ("B varied, A fixed", 2, 3, float(base[0]), float(base[1])),
    ]
    known_array = np.asarray(known_positions, dtype=np.float64) if known_positions else np.empty((0, 4))

    for row_idx, (row_label, width_axis, angle_axis, fixed_width, fixed_angle) in enumerate(row_specs):
        carpet = np.zeros_like(mesh_width)
        cardboard = np.zeros_like(mesh_width)
        robust_lcb = np.zeros_like(mesh_width)
        utility = np.zeros_like(mesh_width)

        for angle_idx, angle in enumerate(angles):
            for width_idx, width in enumerate(widths):
                position = np.asarray(base, dtype=np.float64).copy()
                position[width_axis] = float(width)
                position[angle_axis] = float(angle)
                score, utility_value = score_position(
                    ensemble=ensemble,
                    layout=layout,
                    position=position,
                    uncertainty_weight=uncertainty_weight,
                    novelty_scale=novelty_scale,
                    heterogeneity_scale=heterogeneity_scale,
                    known_positions=known_positions,
                )
                carpet[angle_idx, width_idx] = float(score["q_carpet"])
                cardboard[angle_idx, width_idx] = float(score["q_cardboard"])
                robust_lcb[angle_idx, width_idx] = float(score["robust_q"])
                utility[angle_idx, width_idx] = utility_value

        panels = [
            (carpet, "Mean carpet value"),
            (cardboard, "Mean cardboard value"),
            (robust_lcb, "Robust ensemble value"),
            (utility, "PSO utility"),
        ]
        for axis, (values, title) in zip(axes[row_idx], panels):
            contour = axis.contourf(mesh_width, mesh_angle, values, levels=24, cmap="viridis")
            fig.colorbar(contour, ax=axis)
            if known_array.size:
                axis.scatter(
                    known_array[:, width_axis],
                    known_array[:, angle_axis],
                    c="white",
                    edgecolors="black",
                    linewidths=0.8,
                    s=42,
                    label="tested designs",
                )
            if proposal is not None:
                proposal_position = summary_position(proposal, layout)
                axis.scatter(
                    float(proposal_position[width_axis]),
                    float(proposal_position[angle_axis]),
                    c="red",
                    edgecolors="black",
                    marker="*",
                    s=190,
                    label="ensemble PSO proposal",
                )
            axis.set_title(f"{row_label}: {title}")
            axis.set_xlabel("scale width")
            axis.set_ylabel("attack angle")
            axis.set_xlim(SEARCH_WIDTH_BOUNDS[0], SEARCH_WIDTH_BOUNDS[1])
            axis.set_ylim(SEARCH_ANGLE_BOUNDS[0], SEARCH_ANGLE_BOUNDS[1])
        axes[row_idx, 0].text(
            0.01,
            1.04,
            f"fixed other scale: width={fixed_width:.3f}, angle={fixed_angle:.2f} deg",
            transform=axes[row_idx, 0].transAxes,
            fontsize=9,
        )

    axes[0, 0].legend(loc="best")
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=int(dpi))
    plt.close(fig)


def add_plot(
    output_png: Path,
    ensemble: PopulationCriticEnsemble,
    proposal: Optional[Dict[str, object]],
    layout: str,
    uncertainty_weight: float,
    novelty_scale: float,
    heterogeneity_scale: float,
    known_positions: Sequence[np.ndarray],
    grid_size: int,
    dpi: int,
) -> None:
    if layout == "homogeneous":
        add_homogeneous_plot(
            output_png=output_png,
            ensemble=ensemble,
            proposal=proposal,
            uncertainty_weight=uncertainty_weight,
            novelty_scale=novelty_scale,
            heterogeneity_scale=heterogeneity_scale,
            known_positions=known_positions,
            grid_size=grid_size,
            dpi=dpi,
        )
        return
    add_heterogeneous_plot(
        output_png=output_png,
        ensemble=ensemble,
        proposal=proposal,
        uncertainty_weight=uncertainty_weight,
        novelty_scale=novelty_scale,
        heterogeneity_scale=heterogeneity_scale,
        known_positions=known_positions,
        grid_size=grid_size,
        dpi=dpi,
    )


def propose(args: argparse.Namespace) -> Dict[str, object]:
    set_search_bounds(args.width_min, args.width_max, args.angle_min, args.angle_max)
    checkpoints, replay_path = collect_checkpoints(args)
    layout = str(args.layout)
    set_global_seeds(int(args.seed))
    ensemble = PopulationCriticEnsemble(
        checkpoints=checkpoints,
        replay_path=replay_path,
        batch_size=int(args.batch_size),
        seed=int(args.seed),
    )
    known_positions = collect_known_positions(args, ensemble, layout)
    low, high = layout_search_bounds(layout)

    candidates = run_ensemble_pso(
        scorer=lambda morphology: ensemble.score_candidate(
            morphology,
            uncertainty_weight=float(args.ensemble_uncertainty_weight),
        ),
        layout=layout,
        particles=int(args.particles),
        iterations=int(args.iterations),
        top_k=int(args.top_k),
        seed=int(args.seed),
        known_positions=known_positions,
        min_candidate_distance=float(args.min_candidate_distance),
        novelty_scale=float(args.novelty_scale),
        heterogeneity_scale=float(args.heterogeneity_scale),
        selection_grid_size=int(args.grid_size) if layout == "homogeneous" else 0,
    )

    for candidate in candidates:
        position = summary_position(candidate, layout)
        candidate["normalized_position"] = [
            float(value) for value in ((position - low) / np.maximum(high - low, 1e-9))
        ]

    candidate_key = "homogeneous_candidates" if layout == "homogeneous" else "heterogeneous_candidates"
    payload: Dict[str, object] = {
        "layout": layout,
        "proposal_method": "population_critic_ensemble",
        "score_formula": (
            "final_score = (mean robust critic value - uncertainty_weight * critic disagreement) "
            "* novelty * saturating_heterogeneity"
        ),
        "search_bounds": {
            "a_width": [float(SEARCH_WIDTH_BOUNDS[0]), float(SEARCH_WIDTH_BOUNDS[1])],
            "a_angle": [float(SEARCH_ANGLE_BOUNDS[0]), float(SEARCH_ANGLE_BOUNDS[1])],
            "b_width": [float(SEARCH_WIDTH_BOUNDS[0]), float(SEARCH_WIDTH_BOUNDS[1])],
            "b_angle": [float(SEARCH_ANGLE_BOUNDS[0]), float(SEARCH_ANGLE_BOUNDS[1])],
        },
        "critic_count": len(checkpoints),
        "population_checkpoints": [str(path) for path in checkpoints],
        "population_replay_npz": str(replay_path),
        "ensemble_uncertainty_weight": float(args.ensemble_uncertainty_weight),
        "novelty_scale": float(args.novelty_scale),
        "heterogeneity_scale": float(args.heterogeneity_scale),
        "selection_grid_size": int(args.grid_size) if layout == "homogeneous" else 0,
        "known_design_count": len(known_positions),
        "known_design_positions": [[float(value) for value in position] for position in known_positions],
        "extra_novelty_replay_npz": [str(path) for path in args.extra_novelty_replay_npz],
        "extra_novelty_manifest": [str(path) for path in args.extra_novelty_manifest],
        "min_existing_distance": 0.0,
        "next_candidate": candidates[0] if candidates else None,
        "candidates": candidates,
        candidate_key: candidates,
    }
    if layout == "heterogeneous_ab":
        payload["placement"] = "modules 1,3,5,7 = A; modules 2,4,6,8 = B"

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.output_png:
        add_plot(
            output_png=args.output_png,
            ensemble=ensemble,
            proposal=payload["next_candidate"] if isinstance(payload.get("next_candidate"), dict) else None,
            layout=layout,
            uncertainty_weight=float(args.ensemble_uncertainty_weight),
            novelty_scale=float(args.novelty_scale),
            heterogeneity_scale=float(args.heterogeneity_scale),
            known_positions=known_positions,
            grid_size=int(args.grid_size),
            dpi=int(args.dpi),
        )
    print(json.dumps(payload, indent=2))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use an ensemble of population critics to propose a scale design."
    )
    parser.add_argument("--layout", choices=["homogeneous", "heterogeneous_ab"], default="homogeneous")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--population-checkpoint", action="append", default=[], type=Path)
    parser.add_argument("--population-replay-npz", type=Path)
    parser.add_argument("--extra-novelty-replay-npz", action="append", default=[], type=Path)
    parser.add_argument("--extra-novelty-manifest", action="append", default=[], type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-png", type=Path)
    parser.add_argument("--particles", type=int, default=24)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--ensemble-uncertainty-weight", type=float, default=1.0)
    parser.add_argument("--novelty-scale", type=float, default=0.15)
    parser.add_argument("--heterogeneity-scale", type=float, default=0.15)
    parser.add_argument("--width-min", type=float, default=SCALE_WIDTH_BOUNDS[0])
    parser.add_argument("--width-max", type=float, default=SCALE_WIDTH_BOUNDS[1])
    parser.add_argument("--angle-min", type=float, default=SCALE_ANGLE_BOUNDS[0])
    parser.add_argument("--angle-max", type=float, default=SCALE_ANGLE_BOUNDS[1])
    parser.add_argument("--min-candidate-distance", type=float, default=0.02)
    parser.add_argument("--grid-size", type=int, default=21)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


if __name__ == "__main__":
    propose(parse_args())
