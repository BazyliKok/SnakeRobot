#!/usr/bin/env python
"""Inspect whether a GOAL SAC policy has collapsed to one action per state."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from goal_research_training import (
    ACTION_DIM,
    CONDITIONED_OBSERVATION_DIM,
    HIDDEN_SIZES,
    TERRAIN_LABELS,
    GoalReplayBuffer,
    GoalSACAgent,
    load_torch_payload,
)


ACTION_COLUMNS = [f"Motor{i}" for i in range(1, ACTION_DIM + 1)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a GOAL checkpoint and replay snapshot, then measure the actor "
            "distribution: deterministic action, sampled action spread, pre-tanh "
            "std/log-std, and Monte Carlo entropy."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to episode_XXX_terrain.pt.")
    parser.add_argument(
        "--role",
        choices=("individual", "population"),
        default="individual",
        help="Which agent payload to inspect from the checkpoint.",
    )
    parser.add_argument(
        "--replay-npz",
        type=Path,
        help="Replay snapshot. Defaults to checkpoint run_data individual/population replay.",
    )
    parser.add_argument(
        "--terrain",
        choices=tuple(TERRAIN_LABELS) + ("all",),
        default="all",
        help="Filter replay states to a terrain.",
    )
    parser.add_argument("--num-states", type=int, default=256)
    parser.add_argument("--samples-per-state", type=int, default=256)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory. Defaults to '<checkpoint run dir>/policy_distribution'.",
    )
    return parser.parse_args()


def resolve_path(path: Optional[Path], base_dir: Path) -> Optional[Path]:
    if path is None:
        return None
    return path if path.is_absolute() else (base_dir / path)


def default_replay_path(checkpoint: Dict[str, object], role: str, checkpoint_path: Path) -> Optional[Path]:
    run_data = checkpoint.get("run_data", {})
    if not isinstance(run_data, dict):
        return None
    key = "individual_replay" if role == "individual" else "population_replay"
    raw_path = str(run_data.get(key, "")).strip()
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.exists():
        return path
    if not path.is_absolute():
        candidate = checkpoint_path.parent.parent / path
        if candidate.exists():
            return candidate
    return path


def active_indices(replay: GoalReplayBuffer) -> np.ndarray:
    if replay.size <= 0:
        return np.asarray([], dtype=np.int64)
    if replay.size < replay.capacity and replay.top == replay.size:
        return np.arange(replay.size, dtype=np.int64)
    return np.concatenate(
        [np.arange(replay.top, replay.capacity), np.arange(0, replay.top)]
    )[: replay.size].astype(np.int64)


def load_replay_states(
    replay_path: Path,
    terrain: str,
    num_states: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    replay = GoalReplayBuffer()
    replay.load_npz(replay_path)
    indices = active_indices(replay)
    if indices.size == 0:
        raise RuntimeError(f"Replay snapshot is empty: {replay_path}")

    if terrain != "all":
        terrain_id = TERRAIN_LABELS.index(terrain)
        terrain_mask = replay.terrain_ids[indices, 0] == terrain_id
        indices = indices[terrain_mask]
        if indices.size == 0:
            raise RuntimeError(f"No replay states for terrain '{terrain}' in {replay_path}")

    finite_mask = np.all(np.isfinite(replay.robot_observations[indices]), axis=1)
    finite_mask &= np.all(np.isfinite(replay.conditions[indices]), axis=1)
    indices = indices[finite_mask]
    if indices.size == 0:
        raise RuntimeError(f"No finite replay observations found in {replay_path}")

    rng = np.random.default_rng(seed)
    count = min(int(num_states), indices.size)
    selected = rng.choice(indices, size=count, replace=False)
    observations = np.concatenate(
        [replay.robot_observations[selected], replay.conditions[selected]],
        axis=1,
    ).astype(np.float32)
    replay_actions = replay.actions[selected].astype(np.float32)
    return observations, replay_actions


def make_agent(checkpoint: Dict[str, object], role: str, device: str) -> GoalSACAgent:
    payload_key = "individual_agent" if role == "individual" else "population_agent"
    if payload_key not in checkpoint:
        raise KeyError(f"Checkpoint has no '{payload_key}' payload.")
    hyper = checkpoint[payload_key].get("hyperparameters", {}) if isinstance(checkpoint[payload_key], dict) else {}
    agent = GoalSACAgent(
        obs_dim=int(hyper.get("obs_dim", CONDITIONED_OBSERVATION_DIM)),
        action_dim=int(hyper.get("action_dim", ACTION_DIM)),
        hidden_sizes=tuple(hyper.get("hidden_sizes", HIDDEN_SIZES)),
        lr=float(hyper.get("learning_rate", 1e-3)),
        gamma=float(hyper.get("gamma", 0.99)),
        tau=float(hyper.get("tau", 0.01)),
        alpha_init=float(hyper.get("alpha_init", 0.01)),
        target_entropy=float(hyper.get("target_entropy_internal", -float(ACTION_DIM))),
        grad_clip_value=float(hyper.get("grad_clip_value", 1.0)),
        device=device,
    )
    agent.load_state_dict_payload(checkpoint[payload_key], load_optimizers=False)
    return agent


def policy_distribution(
    agent: GoalSACAgent,
    observations: np.ndarray,
    samples_per_state: int,
) -> Dict[str, np.ndarray | float]:
    obs = torch.as_tensor(observations, dtype=torch.float32, device=agent.device)
    agent.policy.eval()
    with torch.no_grad():
        tuple_outputs = agent._policy_outputs_tuple(
            obs,
            deterministic=False,
            return_log_prob=False,
            reparameterize=False,
        )
        if tuple_outputs is not None:
            _, pre_tanh_mean, pre_tanh_log_std, _ = tuple_outputs
            det_outputs = agent._policy_outputs_tuple(
                obs,
                deterministic=True,
                return_log_prob=False,
                reparameterize=False,
            )
            deterministic = det_outputs[0] if det_outputs is not None else torch.tanh(pre_tanh_mean)
        else:
            dist = agent.policy(obs)
            pre_tanh_mean = getattr(dist, "normal_mean", getattr(dist, "mean", None))
            pre_tanh_std = getattr(dist, "normal_std", None)
            if pre_tanh_mean is None:
                raise RuntimeError("Could not read policy mean from distribution.")
            if pre_tanh_std is None:
                pre_tanh_std = torch.ones_like(pre_tanh_mean)
            pre_tanh_log_std = torch.log(torch.clamp(pre_tanh_std, min=1e-8))
            deterministic = dist.mle_estimate() if hasattr(dist, "mle_estimate") else torch.tanh(pre_tanh_mean)

    samples = []
    log_pis = []
    for _ in range(int(samples_per_state)):
        with torch.no_grad():
            actions, log_pi, _, _ = agent._policy_sample(obs)
        samples.append(actions.detach().cpu().numpy())
        log_pis.append(log_pi.detach().cpu().numpy())

    sample_array = np.stack(samples, axis=0).astype(np.float32)
    log_pi_array = np.stack(log_pis, axis=0).astype(np.float32)
    return {
        "pre_tanh_mean": pre_tanh_mean.detach().cpu().numpy().astype(np.float32),
        "pre_tanh_log_std": pre_tanh_log_std.detach().cpu().numpy().astype(np.float32),
        "pre_tanh_std": np.exp(pre_tanh_log_std.detach().cpu().numpy()).astype(np.float32),
        "deterministic_action": deterministic.detach().cpu().numpy().astype(np.float32),
        "sample_actions": sample_array,
        "sample_log_pi": log_pi_array,
        "sample_entropy": float(-np.mean(log_pi_array)),
    }


def summarize(distribution: Dict[str, np.ndarray | float], replay_actions: np.ndarray, agent: GoalSACAgent) -> Dict[str, float]:
    samples = distribution["sample_actions"]
    deterministic = distribution["deterministic_action"]
    pre_tanh_std = distribution["pre_tanh_std"]
    pre_tanh_log_std = distribution["pre_tanh_log_std"]

    per_state_motor_std = samples.std(axis=0)
    per_state_mean_std = per_state_motor_std.mean(axis=1)
    pre_tanh_entropy_per_motor = 0.5 * np.log(2.0 * math.pi * math.e * np.square(np.clip(pre_tanh_std, 1e-8, None)))
    pre_tanh_entropy_per_state = pre_tanh_entropy_per_motor.sum(axis=1)
    sampled_mean = samples.mean(axis=0)
    return {
        "num_states": int(deterministic.shape[0]),
        "samples_per_state": int(samples.shape[0]),
        "alpha": float(agent.alpha),
        "mc_tanh_entropy_mean": float(distribution["sample_entropy"]),
        "pre_tanh_entropy_mean": float(np.mean(pre_tanh_entropy_per_state)),
        "pre_tanh_std_mean": float(np.mean(pre_tanh_std)),
        "pre_tanh_std_p05": float(np.percentile(pre_tanh_std, 5)),
        "pre_tanh_std_p50": float(np.percentile(pre_tanh_std, 50)),
        "pre_tanh_std_p95": float(np.percentile(pre_tanh_std, 95)),
        "pre_tanh_log_std_mean": float(np.mean(pre_tanh_log_std)),
        "post_tanh_sample_std_mean": float(np.mean(per_state_motor_std)),
        "post_tanh_sample_std_p50_state": float(np.percentile(per_state_mean_std, 50)),
        "post_tanh_sample_std_p95_state": float(np.percentile(per_state_mean_std, 95)),
        "deterministic_abs_action_mean": float(np.mean(np.abs(deterministic))),
        "sampled_abs_action_mean": float(np.mean(np.abs(samples))),
        "sampled_mean_abs_action_mean": float(np.mean(np.abs(sampled_mean))),
        "replay_abs_action_mean": float(np.mean(np.abs(replay_actions))),
        "replay_action_std_mean": float(np.mean(np.std(replay_actions, axis=0))),
    }


def write_summary_csv(path: Path, summary: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


def write_motor_csv(path: Path, distribution: Dict[str, np.ndarray | float]) -> None:
    deterministic = distribution["deterministic_action"]
    sampled = distribution["sample_actions"]
    pre_tanh_std = distribution["pre_tanh_std"]
    pre_tanh_log_std = distribution["pre_tanh_log_std"]
    rows = []
    for motor_idx, motor in enumerate(ACTION_COLUMNS):
        motor_samples = sampled[:, :, motor_idx].reshape(-1)
        rows.append(
            {
                "motor": motor,
                "deterministic_action_mean": float(np.mean(deterministic[:, motor_idx])),
                "deterministic_abs_action_mean": float(np.mean(np.abs(deterministic[:, motor_idx]))),
                "sample_action_mean": float(np.mean(motor_samples)),
                "sample_action_std": float(np.std(motor_samples)),
                "sample_action_p05": float(np.percentile(motor_samples, 5)),
                "sample_action_p50": float(np.percentile(motor_samples, 50)),
                "sample_action_p95": float(np.percentile(motor_samples, 95)),
                "pre_tanh_std_mean": float(np.mean(pre_tanh_std[:, motor_idx])),
                "pre_tanh_log_std_mean": float(np.mean(pre_tanh_log_std[:, motor_idx])),
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_distribution(output_dir: Path, distribution: Dict[str, np.ndarray | float]) -> Sequence[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    deterministic = distribution["deterministic_action"]
    sampled = distribution["sample_actions"]
    pre_tanh_std = distribution["pre_tanh_std"]
    sample_std_by_state = sampled.std(axis=0).mean(axis=1)
    paths = []

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    x = np.arange(1, ACTION_DIM + 1)
    axes[0].boxplot([deterministic[:, i] for i in range(ACTION_DIM)], positions=x, showfliers=False)
    axes[0].axhline(0.0, color="#777777", lw=1)
    axes[0].set_ylabel("Deterministic action")
    axes[0].set_title("Policy Mean Action by Motor")

    axes[1].boxplot([sampled[:, :, i].reshape(-1) for i in range(ACTION_DIM)], positions=x, showfliers=False)
    axes[1].axhline(0.0, color="#777777", lw=1)
    axes[1].set_ylabel("Sampled action")
    axes[1].set_xlabel("Motor")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(ACTION_COLUMNS)
    axes[1].set_title("Sampled Action Distribution by Motor")
    fig.tight_layout()
    path = output_dir / "goal_policy_action_distribution.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].hist(sample_std_by_state, bins=30, color="#4E79A7", alpha=0.9)
    axes[0].set_xlabel("Mean sampled action std per state")
    axes[0].set_ylabel("State count")
    axes[0].set_title("Post-Tanh Commitment Check")
    axes[1].boxplot([pre_tanh_std[:, i] for i in range(ACTION_DIM)], positions=x, showfliers=False)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(ACTION_COLUMNS, rotation=45)
    axes[1].set_ylabel("Pre-tanh std")
    axes[1].set_title("Learned Gaussian Std by Motor")
    fig.tight_layout()
    path = output_dir / "goal_policy_std_commitment.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    return paths


def default_output_dir(checkpoint_path: Path, role: str, terrain: str) -> Path:
    run_dir = checkpoint_path.parent.parent
    episode_label = checkpoint_path.stem
    terrain_label = terrain if terrain else "all"
    return run_dir / "policy_distribution" / f"{episode_label}_{role}_{terrain_label}"


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = load_torch_payload(checkpoint_path)
    replay_path = args.replay_npz or default_replay_path(checkpoint, args.role, checkpoint_path)
    if replay_path is None:
        raise ValueError("Could not infer replay snapshot. Pass --replay-npz.")
    replay_path = replay_path if replay_path.is_absolute() else (Path.cwd() / replay_path)
    if not replay_path.exists():
        raise FileNotFoundError(replay_path)

    output_dir = (args.output_dir or default_output_dir(checkpoint_path, args.role, args.terrain)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    observations, replay_actions = load_replay_states(
        replay_path=replay_path,
        terrain=args.terrain,
        num_states=args.num_states,
        seed=args.seed,
    )
    agent = make_agent(checkpoint, args.role, args.device)
    distribution = policy_distribution(agent, observations, args.samples_per_state)
    summary = summarize(distribution, replay_actions, agent)
    summary.update(
        {
            "checkpoint": str(checkpoint_path),
            "role": args.role,
            "terrain_filter": args.terrain,
            "replay_npz": str(replay_path),
        }
    )

    summary_csv = output_dir / "policy_distribution_summary.csv"
    motor_csv = output_dir / "policy_distribution_by_motor.csv"
    write_summary_csv(summary_csv, summary)
    write_motor_csv(motor_csv, distribution)
    plots = plot_distribution(output_dir, distribution)
    (output_dir / "policy_distribution_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Replay: {replay_path}")
    print(f"Output directory: {output_dir}")
    print(
        "Commitment check: "
        f"post_tanh_sample_std_mean={summary['post_tanh_sample_std_mean']:.4f}, "
        f"pre_tanh_std_mean={summary['pre_tanh_std_mean']:.4f}, "
        f"MC entropy={summary['mc_tanh_entropy_mean']:.4f}, "
        f"alpha={summary['alpha']:.5f}"
    )
    print(f"Wrote {summary_csv}")
    print(f"Wrote {motor_csv}")
    for path in plots:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
