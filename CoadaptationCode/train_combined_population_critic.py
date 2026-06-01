import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from goal_research_training import (  # noqa: E402
    ACTION_DIM,
    CONDITIONED_OBSERVATION_DIM,
    CONDITION_DIM,
    DEFAULT_REPLAY_SIZE,
    HIDDEN_SIZES,
    OBSERVATION_DIM,
    OBSERVATION_LABELS,
    CONDITION_LABELS,
    GoalReplayBuffer,
    GoalSACAgent,
    load_torch_payload,
    set_global_seeds,
    write_json_atomic,
)


def latest_checkpoint(run_dir: Path) -> Path:
    checkpoints = sorted((run_dir / "checkpoints").glob("episode_*_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {run_dir / 'checkpoints'}")
    return checkpoints[-1]


def replay_from_run_dir(run_dir: Path) -> Path:
    replay_path = run_dir / "replay" / "population_replay_latest.npz"
    if not replay_path.exists():
        raise FileNotFoundError(f"No population replay found: {replay_path}")
    return replay_path


def active_indices(replay: GoalReplayBuffer) -> np.ndarray:
    if replay.size <= 0:
        return np.asarray([], dtype=np.int64)
    if replay.size < replay.capacity:
        return np.arange(replay.size, dtype=np.int64)
    return np.arange(replay.capacity, dtype=np.int64)


def load_replay(path: Path) -> GoalReplayBuffer:
    replay = GoalReplayBuffer(capacity=1)
    replay.load_npz(path)
    return replay


def append_replay(target: GoalReplayBuffer, source: GoalReplayBuffer) -> int:
    count = 0
    for idx in active_indices(source):
        target.add_sample(
            observation=source.robot_observations[idx],
            action=source.actions[idx],
            reward=float(source.rewards[idx, 0]),
            next_observation=source.robot_next_observations[idx],
            terminal=bool(source.terminals[idx, 0]),
            condition=source.conditions[idx],
            next_condition=source.next_conditions[idx],
            terrain_id=int(source.terrain_ids[idx, 0]),
            design_id=int(source.design_ids[idx, 0]),
        )
        count += 1
    return count


def merge_replays(paths: Iterable[Path]) -> tuple[GoalReplayBuffer, List[Dict[str, object]]]:
    loaded = []
    source_summaries = []
    total_size = 0
    for path in paths:
        replay = load_replay(path)
        loaded.append((path, replay))
        total_size += len(replay)
        ids, counts = np.unique(replay.design_ids[active_indices(replay)].reshape(-1), return_counts=True)
        source_summaries.append(
            {
                "path": str(path),
                "size": int(len(replay)),
                "design_id_counts": {str(int(i)): int(c) for i, c in zip(ids, counts)},
            }
        )

    if total_size <= 0:
        raise RuntimeError("All input population replay buffers are empty.")

    merged = GoalReplayBuffer(
        capacity=max(total_size, DEFAULT_REPLAY_SIZE),
        obs_dim=OBSERVATION_DIM,
        action_dim=ACTION_DIM,
        condition_dim=CONDITION_DIM,
    )
    for _, replay in loaded:
        append_replay(merged, replay)

    return merged, source_summaries


def agent_from_checkpoint(checkpoint: Dict[str, object], args: argparse.Namespace) -> GoalSACAgent:
    payload = checkpoint["population_agent"]
    hparams = payload.get("hyperparameters", {}) if isinstance(payload, dict) else {}
    agent = GoalSACAgent(
        obs_dim=int(hparams.get("obs_dim", CONDITIONED_OBSERVATION_DIM)),
        action_dim=int(hparams.get("action_dim", ACTION_DIM)),
        hidden_sizes=hparams.get("hidden_sizes", HIDDEN_SIZES),
        lr=float(args.learning_rate if args.learning_rate is not None else hparams.get("lr", 1e-3)),
        gamma=float(args.gamma if args.gamma is not None else hparams.get("gamma", 0.99)),
        tau=float(args.tau if args.tau is not None else hparams.get("tau", 0.01)),
        alpha_init=float(hparams.get("alpha_init", 0.01)),
        target_entropy=float(hparams.get("target_entropy_internal", -float(ACTION_DIM))),
        grad_clip_value=float(hparams.get("grad_clip_value", 1.0)),
    )
    agent.load_state_dict_payload(payload, load_optimizers=bool(args.load_optimizers))
    return agent


def write_csv_rows(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_inputs(args: argparse.Namespace) -> tuple[Path, List[Path]]:
    replay_paths = [Path(path).resolve() for path in args.population_replay_npz]
    run_dirs = [Path(path).resolve() for path in args.run_dir]
    replay_paths.extend(replay_from_run_dir(run_dir).resolve() for run_dir in run_dirs)

    if args.seed_checkpoint:
        seed_checkpoint = Path(args.seed_checkpoint).resolve()
    elif args.seed_run_dir:
        seed_checkpoint = latest_checkpoint(Path(args.seed_run_dir).resolve()).resolve()
    elif run_dirs:
        seed_checkpoint = latest_checkpoint(run_dirs[0]).resolve()
    else:
        raise ValueError("Provide --seed-checkpoint, --seed-run-dir, or at least one --run-dir.")

    if not seed_checkpoint.exists():
        raise FileNotFoundError(f"Seed checkpoint does not exist: {seed_checkpoint}")
    if not replay_paths:
        raise ValueError("Provide at least one --population-replay-npz or --run-dir.")

    unique_replay_paths = []
    seen = set()
    for path in replay_paths:
        if not path.exists():
            raise FileNotFoundError(f"Population replay does not exist: {path}")
        key = str(path).lower()
        if key not in seen:
            unique_replay_paths.append(path)
            seen.add(key)
    return seed_checkpoint, unique_replay_paths


def train_offline(agent: GoalSACAgent, replay: GoalReplayBuffer, args: argparse.Namespace) -> List[Dict[str, object]]:
    rows = []
    for update_idx in range(1, int(args.updates) + 1):
        batch = replay.random_batch(int(args.batch_size))
        diagnostics = agent.train_step(batch)
        row = {
            "update": update_idx,
            "replay_size": len(replay),
        }
        row.update(diagnostics)
        rows.append(row)
        if args.log_every > 0 and (update_idx == 1 or update_idx % args.log_every == 0):
            print(
                f"update {update_idx}/{args.updates}: "
                f"qf1_loss={diagnostics['qf1_loss']:.5g}, "
                f"qf2_loss={diagnostics['qf2_loss']:.5g}, "
                f"policy_loss={diagnostics['policy_loss']:.5g}, "
                f"alpha={diagnostics['alpha']:.5g}"
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge population replay buffers and offline-train a combined population SAC critic."
    )
    parser.add_argument("--run-dir", action="append", default=[], help="Run directory containing replay/ and checkpoints/.")
    parser.add_argument("--population-replay-npz", action="append", default=[], help="Population replay snapshot to merge.")
    parser.add_argument("--seed-checkpoint", type=Path, help="Checkpoint whose population agent initializes training.")
    parser.add_argument("--seed-run-dir", type=Path, help="Run directory whose latest checkpoint initializes training.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--updates", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--load-optimizers", action="store_true")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--log-every", type=int, default=250)
    return parser.parse_args()


def main() -> Dict[str, object]:
    args = parse_args()
    set_global_seeds(int(args.seed))
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else (SCRIPT_DIR / "results_goal_research" / f"{timestamp}_combined_population_critic").resolve()
    )
    replay_dir = output_dir / "replay"
    checkpoint_dir = output_dir / "checkpoints"
    replay_path = replay_dir / "population_replay_latest.npz"
    checkpoint_path = checkpoint_dir / "combined_population_critic.pt"
    diagnostics_path = output_dir / "combined_population_critic_summary.json"
    losses_path = output_dir / "offline_population_losses.csv"

    seed_checkpoint, replay_paths = resolve_inputs(args)
    print(f"Seed checkpoint: {seed_checkpoint}")
    print("Merging population replays:")
    for path in replay_paths:
        print(f"  {path}")

    merged_replay, source_summaries = merge_replays(replay_paths)
    merged_replay.save_npz(replay_path)

    checkpoint = load_torch_payload(seed_checkpoint)
    agent = agent_from_checkpoint(checkpoint, args)
    loss_rows = train_offline(agent, merged_replay, args)
    write_csv_rows(losses_path, loss_rows)

    design_ids, design_counts = np.unique(
        merged_replay.design_ids[active_indices(merged_replay)].reshape(-1),
        return_counts=True,
    )
    terrain_ids, terrain_counts = np.unique(
        merged_replay.terrain_ids[active_indices(merged_replay)].reshape(-1),
        return_counts=True,
    )
    summary = {
        "created_at": timestamp,
        "seed_checkpoint": str(seed_checkpoint),
        "replay_sources": source_summaries,
        "merged_replay": str(replay_path),
        "merged_replay_size": int(len(merged_replay)),
        "design_id_counts": {str(int(i)): int(c) for i, c in zip(design_ids, design_counts)},
        "terrain_id_counts": {str(int(i)): int(c) for i, c in zip(terrain_ids, terrain_counts)},
        "updates": int(args.updates),
        "batch_size": int(args.batch_size),
        "learning_rate": agent.lr,
        "gamma": agent.gamma,
        "tau": agent.tau,
        "load_optimizers": bool(args.load_optimizers),
        "losses_csv": str(losses_path),
        "checkpoint": str(checkpoint_path),
        "final_diagnostics": loss_rows[-1] if loss_rows else {},
    }
    write_json_atomic(diagnostics_path, summary)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "episode": 0,
            "terrain": "offline_combined",
            "episode_in_terrain": 0,
            "terrain_completed": True,
            "run_data": {
                "population_replay": str(replay_path),
                "offline_combined_summary": str(diagnostics_path),
                "offline_population_losses": str(losses_path),
                "source_replays": [str(path) for path in replay_paths],
                "seed_checkpoint": str(seed_checkpoint),
                "offline_updates": int(args.updates),
            },
            "design_id": -1,
            "layout": str(checkpoint.get("layout", "combined_population")),
            "design_vector": checkpoint.get("design_vector", []),
            "population_agent": agent.state_dict(),
            "morphology": checkpoint.get("morphology", {}),
            "observation_labels": OBSERVATION_LABELS,
            "condition_labels": CONDITION_LABELS,
        },
        checkpoint_path,
    )

    print(f"Saved merged replay: {replay_path}")
    print(f"Saved combined checkpoint: {checkpoint_path}")
    print(f"Saved diagnostics: {diagnostics_path}")
    print(f"Saved losses: {losses_path}")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
