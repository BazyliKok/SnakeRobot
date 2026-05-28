#!/usr/bin/env python
"""Plot GOAL research training outputs from a run directory."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ACTION_COLUMNS = [f"Motor{i}_Action" for i in range(1, 8)]
TERRAIN_COLORS = {
    "carpet": "#4E79A7",
    "cardboard": "#F28E2B",
    "foam": "#76B7B2",
    "artificial_grass": "#59A14F",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create summary plots for a GOAL research training run."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Run directory containing training_steps.csv, episode_summary.csv, and losses.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for generated PNG plots. Defaults to '<run-dir>/plots'.",
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=None,
        help="Only plot episodes greater than or equal to this episode.",
    )
    parser.add_argument(
        "--checkpoint-episode",
        type=int,
        default=None,
        help="Also create a standalone XY trajectory plot for one episode.",
    )
    return parser.parse_args()


def candidate_run_dirs() -> Iterable[Path]:
    roots = [
        Path("results_goal_research"),
        Path("CoadaptationCode") / "results_goal_research",
    ]
    for root in roots:
        if root.exists():
            yield from root.iterdir()
    yield from Path.cwd().glob("**/results_goal_research/*")


def latest_run_dir() -> Path:
    candidates = []
    for path in candidate_run_dirs():
        if not path.is_dir():
            continue
        steps = path / "training_steps.csv"
        if steps.exists():
            candidates.append((steps.stat().st_mtime, path))
    if not candidates:
        raise FileNotFoundError(
            "Could not find a run directory with training_steps.csv. Pass --run-dir explicitly."
        )
    return sorted(candidates)[-1][1]


def read_csv(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return pd.DataFrame()
    frame = pd.read_csv(path)
    return frame


def numeric(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def prepare_steps(run_dir: Path, start_from: int | None) -> pd.DataFrame:
    steps = read_csv(run_dir / "training_steps.csv")
    steps = numeric(
        steps,
        [
            "episode",
            "episode_in_terrain",
            "step",
            "reward",
            "position_x_cm",
            "position_y_cm",
            "position_z_cm",
            "start_x_cm",
            "start_y_cm",
            "start_z_cm",
            "forward_curr_cm",
            "distance_error_cm",
            "forward_reward",
            "yaw_reward",
            "yaw_penalty_rad",
            "success_bonus",
            "time_penalty",
            *ACTION_COLUMNS,
        ],
    )
    steps = steps.sort_values(["episode", "step"]).reset_index(drop=True)
    if start_from is not None:
        steps = steps[steps["episode"] >= int(start_from)].copy()
    steps["cumulative_reward"] = steps.groupby("episode")["reward"].cumsum()
    return steps


def prepare_episode_summary(run_dir: Path, steps: pd.DataFrame) -> pd.DataFrame:
    summary_path = run_dir / "episode_summary.csv"
    if summary_path.exists():
        summary = read_csv(summary_path)
        summary = numeric(
            summary,
            [
                "episode",
                "episode_in_terrain",
                "steps",
                "episode_return",
                "forward_progress_cm",
                "remaining_distance_cm",
                "final_z_cm",
            ],
        )
        summary = summary[summary["episode"].isin(steps["episode"].unique())]
    else:
        grouped = steps.groupby("episode", as_index=False)
        summary = grouped.agg(
            terrain=("terrain", "last"),
            episode_in_terrain=("episode_in_terrain", "last"),
            steps=("step", "max"),
            episode_return=("reward", "sum"),
            forward_progress_cm=("forward_curr_cm", "last"),
            remaining_distance_cm=("distance_error_cm", "last"),
            final_z_cm=("position_z_cm", "last"),
        )
    if "terrain" not in summary.columns and "episode" in summary.columns:
        terrain_by_episode = steps.groupby("episode")["terrain"].last()
        summary["terrain"] = summary["episode"].map(terrain_by_episode)
    return summary.sort_values("episode").reset_index(drop=True)


def prepare_losses(run_dir: Path, start_from: int | None) -> pd.DataFrame:
    losses = read_csv(run_dir / "losses.csv", required=False)
    if losses.empty:
        return losses
    losses = numeric(
        losses,
        [
            "episode",
            "update_in_episode",
            "replay_size",
            "qf1_loss",
            "qf2_loss",
            "policy_loss",
            "alpha",
            "policy_mean_abs",
            "policy_log_std_mean",
        ],
    )
    if start_from is not None and "episode" in losses.columns:
        losses = losses[losses["episode"] >= int(start_from)].copy()
    return losses


def terrain_color(terrain: object) -> str:
    return TERRAIN_COLORS.get(str(terrain), "#777777")


def shade_terrain_blocks(ax: plt.Axes, summary: pd.DataFrame) -> None:
    if summary.empty or "terrain" not in summary.columns:
        return
    block_id = (summary["terrain"] != summary["terrain"].shift()).cumsum()
    blocks = summary.assign(block_id=block_id).groupby("block_id", as_index=False).agg(
        terrain=("terrain", "last"),
        first=("episode", "min"),
        last=("episode", "max"),
    )
    for _, row in blocks.iterrows():
        ax.axvspan(
            float(row["first"]) - 0.5,
            float(row["last"]) + 0.5,
            color=terrain_color(row["terrain"]),
            alpha=0.08,
            linewidth=0,
        )


def finish(ax: plt.Axes, xlabel: str = "Episode") -> None:
    ax.grid(True, alpha=0.25)
    ax.set_xlabel(xlabel)


def plot_learning_curve(summary: pd.DataFrame, output_dir: Path) -> Path | None:
    if summary.empty:
        return None
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    x = summary["episode"].to_numpy()
    for ax in axes:
        shade_terrain_blocks(ax, summary)

    axes[0].plot(x, summary["episode_return"], color="#222222", marker="o", lw=1.7)
    axes[0].set_ylabel("Episode return")
    axes[0].set_title("GOAL Training Return")
    finish(axes[0], xlabel="")

    if "forward_progress_cm" in summary.columns:
        axes[1].plot(x, summary["forward_progress_cm"], color="#4E79A7", marker="o", lw=1.7)
        axes[1].set_ylabel("Forward progress (cm)")
    finish(axes[1])

    fig.tight_layout()
    path = output_dir / "goal_learning_curve.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_action_heatmap(steps: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> Path | None:
    action_cols = [column for column in ACTION_COLUMNS if column in steps.columns]
    if not action_cols:
        return None
    actions = steps.groupby("episode")[action_cols].agg(lambda values: values.abs().mean())
    actions = actions.reindex(summary["episode"])

    fig, ax = plt.subplots(figsize=(12, 4.8))
    image = ax.imshow(actions.T.to_numpy(), aspect="auto", cmap="magma", interpolation="nearest")
    ax.set_yticks(np.arange(len(action_cols)))
    ax.set_yticklabels([column.replace("_Action", "") for column in action_cols])
    ax.set_xticks(np.arange(len(actions.index)))
    ax.set_xticklabels([str(int(ep)) for ep in actions.index], rotation=90)
    ax.set_xlabel("Episode")
    ax.set_title("Mean Absolute Motor Action")
    fig.colorbar(image, ax=ax, label="|action|")
    fig.tight_layout()
    path = output_dir / "goal_action_heatmap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_trajectories(steps: pd.DataFrame, output_dir: Path) -> Path | None:
    required = {"position_x_cm", "position_y_cm", "start_x_cm", "start_y_cm"}
    if not required.issubset(steps.columns):
        return None
    terrains = list(dict.fromkeys(steps["terrain"].dropna().astype(str)))
    if not terrains:
        terrains = ["all"]
    fig, axes = plt.subplots(1, len(terrains), figsize=(6.5 * len(terrains), 5.4), squeeze=False)
    for ax, terrain in zip(axes[0], terrains):
        frame = steps if terrain == "all" else steps[steps["terrain"].astype(str) == terrain]
        episodes = sorted(frame["episode"].dropna().unique())
        colors = plt.cm.viridis(np.linspace(0.15, 0.95, max(len(episodes), 1)))
        for color, episode in zip(colors, episodes):
            ep_frame = frame[frame["episode"] == episode].sort_values("step")
            x = ep_frame["position_x_cm"] - ep_frame["start_x_cm"].iloc[0]
            y = ep_frame["position_y_cm"] - ep_frame["start_y_cm"].iloc[0]
            ax.plot(x, y, lw=1.5, color=color, label=f"ep {int(episode)}")
        ax.set_title(str(terrain).replace("_", " ").title())
        ax.set_xlabel("X displacement (cm)")
        ax.set_ylabel("Y displacement (cm)")
        ax.axis("equal")
        ax.grid(True, alpha=0.25)
        if len(episodes) <= 12:
            ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.suptitle("Episode-Relative XY Trajectories", y=1.02)
    fig.tight_layout()
    path = output_dir / "goal_trajectories_xy.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_reward_components(steps: pd.DataFrame, output_dir: Path) -> Path | None:
    columns = [
        column
        for column in ["forward_reward", "yaw_reward", "success_bonus", "time_penalty"]
        if column in steps.columns
    ]
    if not columns:
        return None
    components = steps.groupby("episode")[columns].mean()
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for column in columns:
        ax.plot(components.index, components[column], marker="o", lw=1.4, label=column)
    ax.set_ylabel("Mean component per step")
    ax.set_title("Reward Components")
    ax.legend(frameon=False)
    finish(ax)
    fig.tight_layout()
    path = output_dir / "goal_reward_components.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_losses(losses: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> Path | None:
    if losses.empty:
        return None
    grouped = losses.groupby(["episode", "role"], as_index=False).agg(
        qf1_loss=("qf1_loss", "mean"),
        qf2_loss=("qf2_loss", "mean"),
        policy_loss=("policy_loss", "mean"),
        alpha=("alpha", "last"),
        policy_mean_abs=("policy_mean_abs", "mean"),
        policy_log_std_mean=("policy_log_std_mean", "mean"),
    )
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for ax in axes:
        shade_terrain_blocks(ax, summary)

    for role, frame in grouped.groupby("role"):
        axes[0].plot(frame["episode"], frame["qf1_loss"], marker="o", lw=1.2, label=f"{role} q1")
        axes[0].plot(frame["episode"], frame["qf2_loss"], marker="o", lw=1.2, linestyle="--", label=f"{role} q2")
        axes[1].plot(frame["episode"], frame["policy_loss"], marker="o", lw=1.3, label=str(role))
        axes[2].plot(frame["episode"], frame["alpha"], marker="o", lw=1.3, label=f"{role} alpha")

    axes[0].set_ylabel("Q loss")
    axes[1].set_ylabel("Policy loss")
    axes[2].set_ylabel("Alpha")
    for ax in axes:
        ax.legend(frameon=False)
        finish(ax, xlabel="")
    finish(axes[2])
    fig.suptitle("SAC Training Diagnostics", y=0.995)
    fig.tight_layout()
    path = output_dir / "goal_training_losses.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_checkpoint_trajectory(steps: pd.DataFrame, episode: int, output_dir: Path) -> Path | None:
    ep_frame = steps[steps["episode"] == int(episode)].sort_values("step")
    required = {"position_x_cm", "position_y_cm", "start_x_cm", "start_y_cm"}
    if ep_frame.empty or not required.issubset(ep_frame.columns):
        print(f"Warning: no trajectory data found for episode {episode}.")
        return None
    fig, ax = plt.subplots(figsize=(7.5, 6))
    x = ep_frame["position_x_cm"] - ep_frame["start_x_cm"].iloc[0]
    y = ep_frame["position_y_cm"] - ep_frame["start_y_cm"].iloc[0]
    terrain = str(ep_frame["terrain"].iloc[-1]) if "terrain" in ep_frame.columns else "unknown"
    ax.plot(x, y, lw=2.2, color=terrain_color(terrain))
    ax.scatter(x.iloc[0], y.iloc[0], color="#222222", label="start", zorder=3)
    ax.scatter(x.iloc[-1], y.iloc[-1], color="#E15759", label="end", zorder=3)
    ax.set_title(f"Episode {episode} Trajectory ({terrain})")
    ax.set_xlabel("X displacement (cm)")
    ax.set_ylabel("Y displacement (cm)")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = output_dir / f"goal_episode_{int(episode):03d}_trajectory.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_summary_csv(summary: pd.DataFrame, output_dir: Path) -> Path:
    path = output_dir / "goal_episode_summary.csv"
    summary.to_csv(path, index=False)
    return path


def main() -> None:
    args = parse_args()
    run_dir = (args.run_dir or latest_run_dir()).resolve()
    output_dir = (args.output_dir or run_dir / "plots").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    steps = prepare_steps(run_dir, args.start_from)
    summary = prepare_episode_summary(run_dir, steps)
    losses = prepare_losses(run_dir, args.start_from)
    summary_csv = write_summary_csv(summary, output_dir)

    plots = [
        plot_learning_curve(summary, output_dir),
        plot_action_heatmap(steps, summary, output_dir),
        plot_trajectories(steps, output_dir),
        plot_reward_components(steps, output_dir),
        plot_losses(losses, summary, output_dir),
    ]
    if args.checkpoint_episode is not None:
        plots.append(plot_checkpoint_trajectory(steps, args.checkpoint_episode, output_dir))

    print(f"Run directory: {run_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Episodes plotted: {summary['episode'].min()}-{summary['episode'].max()} ({summary['episode'].nunique()} total)")
    print(f"Wrote summary CSV: {summary_csv}")
    print("Created plots:")
    for plot in plots:
        if plot is not None:
            print(f"  {plot}")


if __name__ == "__main__":
    main()
