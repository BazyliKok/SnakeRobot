#!/usr/bin/env python
"""Create summary plots for mixed-terrain snake robot training logs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REWARD_COMPONENTS = [
    "Progress_Reward",
    "X_Drift_Penalty",
    "Heading_Penalty",
    "Living_Penalty",
    "No_Progress_Penalty",
    "Backward_Penalty",
]

ACTION_COLUMNS = [f"Motor{i}_Action" for i in range(1, 8)]

TERRAIN_COLORS = {
    "artificial_grass": "#59A14F",
    "carpet": "#4E79A7",
    "foam": "#76B7B2",
    "cardboard": "#F28E2B",
}

DEFAULT_COLORS = [
    "#59A14F",
    "#4E79A7",
    "#76B7B2",
    "#F28E2B",
    "#E15759",
    "#B07AA1",
    "#9C755F",
]

REWARD_COMPONENT_SCALES = {
    "sum_Progress_Reward": 1.0,
    "sum_X_Drift_Penalty": -0.10,
    "sum_Heading_Penalty": -0.05,
    "sum_Living_Penalty": -1.0,
    "sum_No_Progress_Penalty": -1.0,
    "sum_Backward_Penalty": -1.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot reward/loss logs from snake robot mixed-terrain runs."
    )
    parser.add_argument(
        "--prefix",
        default="2026_04_20",
        help="Date/file prefix to search for, for example 2026_04_16.",
    )
    parser.add_argument(
        "--reward-file",
        type=Path,
        help="Reward CSV path. Defaults to '<prefix>Rewards*' in the current folder.",
    )
    parser.add_argument(
        "--loss-file",
        type=Path,
        help="Loss CSV path. Defaults to '<prefix>Losses*' in the current folder.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Folder for PNG plots and CSV summaries.",
    )
    parser.add_argument(
        "--duplicate-policy",
        choices=("latest", "first"),
        default="latest",
        help=(
            "How to choose a segment when an episode index appears under multiple "
            "Run_ID values."
        ),
    )
    return parser.parse_args()


def find_log_file(prefix: str, kind: str) -> Path:
    matches = sorted(Path.cwd().glob(f"{prefix}{kind}*"))
    valid_suffixes = {"", ".csv", ".txt"}
    matches = [
        path
        for path in matches
        if path.is_file() and path.suffix.lower() in valid_suffixes
    ]
    if not matches:
        raise FileNotFoundError(f"No file found matching {prefix}{kind}*")
    if len(matches) > 1:
        names = ", ".join(path.name for path in matches)
        raise ValueError(f"Multiple {kind.lower()} files match {prefix}: {names}")
    return matches[0]


def sort_logs(frame: pd.DataFrame) -> pd.DataFrame:
    cols = [col for col in ("Episode", "Run_ID", "Timestep") if col in frame.columns]
    return frame.sort_values(cols).reset_index(drop=True)


def segment_summary(frame: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["Episode", "Run_ID"]
    aggregations = {
        "rows": ("Episode", "size"),
    }
    if "Timestep" in frame:
        aggregations.update(
            first_timestep=("Timestep", "min"),
            last_timestep=("Timestep", "max"),
        )
    if "Terrain" in frame:
        aggregations["terrain"] = ("Terrain", "last")
    if "Terrain_Block_Index" in frame:
        aggregations["terrain_block"] = ("Terrain_Block_Index", "last")
    if "Cumulative_Rewards" in frame:
        aggregations["final_cumulative_reward"] = ("Cumulative_Rewards", "last")
    if "Rewards" in frame:
        aggregations["sum_reward"] = ("Rewards", "sum")
    if "Distance_Progress_Cm" in frame:
        aggregations["sum_progress_cm"] = ("Distance_Progress_Cm", "sum")
    if "Raw_Distance_Progress_Cm" in frame:
        aggregations["sum_raw_progress_cm"] = ("Raw_Distance_Progress_Cm", "sum")

    return frame.groupby(group_cols, as_index=False).agg(**aggregations)


def choose_duplicate_segments(
    frame: pd.DataFrame, policy: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    segments = segment_summary(frame)
    duplicate_segments = segments[
        segments.duplicated("Episode", keep=False)
    ].copy()

    ordered = segments.sort_values(["Episode", "Run_ID"])
    if policy == "latest":
        selected = ordered.groupby("Episode", as_index=False).tail(1)
    else:
        selected = ordered.groupby("Episode", as_index=False).head(1)

    selected_keys = set(zip(selected["Episode"], selected["Run_ID"]))
    selected_mask = [
        (episode, run_id) in selected_keys
        for episode, run_id in zip(frame["Episode"], frame["Run_ID"])
    ]
    cleaned = frame.loc[selected_mask].copy()

    if not duplicate_segments.empty:
        duplicate_segments["selected"] = [
            (episode, run_id) in selected_keys
            for episode, run_id in zip(
                duplicate_segments["Episode"], duplicate_segments["Run_ID"]
            )
        ]

    return sort_logs(cleaned), duplicate_segments.sort_values(["Episode", "Run_ID"])


def add_optional_agg(aggs: dict, frame: pd.DataFrame, col: str, out_col: str, func):
    if col in frame.columns:
        aggs[out_col] = (col, func)


def make_episode_summary(rewards: pd.DataFrame) -> pd.DataFrame:
    rewards = rewards.sort_values(["Episode", "Timestep"]).copy()

    aggs = {
        "run_id": ("Run_ID", "last"),
        "steps": ("Timestep", "count"),
        "last_timestep": ("Timestep", "max"),
        "terrain": ("Terrain", "last"),
        "terrain_id": ("Terrain_ID", "last"),
        "terrain_block": ("Terrain_Block_Index", "last"),
        "episode_in_block": ("Episode_In_Terrain_Block", "last"),
        "final_cumulative_reward": ("Cumulative_Rewards", "last"),
        "sum_reward": ("Rewards", "sum"),
        "mean_reward_per_step": ("Rewards", "mean"),
        "sum_progress_cm": ("Distance_Progress_Cm", "sum"),
        "sum_raw_progress_cm": ("Raw_Distance_Progress_Cm", "sum"),
        "mean_progress_reward": ("Progress_Reward", "mean"),
        "sum_progress_reward": ("Progress_Reward", "sum"),
        "start_x": ("X_Position", "first"),
        "final_x": ("X_Position", "last"),
        "start_y": ("Y_Position", "first"),
        "final_y": ("Y_Position", "last"),
        "mean_abs_x_position": ("X_Position", lambda series: series.abs().mean()),
        "mean_abs_y_position": ("Y_Position", lambda series: series.abs().mean()),
        "no_progress_steps": ("No_Progress_Penalty", lambda s: int((s > 0).sum())),
        "backward_steps": ("Backward_Penalty", lambda s: int((s > 0).sum())),
    }
    for col in REWARD_COMPONENTS:
        if col in rewards.columns:
            aggs[f"sum_{col}"] = (col, "sum")
            aggs[f"mean_{col}"] = (col, "mean")
    for col in ACTION_COLUMNS:
        if col in rewards.columns:
            aggs[f"mean_abs_{col}"] = (col, lambda s: s.abs().mean())

    summary = rewards.groupby("Episode", as_index=False).agg(**aggs)
    summary["delta_x"] = summary["final_x"] - summary["start_x"]
    summary["delta_y"] = summary["final_y"] - summary["start_y"]
    summary["reward_sum_delta"] = (
        summary["final_cumulative_reward"] - summary["sum_reward"]
    )
    return summary.sort_values("Episode").reset_index(drop=True)


def make_terrain_summary(episode_summary: pd.DataFrame) -> pd.DataFrame:
    grouped = episode_summary.groupby(["terrain_block", "terrain"], as_index=False)
    summary = grouped.agg(
        episodes=("Episode", "count"),
        first_episode=("Episode", "min"),
        last_episode=("Episode", "max"),
        mean_reward=("final_cumulative_reward", "mean"),
        median_reward=("final_cumulative_reward", "median"),
        std_reward=("final_cumulative_reward", "std"),
        best_reward=("final_cumulative_reward", "max"),
        worst_reward=("final_cumulative_reward", "min"),
        mean_progress_cm=("sum_progress_cm", "mean"),
        mean_raw_progress_cm=("sum_raw_progress_cm", "mean"),
        mean_steps=("steps", "mean"),
        mean_no_progress_steps=("no_progress_steps", "mean"),
        mean_backward_steps=("backward_steps", "mean"),
        mean_delta_x=("delta_x", "mean"),
        mean_delta_y=("delta_y", "mean"),
    )
    return summary.sort_values("terrain_block").reset_index(drop=True)


def terrain_order(episode_summary: pd.DataFrame) -> list[str]:
    ordered = (
        episode_summary[["terrain_block", "terrain"]]
        .drop_duplicates()
        .sort_values("terrain_block")
    )
    return ordered["terrain"].tolist()


def color_for_terrain(terrain: str, index: int = 0) -> str:
    return TERRAIN_COLORS.get(terrain, DEFAULT_COLORS[index % len(DEFAULT_COLORS)])


def shade_terrain_blocks(ax: plt.Axes, episode_summary: pd.DataFrame) -> None:
    blocks = (
        episode_summary.groupby(["terrain_block", "terrain"], as_index=False)
        .agg(first=("Episode", "min"), last=("Episode", "max"))
        .sort_values("terrain_block")
    )
    for idx, row in blocks.iterrows():
        color = color_for_terrain(row["terrain"], idx)
        ax.axvspan(row["first"] - 0.5, row["last"] + 0.5, color=color, alpha=0.08)
        label = str(row["terrain"]).replace("_", " ")
        center = (row["first"] + row["last"]) / 2.0
        ax.text(
            center,
            0.98,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9,
            color="#333333",
        )
        if idx > 0:
            ax.axvline(row["first"] - 0.5, color="#555555", lw=0.8, alpha=0.45)


def finish_axis(ax: plt.Axes, xlabel: str | None = "Episode index") -> None:
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.grid(True, color="#DDDDDD", linewidth=0.8, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_episode_overview(summary: pd.DataFrame, output_dir: Path) -> Path:
    x = summary["Episode"]
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)

    ax = axes[0]
    shade_terrain_blocks(ax, summary)
    ax.plot(x, summary["final_cumulative_reward"], color="#222222", lw=1.8)
    ax.scatter(
        x,
        summary["final_cumulative_reward"],
        c=[color_for_terrain(t) for t in summary["terrain"]],
        s=42,
        edgecolor="white",
        linewidth=0.7,
        zorder=3,
    )
    ax.axhline(0, color="#777777", lw=0.9, alpha=0.8)
    ax.set_ylabel("Final cumulative reward")
    ax.set_title("Episode performance across terrain changes")
    finish_axis(ax, xlabel=None)

    ax = axes[1]
    shade_terrain_blocks(ax, summary)
    ax.bar(
        x,
        summary["sum_progress_cm"],
        color=[color_for_terrain(t) for t in summary["terrain"]],
        alpha=0.78,
        label="Shaped progress",
    )
    ax.plot(
        x,
        summary["sum_raw_progress_cm"],
        color="#222222",
        marker="o",
        ms=3.5,
        lw=1.2,
        label="Raw progress",
    )
    ax.axhline(0, color="#777777", lw=0.9, alpha=0.8)
    ax.set_ylabel("Total progress (cm)")
    ax.legend(frameon=False, loc="upper right")
    finish_axis(ax, xlabel=None)

    ax = axes[2]
    shade_terrain_blocks(ax, summary)
    ax.bar(
        x,
        summary["no_progress_steps"],
        color="#E15759",
        alpha=0.65,
        label="No-progress steps",
    )
    ax.plot(
        x,
        summary["steps"],
        color="#222222",
        marker="o",
        ms=3.5,
        lw=1.2,
        label="Episode length",
    )
    ax.set_ylabel("Step count")
    ax.legend(frameon=False, loc="upper left")
    finish_axis(ax)

    axes[-1].set_xticks(summary["Episode"])
    axes[-1].tick_params(axis="x", rotation=90)
    fig.tight_layout()
    path = output_dir / "episode_performance_overview.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_terrain_summary(summary: pd.DataFrame, output_dir: Path) -> Path:
    order = terrain_order(summary)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2))
    metrics = [
        ("final_cumulative_reward", "Final cumulative reward"),
        ("sum_progress_cm", "Total progress (cm)"),
        ("no_progress_steps", "No-progress steps"),
    ]

    rng = np.random.default_rng(42)
    for ax, (metric, label) in zip(axes, metrics):
        data = [summary.loc[summary["terrain"] == terrain, metric].values for terrain in order]
        box = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False)
        for i, patch in enumerate(box["boxes"]):
            patch.set_facecolor(color_for_terrain(order[i], i))
            patch.set_alpha(0.26)
            patch.set_edgecolor(color_for_terrain(order[i], i))
        for part in ("whiskers", "caps", "medians"):
            for item in box[part]:
                item.set_color("#333333")
        for i, values in enumerate(data, start=1):
            jitter = rng.normal(0, 0.045, len(values))
            ax.scatter(
                np.full(len(values), i) + jitter,
                values,
                color=color_for_terrain(order[i - 1], i - 1),
                s=32,
                alpha=0.86,
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
        ax.axhline(0, color="#777777", lw=0.9, alpha=0.65)
        ax.set_xticks(range(1, len(order) + 1), [name.replace("_", " ") for name in order])
        ax.tick_params(axis="x", rotation=25)
        ax.set_ylabel(label)
        finish_axis(ax, xlabel=None)

    fig.suptitle("Per-terrain episode distributions", y=1.02)
    fig.tight_layout()
    path = output_dir / "terrain_block_distributions.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_trajectories(rewards: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> Path:
    blocks = (
        summary[["terrain_block", "terrain"]]
        .drop_duplicates()
        .sort_values("terrain_block")
        .reset_index(drop=True)
    )
    n_blocks = len(blocks)
    n_cols = min(2, max(1, n_blocks))
    n_rows = int(np.ceil(n_blocks / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5.5 * n_rows))
    axes_array = np.atleast_1d(axes).ravel()

    for ax_idx, (_, block) in enumerate(blocks.iterrows()):
        ax = axes_array[ax_idx]
        block_summary = summary[summary["terrain_block"] == block["terrain_block"]]
        block_rewards = rewards[rewards["Episode"].isin(block_summary["Episode"])]
        color = color_for_terrain(block["terrain"], ax_idx)
        episodes = sorted(block_rewards["Episode"].unique())
        alphas = np.linspace(0.45, 0.95, max(len(episodes), 1))

        for ep, alpha in zip(episodes, alphas):
            ep_frame = block_rewards[block_rewards["Episode"] == ep].sort_values("Timestep")
            x_relative = ep_frame["X_Position"] - ep_frame["X_Position"].iloc[0]
            y_relative = ep_frame["Y_Position"] - ep_frame["Y_Position"].iloc[0]
            ax.plot(
                x_relative,
                y_relative,
                color=color,
                alpha=float(alpha),
                lw=1.35,
                label=str(ep),
            )
            ax.scatter(
                x_relative.iloc[0],
                y_relative.iloc[0],
                color=color,
                marker="o",
                s=18,
                alpha=float(alpha),
            )
            ax.scatter(
                x_relative.iloc[-1],
                y_relative.iloc[-1],
                color=color,
                marker="x",
                s=28,
                alpha=float(alpha),
            )

        ax.set_title(str(block["terrain"]).replace("_", " "))
        ax.set_xlabel("X displacement from episode start")
        ax.set_ylabel("Y displacement from episode start")
        ax.set_aspect("equal", adjustable="box")
        ax.legend(title="Episode", ncol=2, fontsize=7, title_fontsize=8, frameon=False)
        finish_axis(ax, xlabel="X displacement from episode start")

    for ax in axes_array[n_blocks:]:
        ax.axis("off")

    fig.suptitle("Episode-relative XY trajectories by terrain", y=1.02)
    fig.tight_layout()
    path = output_dir / "trajectories_by_terrain.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_reward_components(summary: pd.DataFrame, output_dir: Path) -> Path:
    order_frame = (
        summary[["terrain_block", "terrain"]]
        .drop_duplicates()
        .sort_values("terrain_block")
        .reset_index(drop=True)
    )
    component_cols = [f"sum_{col}" for col in REWARD_COMPONENTS if f"sum_{col}" in summary]
    component_means = (
        summary.groupby(["terrain_block", "terrain"], as_index=False)[component_cols]
        .mean()
        .sort_values("terrain_block")
    )

    x = np.arange(len(component_means))
    fig, ax = plt.subplots(figsize=(12, 6))
    positive_bottom = np.zeros(len(component_means))
    negative_bottom = np.zeros(len(component_means))

    labels_and_colors = [
        ("sum_Progress_Reward", "Progress reward", "#59A14F"),
        ("sum_X_Drift_Penalty", "X drift penalty", "#E15759"),
        ("sum_Heading_Penalty", "Heading penalty", "#B07AA1"),
        ("sum_Living_Penalty", "Living penalty", "#9C755F"),
        ("sum_No_Progress_Penalty", "No-progress penalty", "#F28E2B"),
        ("sum_Backward_Penalty", "Backward penalty", "#4E79A7"),
    ]

    for col, label, color in labels_and_colors:
        if col not in component_means:
            continue
        values = (
            component_means[col].to_numpy(dtype=float)
            * REWARD_COMPONENT_SCALES.get(col, 1.0)
        )
        positive_mask = values >= 0
        negative_mask = ~positive_mask

        if np.any(positive_mask):
            ax.bar(
                x[positive_mask],
                values[positive_mask],
                bottom=positive_bottom[positive_mask],
                color=color,
                alpha=0.78,
                label=label,
            )
            positive_bottom[positive_mask] += values[positive_mask]

        if np.any(negative_mask):
            ax.bar(
                x[negative_mask],
                values[negative_mask],
                bottom=negative_bottom[negative_mask],
                color=color,
                alpha=0.78,
                label="_nolegend_",
            )
            negative_bottom[negative_mask] += values[negative_mask]

    net = (
        summary.groupby(["terrain_block", "terrain"], as_index=False)[
            "final_cumulative_reward"
        ]
        .mean()
        .sort_values("terrain_block")["final_cumulative_reward"]
        .to_numpy()
    )
    ax.plot(x, net, color="#222222", marker="o", lw=1.6, label="Net reward")
    ax.axhline(0, color="#777777", lw=0.9, alpha=0.8)
    ax.set_xticks(
        x,
        [name.replace("_", " ") for name in order_frame["terrain"]],
        rotation=25,
        ha="right",
    )
    ax.set_ylabel("Mean episode contribution to reward")
    ax.set_title("Mean reward components by terrain block (scaled)")
    ax.legend(frameon=False, ncol=2)
    finish_axis(ax, xlabel=None)
    fig.tight_layout()
    path = output_dir / "reward_component_breakdown.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_action_heatmap(rewards: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> Path | None:
    action_cols = [col for col in ACTION_COLUMNS if col in rewards.columns]
    if not action_cols:
        return None

    actions = rewards.groupby("Episode")[action_cols].agg(lambda s: s.abs().mean())
    actions = actions.reindex(summary["Episode"])

    fig, ax = plt.subplots(figsize=(13, 4.8))
    image = ax.imshow(actions.T, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(action_cols)), [col.replace("_Action", "") for col in action_cols])
    ax.set_xticks(range(len(actions.index)), actions.index)
    ax.tick_params(axis="x", rotation=90)
    ax.set_xlabel("Episode index")
    ax.set_title("Mean absolute motor action per episode")

    blocks = (
        summary.groupby(["terrain_block", "terrain"], as_index=False)
        .agg(first=("Episode", "min"), last=("Episode", "max"))
        .sort_values("terrain_block")
    )
    episode_to_pos = {episode: i for i, episode in enumerate(actions.index)}
    for idx, row in blocks.iterrows():
        first = episode_to_pos[row["first"]]
        last = episode_to_pos[row["last"]]
        center = (first + last) / 2
        ax.text(
            center,
            -0.75,
            str(row["terrain"]).replace("_", " "),
            ha="center",
            va="center",
            fontsize=9,
            color="#333333",
        )
        if idx > 0:
            ax.axvline(first - 0.5, color="white", lw=1.4, alpha=0.95)

    cbar = fig.colorbar(image, ax=ax, shrink=0.88)
    cbar.set_label("Mean absolute action")
    fig.tight_layout()
    path = output_dir / "motor_action_heatmap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_losses(losses: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> Path | None:
    if losses.empty:
        return None

    losses = losses.sort_values(["Episode", "Run_ID"]).copy()
    fig, axes = plt.subplots(3, 1, figsize=(12.5, 9), sharex=True)

    for ax in axes:
        shade_terrain_blocks(ax, summary)

    axes[0].plot(losses["Episode"], losses["Ind_Q1_Loss"], marker="o", lw=1.4, label="Q1")
    axes[0].plot(losses["Episode"], losses["Ind_Q2_Loss"], marker="o", lw=1.4, label="Q2")
    axes[0].set_ylabel("Individual Q loss")
    axes[0].legend(frameon=False)
    finish_axis(axes[0], xlabel=None)

    axes[1].plot(
        losses["Episode"],
        losses["Ind_Policy_Loss"],
        marker="o",
        lw=1.4,
        color="#E15759",
        label="Individual policy",
    )
    axes[1].set_ylabel("Individual policy loss")
    axes[1].legend(frameon=False)
    finish_axis(axes[1], xlabel=None)

    pop_cols = ["Pop_Q1_Loss", "Pop_Q2_Loss", "Pop_Policy_Loss"]
    plotted = False
    for col in pop_cols:
        if col in losses.columns and not np.allclose(losses[col], 0):
            axes[2].plot(losses["Episode"], losses[col], marker="o", lw=1.4, label=col)
            plotted = True
    if plotted:
        axes[2].legend(frameon=False)
        axes[2].set_ylabel("Population loss")
    else:
        axes[2].text(
            0.5,
            0.5,
            "Population losses are zero in this log",
            ha="center",
            va="center",
            transform=axes[2].transAxes,
            color="#555555",
        )
        axes[2].set_ylabel("Population loss")
    finish_axis(axes[2])
    axes[2].set_xticks(summary["Episode"])
    axes[2].tick_params(axis="x", rotation=90)

    fig.suptitle("Training losses", y=0.995)
    fig.tight_layout()
    path = output_dir / "training_losses.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False)


def print_summary(
    reward_file: Path,
    loss_file: Path,
    output_dir: Path,
    rewards: pd.DataFrame,
    cleaned_rewards: pd.DataFrame,
    duplicate_rewards: pd.DataFrame,
    episode_summary: pd.DataFrame,
    terrain_summary: pd.DataFrame,
    plots: Iterable[Path | None],
) -> None:
    print(f"Reward file: {reward_file}")
    print(f"Loss file: {loss_file}")
    print(f"Output directory: {output_dir}")
    print(f"Reward rows: {len(rewards)} raw, {len(cleaned_rewards)} after duplicate handling")
    print(
        "Episodes: "
        f"{episode_summary['Episode'].min()}-{episode_summary['Episode'].max()} "
        f"({episode_summary['Episode'].nunique()} unique)"
    )
    if duplicate_rewards.empty:
        print("Duplicate reward episode segments: none")
    else:
        episodes = ", ".join(map(str, sorted(duplicate_rewards["Episode"].unique())))
        print(f"Duplicate reward episode segments found for: {episodes}")
    print("\nTerrain block summary:")
    cols = [
        "terrain_block",
        "terrain",
        "episodes",
        "mean_reward",
        "median_reward",
        "mean_progress_cm",
        "mean_no_progress_steps",
    ]
    print(terrain_summary[cols].to_string(index=False, float_format=lambda x: f"{x:0.3f}"))
    print("\nCreated plots:")
    for path in plots:
        if path is not None:
            print(f" - {path}")


def main() -> None:
    args = parse_args()
    reward_file = args.reward_file or find_log_file(args.prefix, "Rewards")
    loss_file = args.loss_file or find_log_file(args.prefix, "Losses")
    output_dir = args.output_dir or Path("plots") / f"{args.prefix}_mixed_terrain"
    output_dir.mkdir(parents=True, exist_ok=True)

    rewards = sort_logs(pd.read_csv(reward_file))
    losses_raw = sort_logs(pd.read_csv(loss_file))
    cleaned_rewards, duplicate_rewards = choose_duplicate_segments(
        rewards, args.duplicate_policy
    )
    cleaned_losses, duplicate_losses = choose_duplicate_segments(
        losses_raw, args.duplicate_policy
    )

    episode_summary = make_episode_summary(cleaned_rewards)
    terrain_summary = make_terrain_summary(episode_summary)

    save_csv(episode_summary, output_dir / "episode_summary.csv")
    save_csv(terrain_summary, output_dir / "terrain_block_summary.csv")
    save_csv(duplicate_rewards, output_dir / "duplicate_reward_segments.csv")
    save_csv(duplicate_losses, output_dir / "duplicate_loss_segments.csv")

    plots = [
        plot_episode_overview(episode_summary, output_dir),
        plot_terrain_summary(episode_summary, output_dir),
        plot_trajectories(cleaned_rewards, episode_summary, output_dir),
        plot_reward_components(episode_summary, output_dir),
        plot_action_heatmap(cleaned_rewards, episode_summary, output_dir),
        plot_losses(cleaned_losses, episode_summary, output_dir),
    ]

    print_summary(
        reward_file,
        loss_file,
        output_dir,
        rewards,
        cleaned_rewards,
        duplicate_rewards,
        episode_summary,
        terrain_summary,
        plots,
    )


if __name__ == "__main__":
    main()
