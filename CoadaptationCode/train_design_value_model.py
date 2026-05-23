import argparse
import csv
import json
import os
import sys
from datetime import datetime

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from propose_scale_design import (
    CHECKPOINT_DESIGN_PARAMETER_BOUNDS,
    HETEROGENEOUS_DESIGN_SOURCES,
    INITIAL_DESIGN_SOURCES,
    RESULTS_DIR,
    TERRAINS,
    SnakeEnv,
    actual_width,
    candidate_from_full_design,
    coerce_design_vector,
    design_key,
    load_replay,
    robust_score,
    rollouts_from_buffer,
)


MODEL_DIR = os.path.join(RESULTS_DIR, "design_value_models")


def design_sources_for_mode(design_mode):
    if design_mode == "homogeneous":
        return INITIAL_DESIGN_SOURCES
    if design_mode == "heterogeneous":
        return HETEROGENEOUS_DESIGN_SOURCES
    raise ValueError(f"Unsupported design_mode: {design_mode}")


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def spearman_correlation(reference, values):
    try:
        from scipy.stats import spearmanr
        result = spearmanr(reference, values)
        return float(result.correlation) if np.isfinite(result.correlation) else np.nan
    except Exception:
        return np.nan


def feature_bounds(design_mode):
    return [(float(low), float(high)) for low, high in SnakeEnv.get_optimization_bounds(design_mode)]


def feature_schema(design_mode, bounds):
    if design_mode == "homogeneous":
        design_features = ["width_norm", "angle_norm"]
        optimizer_names = ["width_ratio", "attack_angle_deg"]
    else:
        design_features = ["A_width_norm", "A_angle_norm", "B_width_norm", "B_angle_norm"]
        optimizer_names = ["A_width_ratio", "A_attack_angle_deg", "B_width_ratio", "B_attack_angle_deg"]
    return {
        "design_mode": design_mode,
        "optimizer_parameter_names": optimizer_names,
        "full_design_parameter_names": ["A_width_ratio", "A_attack_angle_deg", "B_width_ratio", "B_attack_angle_deg"],
        "feature_names": design_features + [f"terrain_{terrain}" for terrain in TERRAINS],
        "feature_bounds": [(float(low), float(high)) for low, high in bounds],
    }


def normalize(values, bounds):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    lows = np.asarray([b[0] for b in bounds], dtype=np.float64)
    highs = np.asarray([b[1] for b in bounds], dtype=np.float64)
    return ((values - lows) / np.maximum(highs - lows, 1e-9)).tolist()


def encode_feature_row(row, terrain, bounds, design_mode):
    if design_mode == "homogeneous":
        candidate = [row["width_ratio"], row["attack_angle_deg"]]
    else:
        candidate = [row["A_width_ratio"], row["A_attack_angle_deg"], row["B_width_ratio"], row["B_attack_angle_deg"]]
    return normalize(candidate, bounds) + [1.0 if terrain == t else 0.0 for t in TERRAINS]


def rows_to_xy(rows, bounds, design_mode):
    x = np.asarray([encode_feature_row(row, row["terrain"], bounds, design_mode) for row in rows], dtype=np.float64)
    y = np.asarray([row["return"] for row in rows], dtype=np.float64)
    return x, y


def design_fields(full_design):
    return {
        "A_width_ratio": float(full_design[0]),
        "A_attack_angle_deg": float(full_design[1]),
        "A_actual_width": actual_width(full_design[0]),
        "B_width_ratio": float(full_design[2]),
        "B_attack_angle_deg": float(full_design[3]),
        "B_actual_width": actual_width(full_design[2]),
        "width_ratio": float(full_design[0]),
        "actual_width": actual_width(full_design[0]),
        "attack_angle_deg": float(full_design[1]),
    }


def extract_rollout_rows(design_mode):
    rows = []
    for source in design_sources_for_mode(design_mode):
        replay_path = os.path.join(RESULTS_DIR, source["replay"])
        if not os.path.exists(replay_path):
            raise FileNotFoundError(
                f"Missing replay for {design_mode} design {source['index']}: {replay_path}"
            )
        replay = load_replay(replay_path)
        design_bounds = source.get("design_bounds", CHECKPOINT_DESIGN_PARAMETER_BOUNDS)
        full_design = coerce_design_vector(source["design"], design_bounds)
        target_key = design_key(full_design, design_bounds)
        terrain_buffers = replay.get("population_buffers_by_terrain")
        if terrain_buffers is None:
            terrain_buffers = {TERRAINS[0]: replay["population_buffer"]}

        rollout_count = {terrain: 0 for terrain in TERRAINS}
        for default_terrain, buffer in terrain_buffers.items():
            if default_terrain not in TERRAINS:
                continue
            for rollout in rollouts_from_buffer(buffer, default_terrain, design_bounds):
                terrain = rollout["terrain"]
                if terrain not in TERRAINS or rollout["design_key"] != target_key:
                    continue
                rollout_index = rollout_count[terrain]
                rollout_count[terrain] += 1
                rows.append({
                    "design_index": int(source["index"]),
                    "checkpoint": source["checkpoint"],
                    "tag": source["tag"],
                    "replay": source["replay"],
                    "terrain": terrain,
                    "rollout_index": int(rollout_index),
                    "return": float(rollout["return"]),
                    "length": int(rollout["length"]),
                    **design_fields(full_design),
                })
    return rows


def filter_episode_window_rows(rows, start, end):
    start = int(start)
    end = int(end)
    filtered = []
    for row in rows:
        rollout_index = int(row["rollout_index"])
        if start <= rollout_index < end:
            filtered.append(dict(row, episode_window_start=start, episode_window_end=end))
    return filtered


def filter_balanced_rows(rows, max_episodes):
    max_episodes = int(max_episodes)
    filtered = []
    for design_index in sorted({row["design_index"] for row in rows}):
        for terrain in TERRAINS:
            terrain_rows = sorted(
                [row for row in rows if row["design_index"] == design_index and row["terrain"] == terrain],
                key=lambda row: int(row["rollout_index"]),
            )
            for row in terrain_rows[:max_episodes]:
                filtered.append(dict(row, max_episodes_per_design_terrain=max_episodes))
    return filtered


def aggregate_rows(rows):
    aggregated = []
    for design_index in sorted({row["design_index"] for row in rows}):
        for terrain in TERRAINS:
            group = [row for row in rows if row["design_index"] == design_index and row["terrain"] == terrain]
            if not group:
                continue
            returns = np.asarray([row["return"] for row in group], dtype=np.float64)
            lengths = np.asarray([row["length"] for row in group], dtype=np.float64)
            variance = float(returns.var(ddof=1)) if len(returns) > 1 else 0.0
            template = group[0]
            aggregated.append({
                **{k: template[k] for k in template if k not in {"return", "length", "rollout_index"}},
                "terrain": terrain,
                "rollout_index": -1,
                "return": float(returns.mean()),
                "length": float(lengths.mean()),
                "source_rollout_count": int(len(returns)),
                "return_std": float(returns.std()),
                "return_sample_std": float(np.sqrt(variance)),
                "return_variance": variance,
                "standard_error_variance": float(variance / max(len(returns), 1)),
                "return_min": float(returns.min()),
                "return_max": float(returns.max()),
            })
    return aggregated


def summarize_designs(rows, robustness_lambda):
    terrain_summary = {}
    design_summary = {}
    for design_index in sorted({row["design_index"] for row in rows}):
        design_rows = [row for row in rows if row["design_index"] == design_index]
        terrain_means = {}
        terrain_stds = {}
        terrain_counts = {}
        for terrain in TERRAINS:
            values = np.asarray([row["return"] for row in design_rows if row["terrain"] == terrain], dtype=np.float64)
            terrain_counts[terrain] = int(len(values))
            terrain_means[terrain] = float(values.mean()) if len(values) else np.nan
            terrain_stds[terrain] = float(values.std()) if len(values) else np.nan
            terrain_summary[f"design_{design_index}_{terrain}"] = {
                "design_index": int(design_index),
                "terrain": terrain,
                "count": int(len(values)),
                "mean": float(values.mean()) if len(values) else np.nan,
                "std": float(values.std()) if len(values) else np.nan,
                "min": float(values.min()) if len(values) else np.nan,
                "max": float(values.max()) if len(values) else np.nan,
            }
        valid = [terrain_means[t] for t in TERRAINS if np.isfinite(terrain_means[t])]
        template = design_rows[0]
        design_summary[int(design_index)] = {
            "design_index": int(design_index),
            **{k: template[k] for k in [
                "width_ratio", "actual_width", "attack_angle_deg",
                "A_width_ratio", "A_attack_angle_deg", "A_actual_width",
                "B_width_ratio", "B_attack_angle_deg", "B_actual_width",
            ]},
            "terrain_means": terrain_means,
            "terrain_stds": terrain_stds,
            "terrain_counts": terrain_counts,
            "robustness": robust_score(valid, robustness_lambda),
        }
    return terrain_summary, design_summary


def train_gpr(rows, bounds, design_mode, args):
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern

    x, y = rows_to_xy(rows, bounds, design_mode)
    design_dims = len(bounds)
    kernel = ConstantKernel(1.0, (0.1, 10.0)) * Matern(
        length_scale=np.asarray([0.35] * design_dims + [1.0] * len(TERRAINS), dtype=np.float64),
        length_scale_bounds=(0.12, 5.0),
        nu=2.5,
    )

    alpha_raw = np.asarray([row.get("standard_error_variance", args.gpr_min_alpha) for row in rows], dtype=np.float64)
    y_mean = float(y.mean())
    y_std = float(y.std())
    if not np.isfinite(y_std) or y_std < 1e-9:
        y_std = 1.0
    y_train = (y - y_mean) / y_std
    alpha_train = np.maximum(alpha_raw / (y_std ** 2), float(args.gpr_min_alpha))

    model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha_train,
        normalize_y=False,
        n_restarts_optimizer=int(args.gpr_restarts),
        random_state=int(args.seed),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(x, y_train)

    model._snake_target_normalized = True
    model._snake_target_mean = y_mean
    model._snake_target_std = y_std
    model._snake_alpha_mode = "sem"
    model._snake_alpha_raw = alpha_raw
    model._snake_alpha_train = alpha_train
    return model


def predict(model, x):
    mean, std = model.predict(np.asarray(x, dtype=np.float64), return_std=True)
    mean = np.asarray(mean, dtype=np.float64) * float(model._snake_target_std) + float(model._snake_target_mean)
    std = np.asarray(std, dtype=np.float64) * float(model._snake_target_std)
    return mean, std


def predict_design(model, row, bounds, design_mode, robustness_lambda):
    terrain_means = {}
    terrain_stds = {}
    for terrain in TERRAINS:
        x = [encode_feature_row(row, terrain, bounds, design_mode)]
        mean, std = predict(model, x)
        terrain_means[terrain] = float(mean[0])
        terrain_stds[terrain] = float(std[0])
    values = [terrain_means[t] for t in TERRAINS]
    return {
        "terrain_means": terrain_means,
        "terrain_stds": terrain_stds,
        "robustness": robust_score(values, robustness_lambda),
        "uncertainty": float(np.sqrt(np.mean([terrain_stds[t] ** 2 for t in TERRAINS]) + np.std(values) ** 2)),
    }


def evaluate_known_designs(model, design_summary, bounds, design_mode, robustness_lambda):
    predictions = {}
    for design_index, row in design_summary.items():
        predictions[int(design_index)] = predict_design(model, row, bounds, design_mode, robustness_lambda)
    measured = np.asarray([design_summary[i]["robustness"] for i in sorted(design_summary)], dtype=np.float64)
    predicted = np.asarray([predictions[i]["robustness"] for i in sorted(predictions)], dtype=np.float64)
    return {
        "predictions": predictions,
        "spearman": spearman_correlation(measured, predicted),
        "rmse": float(np.sqrt(np.mean((predicted - measured) ** 2))),
        "measured_best_design": int(sorted(design_summary)[int(np.nanargmax(measured))]),
        "predicted_best_design": int(sorted(predictions)[int(np.nanargmax(predicted))]),
    }


def leave_one_design_out(rows, evaluation_rows, bounds, design_mode, args):
    results = []
    for design_index in sorted({row["design_index"] for row in rows}):
        train_rows = [row for row in rows if row["design_index"] != design_index]
        test_rows = [row for row in evaluation_rows if row["design_index"] == design_index]
        if not train_rows or not test_rows:
            continue
        model = train_gpr(train_rows, bounds, design_mode, args)
        _, measured = summarize_designs(test_rows, args.robustness_lambda)
        predicted = predict_design(model, measured[int(design_index)], bounds, design_mode, args.robustness_lambda)
        x_test, y_test = rows_to_xy(test_rows, bounds, design_mode)
        pred_returns, _ = predict(model, x_test)
        results.append({
            "held_out_design": int(design_index),
            "measured_robustness": float(measured[int(design_index)]["robustness"]),
            "predicted_robustness": float(predicted["robustness"]),
            "robustness_error": float(predicted["robustness"] - measured[int(design_index)]["robustness"]),
            "episode_rmse": float(np.sqrt(np.mean((pred_returns - y_test) ** 2))),
            "predicted_uncertainty": float(predicted["uncertainty"]),
        })
    if not results:
        return [], {"loo_mae": np.nan, "loo_rmse": np.nan, "loo_spearman": np.nan}
    errors = np.asarray([row["robustness_error"] for row in results], dtype=np.float64)
    measured = np.asarray([row["measured_robustness"] for row in results], dtype=np.float64)
    predicted = np.asarray([row["predicted_robustness"] for row in results], dtype=np.float64)
    return results, {
        "loo_mae": float(np.mean(np.abs(errors))),
        "loo_rmse": float(np.sqrt(np.mean(errors ** 2))),
        "loo_spearman": spearman_correlation(measured, predicted),
    }


def write_csv(rows, path):
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def known_prediction_rows(design_summary, predictions):
    rows = []
    for design_index in sorted(design_summary):
        measured = design_summary[design_index]
        predicted = predictions[int(design_index)]
        row = {
            "design_index": int(design_index),
            "A_width_ratio": measured["A_width_ratio"],
            "A_attack_angle_deg": measured["A_attack_angle_deg"],
            "B_width_ratio": measured["B_width_ratio"],
            "B_attack_angle_deg": measured["B_attack_angle_deg"],
            "measured_robustness": measured["robustness"],
            "predicted_robustness": predicted["robustness"],
            "robustness_error": predicted["robustness"] - measured["robustness"],
            "predicted_uncertainty": predicted["uncertainty"],
        }
        for terrain in TERRAINS:
            row[f"{terrain}_measured_mean"] = measured["terrain_means"][terrain]
            row[f"{terrain}_measured_std"] = measured["terrain_stds"][terrain]
            row[f"{terrain}_measured_count"] = measured["terrain_counts"][terrain]
            row[f"{terrain}_predicted_mean"] = predicted["terrain_means"][terrain]
            row[f"{terrain}_predicted_std"] = predicted["terrain_stds"][terrain]
        rows.append(row)
    return rows


def print_known_prediction_table(rows):
    print("")
    print("=== DESIGN-VALUE KNOWN DESIGN CHECK ===")
    print(
        "idx | A(width,angle) | B(width,angle) | measured_robust | "
        "predicted_robust | error | uncertainty | cardboard meas/pred | carpet meas/pred"
    )
    print("-" * 148)
    for row in rows:
        print(
            f"{row['design_index']:>3} | "
            f"({row['A_width_ratio']:.3f},{row['A_attack_angle_deg']:>5.1f}) | "
            f"({row['B_width_ratio']:.3f},{row['B_attack_angle_deg']:>5.1f}) | "
            f"{row['measured_robustness']:>15.5g} | "
            f"{row['predicted_robustness']:>16.5g} | "
            f"{row['robustness_error']:>7.3g} | "
            f"{row['predicted_uncertainty']:>11.4g} | "
            f"{row['cardboard_measured_mean']:>8.4g}/{row['cardboard_predicted_mean']:>8.4g} | "
            f"{row['carpet_measured_mean']:>8.4g}/{row['carpet_predicted_mean']:>8.4g}"
        )
    print("========================================")


def _predict_heterogeneous_slice(model, candidates, bounds, design_mode, robustness_lambda):
    means = []
    stds = []
    for terrain in TERRAINS:
        x = [encode_feature_row(dict(row, terrain=terrain), terrain, bounds, design_mode) for row in candidates]
        mean, std = predict(model, x)
        means.append(mean)
        stds.append(std)
    means = np.asarray(means, dtype=np.float64)
    stds = np.asarray(stds, dtype=np.float64)
    robustness = means.mean(axis=0) - float(robustness_lambda) * means.std(axis=0)
    uncertainty = np.sqrt(np.mean(stds ** 2, axis=0) + np.std(means, axis=0) ** 2)
    return robustness, uncertainty


def _mark_heterogeneous_projection(axis, design_summary, reference, mode):
    if mode == "A":
        x_key, y_key = "A_width_ratio", "A_attack_angle_deg"
        x_label, y_label = "A width ratio", "A attack angle (deg)"
    else:
        x_key, y_key = "B_width_ratio", "B_attack_angle_deg"
        x_label, y_label = "B width ratio", "B attack angle (deg)"

    for design_index, row in design_summary.items():
        axis.scatter(
            [row[x_key]],
            [row[y_key]],
            s=80,
            marker="o",
            facecolors="none",
            edgecolors="black",
            linewidths=1.2,
        )
        axis.text(row[x_key], row[y_key], f" {design_index}", fontsize=8)
    axis.scatter(
        [reference[x_key]],
        [reference[y_key]],
        marker="D",
        s=70,
        color="tab:blue",
        label="slice anchor",
    )
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.grid(True, alpha=0.25)


def save_prediction_plot(model, design_summary, bounds, design_mode, args, output_dir, timestamp):
    if not args.save_plot:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    if design_mode != "homogeneous":
        prediction_rows = known_prediction_rows(
            design_summary,
            evaluate_known_designs(
                model,
                design_summary,
                bounds,
                design_mode,
                args.robustness_lambda,
            )["predictions"],
        )
        output_path = os.path.join(output_dir, f"{timestamp}_design_value_known_predictions.png")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
        measured = np.asarray([row["measured_robustness"] for row in prediction_rows], dtype=np.float64)
        predicted = np.asarray([row["predicted_robustness"] for row in prediction_rows], dtype=np.float64)
        labels = [str(row["design_index"]) for row in prediction_rows]
        lower = float(min(measured.min(), predicted.min()))
        upper = float(max(measured.max(), predicted.max()))
        pad = max((upper - lower) * 0.08, 1.0)
        axes[0].scatter(measured, predicted, s=80, facecolors="none", edgecolors="black")
        axes[0].plot([lower - pad, upper + pad], [lower - pad, upper + pad], color="tab:red", lw=1.2)
        for label, x, y in zip(labels, measured, predicted):
            axes[0].text(x, y, f" {label}", fontsize=9)
        axes[0].set_xlabel("Measured robustness")
        axes[0].set_ylabel("Predicted robustness")
        axes[0].set_title("Known design fit")
        axes[0].grid(True, alpha=0.25)

        x_pos = np.arange(len(prediction_rows))
        axes[1].bar(x_pos - 0.18, measured, width=0.36, label="measured")
        axes[1].bar(x_pos + 0.18, predicted, width=0.36, label="predicted")
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(labels)
        axes[1].set_xlabel("Heterogeneous design index")
        axes[1].set_ylabel("Robustness")
        axes[1].set_title("Measured vs predicted")
        axes[1].legend()
        axes[1].grid(True, axis="y", alpha=0.25)
        fig.savefig(output_path, dpi=180)
        plt.close(fig)

        reference = max(design_summary.values(), key=lambda row: row["robustness"])
        a_width_values = np.linspace(bounds[0][0], bounds[0][1], int(args.plot_grid_size))
        a_angle_values = np.linspace(bounds[1][0], bounds[1][1], int(args.plot_grid_size))
        mesh_aw, mesh_aa = np.meshgrid(a_width_values, a_angle_values)
        a_candidates = [
            {
                "A_width_ratio": aw,
                "A_attack_angle_deg": aa,
                "B_width_ratio": reference["B_width_ratio"],
                "B_attack_angle_deg": reference["B_attack_angle_deg"],
            }
            for aw, aa in zip(mesh_aw.reshape(-1), mesh_aa.reshape(-1))
        ]
        a_robustness, a_uncertainty = _predict_heterogeneous_slice(
            model,
            a_candidates,
            bounds,
            design_mode,
            args.robustness_lambda,
        )
        a_robustness = a_robustness.reshape(mesh_aw.shape)
        a_uncertainty = a_uncertainty.reshape(mesh_aw.shape)

        b_width_values = np.linspace(bounds[2][0], bounds[2][1], int(args.plot_grid_size))
        b_angle_values = np.linspace(bounds[3][0], bounds[3][1], int(args.plot_grid_size))
        mesh_bw, mesh_ba = np.meshgrid(b_width_values, b_angle_values)
        b_candidates = [
            {
                "A_width_ratio": reference["A_width_ratio"],
                "A_attack_angle_deg": reference["A_attack_angle_deg"],
                "B_width_ratio": bw,
                "B_attack_angle_deg": ba,
            }
            for bw, ba in zip(mesh_bw.reshape(-1), mesh_ba.reshape(-1))
        ]
        b_robustness, b_uncertainty = _predict_heterogeneous_slice(
            model,
            b_candidates,
            bounds,
            design_mode,
            args.robustness_lambda,
        )
        b_robustness = b_robustness.reshape(mesh_bw.shape)
        b_uncertainty = b_uncertainty.reshape(mesh_bw.shape)

        landscape_path = os.path.join(output_dir, f"{timestamp}_design_value_landscape_heterogeneous.png")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
        axes = axes.reshape(-1)

        plot = axes[0].contourf(mesh_aw, mesh_aa, a_robustness, levels=24, cmap="viridis")
        fig.colorbar(plot, ax=axes[0], label="Predicted robustness")
        _mark_heterogeneous_projection(axes[0], design_summary, reference, "A")
        axes[0].set_title(
            f"Vary Scale A | B fixed=({reference['B_width_ratio']:.3f}, "
            f"{reference['B_attack_angle_deg']:.1f})"
        )

        plot = axes[1].contourf(mesh_aw, mesh_aa, a_uncertainty, levels=24, cmap="magma")
        fig.colorbar(plot, ax=axes[1], label="Predictive uncertainty")
        _mark_heterogeneous_projection(axes[1], design_summary, reference, "A")
        axes[1].set_title(
            f"Uncertainty varying A | B fixed=({reference['B_width_ratio']:.3f}, "
            f"{reference['B_attack_angle_deg']:.1f})"
        )

        plot = axes[2].contourf(mesh_bw, mesh_ba, b_robustness, levels=24, cmap="viridis")
        fig.colorbar(plot, ax=axes[2], label="Predicted robustness")
        _mark_heterogeneous_projection(axes[2], design_summary, reference, "B")
        axes[2].set_title(
            f"Vary Scale B | A fixed=({reference['A_width_ratio']:.3f}, "
            f"{reference['A_attack_angle_deg']:.1f})"
        )

        plot = axes[3].contourf(mesh_bw, mesh_ba, b_uncertainty, levels=24, cmap="magma")
        fig.colorbar(plot, ax=axes[3], label="Predictive uncertainty")
        _mark_heterogeneous_projection(axes[3], design_summary, reference, "B")
        axes[3].set_title(
            f"Uncertainty varying B | A fixed=({reference['A_width_ratio']:.3f}, "
            f"{reference['A_attack_angle_deg']:.1f})"
        )

        fig.suptitle(
            "Heterogeneous design-value model 4D slices | "
            f"anchor=best measured design {int(reference['design_index'])}",
            fontsize=11,
        )
        fig.savefig(landscape_path, dpi=180)
        plt.close(fig)
        return landscape_path

    width_values = np.linspace(bounds[0][0], bounds[0][1], int(args.plot_grid_size))
    angle_values = np.linspace(bounds[1][0], bounds[1][1], int(args.plot_grid_size))
    mesh_w, mesh_a = np.meshgrid(width_values, angle_values)
    candidates = [
        {"width_ratio": w, "attack_angle_deg": a, "terrain": TERRAINS[0]}
        for w, a in zip(mesh_w.reshape(-1), mesh_a.reshape(-1))
    ]

    means = []
    stds = []
    for terrain in TERRAINS:
        x = [encode_feature_row(dict(row, terrain=terrain), terrain, bounds, design_mode) for row in candidates]
        mean, std = predict(model, x)
        means.append(mean)
        stds.append(std)
    means = np.asarray(means)
    stds = np.asarray(stds)
    robustness = (means.mean(axis=0) - args.robustness_lambda * means.std(axis=0)).reshape(mesh_w.shape)
    uncertainty = np.sqrt(np.mean(stds ** 2, axis=0) + np.std(means, axis=0) ** 2).reshape(mesh_w.shape)

    output_path = os.path.join(output_dir, f"{timestamp}_design_value_landscape.png")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    r_plot = axes[0].contourf(mesh_w, mesh_a, robustness, levels=24, cmap="viridis")
    u_plot = axes[1].contourf(mesh_w, mesh_a, uncertainty, levels=24, cmap="magma")
    fig.colorbar(r_plot, ax=axes[0], label="Predicted robustness")
    fig.colorbar(u_plot, ax=axes[1], label="Predictive uncertainty")
    for axis in axes:
        for design_index, row in design_summary.items():
            axis.scatter([row["width_ratio"]], [row["attack_angle_deg"]], s=80, marker="o", facecolors="none", edgecolors="black")
            axis.text(row["width_ratio"], row["attack_angle_deg"], f" {design_index}", fontsize=8)
        axis.set_xlabel("Width ratio")
        axis.set_ylabel("Attack angle (deg)")
        axis.grid(True, alpha=0.25)
    axes[0].set_title("Design-value robustness")
    axes[1].set_title("Design-value uncertainty")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-mode", choices=["homogeneous", "heterogeneous"], default="homogeneous")
    parser.add_argument("--model-type", choices=["gpr"], default="gpr")
    parser.add_argument("--training-mode", choices=["balanced_episodes"], default="balanced_episodes")
    parser.add_argument("--max-episodes-per-design-terrain", type=int, default=10)
    parser.add_argument("--episode-window-start", type=int, default=20)
    parser.add_argument("--episode-window-end", type=int, default=30)
    parser.add_argument("--gpr-aggregate-rollouts", action="store_true", default=True)
    parser.add_argument("--gpr-alpha-mode", choices=["sem"], default="sem")
    parser.add_argument("--gpr-min-alpha", type=float, default=1e-6)
    parser.add_argument("--gpr-restarts", type=int, default=20)
    parser.add_argument("--robustness-lambda", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--plot-grid-size", type=int, default=80)
    parser.add_argument("--save-csv", action="store_true")
    parser.add_argument("--save-plot", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(MODEL_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")

    rollout_rows = extract_rollout_rows(args.design_mode)
    window_rows = filter_episode_window_rows(rollout_rows, args.episode_window_start, args.episode_window_end)
    balanced_rows = filter_balanced_rows(window_rows, args.max_episodes_per_design_terrain)
    training_rows = aggregate_rows(balanced_rows)
    bounds = feature_bounds(args.design_mode)

    if not training_rows:
        raise RuntimeError(
            f"No training rows found for episode window "
            f"[{args.episode_window_start}, {args.episode_window_end})."
        )

    model = train_gpr(training_rows, bounds, args.design_mode, args)
    terrain_summary, design_summary = summarize_designs(rollout_rows, args.robustness_lambda)
    selected_terrain_summary, selected_design_summary = summarize_designs(balanced_rows, args.robustness_lambda)
    known_eval = evaluate_known_designs(model, selected_design_summary, bounds, args.design_mode, args.robustness_lambda)
    known_prediction_table = known_prediction_rows(selected_design_summary, known_eval["predictions"])
    loo_rows, loo_summary = leave_one_design_out(training_rows, balanced_rows, bounds, args.design_mode, args)

    training_csv_path = os.path.join(MODEL_DIR, f"{timestamp}_design_value_training_rows.csv")
    rollout_csv_path = os.path.join(MODEL_DIR, f"{timestamp}_design_value_rollout_rows.csv")
    predictions_csv_path = os.path.join(MODEL_DIR, f"{timestamp}_design_value_known_predictions.csv")
    if args.save_csv:
        write_csv(training_rows, training_csv_path)
        write_csv(rollout_rows, rollout_csv_path)
        write_csv(known_prediction_table, predictions_csv_path)

    plot_path = save_prediction_plot(model, selected_design_summary, bounds, args.design_mode, args, MODEL_DIR, timestamp)

    diagnostics = {
        "created_at": timestamp,
        "model_type": "GaussianProcessRegressor",
        "model_type_key": "gpr",
        "training_mode": "balanced_episodes",
        "effective_training_mode": "balanced_episodes_aggregated_conditions",
        "design_mode": args.design_mode,
        "feature_schema": feature_schema(args.design_mode, bounds),
        "feature_bounds": bounds,
        "gpr_aggregate_rollouts": True,
        "gpr_alpha_mode": "sem",
        "gpr_min_alpha": args.gpr_min_alpha,
        "gpr_restarts": args.gpr_restarts,
        "gpr_target_mean": float(model._snake_target_mean),
        "gpr_target_std": float(model._snake_target_std),
        "gpr_kernel": str(model.kernel),
        "gpr_fitted_kernel": str(model.kernel_),
        "max_episodes_per_design_terrain": args.max_episodes_per_design_terrain,
        "episode_window_start": args.episode_window_start,
        "episode_window_end": args.episode_window_end,
        "terrain_summary": terrain_summary,
        "selected_terrain_summary": selected_terrain_summary,
        "design_summary": design_summary,
        "selected_design_summary": selected_design_summary,
        "known_design_eval": known_eval,
        "known_prediction_table": known_prediction_table,
        "leave_one_design_out": {"rows": loo_rows, "aggregate": loo_summary},
        "row_count": len(training_rows),
        "selected_row_count": len(balanced_rows),
        "episode_window_row_count": len(window_rows),
        "rollout_row_count": len(rollout_rows),
        "aggregate_row_count": len(training_rows),
        "training_csv_path": training_csv_path if args.save_csv else None,
        "rollout_csv_path": rollout_csv_path if args.save_csv else None,
        "predictions_csv_path": predictions_csv_path if args.save_csv else None,
        "plot_path": plot_path,
    }

    import joblib
    bundle = {
        "model": model,
        "model_type": "gpr",
        "design_mode": args.design_mode,
        "terrains": TERRAINS,
        "feature_bounds": bounds,
        "feature_schema": feature_schema(args.design_mode, bounds),
        "diagnostics": diagnostics,
    }
    model_path = os.path.join(MODEL_DIR, f"{timestamp}_design_value_model.joblib")
    diagnostics_path = os.path.join(MODEL_DIR, f"{timestamp}_design_value_diagnostics.json")
    joblib.dump(bundle, model_path)
    with open(diagnostics_path, "w") as f:
        json.dump(json_safe(diagnostics), f, indent=2)

    print(f"Saved model: {model_path}")
    print(f"Saved diagnostics: {diagnostics_path}")
    if args.save_csv:
        print(f"Saved training CSV: {training_csv_path}")
        print(f"Saved rollout CSV: {rollout_csv_path}")
        print(f"Saved prediction CSV: {predictions_csv_path}")
    if plot_path:
        print(f"Saved plot: {plot_path}")
    print_known_prediction_table(known_prediction_table)
    print(f"Known-design Spearman: {known_eval['spearman']:.4g}, RMSE: {known_eval['rmse']:.4g}")
    print(f"LOO MAE: {loo_summary['loo_mae']:.4g}, RMSE: {loo_summary['loo_rmse']:.4g}, Spearman: {loo_summary['loo_spearman']:.4g}")


if __name__ == "__main__":
    main()
