import argparse
import csv
import gc
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

import rlkit.torch.networks as rlkit_networks
from snakeenv_thread_coadapt import MotorFaultError, SnakeEnv


def identity(x):
    return x


# Older saved rlkit modules reference symbols that may be missing from newer
# RLKit forks. Define them before loading trusted local checkpoints.
rlkit_networks.identity = identity
if (
    not hasattr(rlkit_networks, "FlattenMlp")
    and hasattr(rlkit_networks, "ConcatMlp")
):
    rlkit_networks.FlattenMlp = rlkit_networks.ConcatMlp


TERRAINS_DEFAULT = "cardboard,carpet"
TAG_FILTER_DEFAULT = "scale_ab_carpet_cardboard"
DETERMINISM_TOLERANCE = 1e-7
DEFAULT_INCLUDED_PREFIXES = {
    # Design 0 was resumed/extended across runs and its longest usable
    # carpet/cardboard checkpoint is intentionally part of the default set.
    "2026_05_18_DesignCycle0_ep76",
}


@dataclass
class CheckpointSpec:
    mode: str
    design_index: int
    tag: str
    prefix: str
    metadata_path: Path
    metadata: dict
    complete: bool
    expected_episodes: int
    episode_counter: int
    design: list


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate scale-design checkpoints with fully deterministic "
            "policy actions from the policy distribution MLE."
        )
    )
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--terrains", default=TERRAINS_DEFAULT)
    parser.add_argument("--only-mode", choices=("homogeneous", "heterogeneous", "all"), default="all")
    parser.add_argument("--results-dir", default=os.path.join("CoadaptationCode", "results_bazyli"))
    parser.add_argument(
        "--output-dir",
        default=os.path.join("CoadaptationCode", "results_bazyli", "deterministic_evaluations"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-incomplete", action="store_true")
    parser.add_argument("--checkpoint-prefix", default="")
    parser.add_argument(
        "--results-tag",
        default="",
        help="Optional exact results tag to disambiguate --checkpoint-prefix.",
    )
    parser.add_argument(
        "--tag-contains",
        default=TAG_FILTER_DEFAULT,
        help="Only auto-discover checkpoints whose results_tag contains this text.",
    )
    parser.add_argument("--max-steps", type=int, default=175)
    parser.add_argument("--seed", type=int, default=12345)
    return parser.parse_args()


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_torch_load(path, **kwargs):
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError as exc:
        if "weights_only" not in str(exc):
            raise
        return torch.load(path, **kwargs)


def parse_terrains(raw_value):
    terrains = [item.strip() for item in str(raw_value).split(",") if item.strip()]
    if not terrains:
        raise ValueError("At least one terrain is required.")
    invalid = [terrain for terrain in terrains if terrain not in SnakeEnv.terrains]
    if invalid:
        raise ValueError(f"Unknown terrain(s): {invalid}. Use one of {SnakeEnv.terrains}.")
    return terrains


def checkpoint_prefix_from_metadata_path(path):
    name = path.name
    marker = "_metadata"
    if marker not in name:
        raise ValueError(f"Metadata filename does not contain '{marker}': {name}")
    return name.split(marker, 1)[0]


def expected_episodes(metadata):
    schedule = metadata.get("training_episode_schedule") or []
    if schedule:
        return int(len(schedule))
    terrains = metadata.get("active_terrains") or metadata.get("terrain_sequence") or []
    block_size = metadata.get(
        "episodes_per_terrain",
        metadata.get("training_terrain_block_size", 0),
    )
    try:
        return int(len(terrains) * int(block_size))
    except (TypeError, ValueError):
        return 0


def design_index_matches_tag(tag, design_index):
    matches = re.findall(r"(?:^|_)design(\d+)(?:_|$)", str(tag))
    if not matches:
        return True
    return any(int(match) == int(design_index) for match in matches)


def resolve_design(metadata):
    for key in ("optimized_params", "fixed_scale_design"):
        value = metadata.get(key)
        if value is not None:
            return SnakeEnv._coerce_design_vector(value)

    mode = str(metadata.get("scale_design_mode", "homogeneous")).strip().lower()
    design_index = int(metadata.get("design_counter", 0))
    initial_designs = SnakeEnv.get_init_design_parameters(mode)
    if 0 <= design_index < len(initial_designs):
        return SnakeEnv._coerce_design_vector(initial_designs[design_index])

    return SnakeEnv._coerce_design_vector(metadata.get("current_design", SnakeEnv.get_default_design()))


def make_spec(path, metadata):
    mode = str(metadata.get("scale_design_mode", "")).strip().lower()
    tag = str(metadata.get("results_tag", "")).strip()
    design_index = int(metadata.get("design_counter", 0))
    prefix = checkpoint_prefix_from_metadata_path(path)
    episode_counter = int(metadata.get("episode_counter", 0))
    expected = expected_episodes(metadata)
    complete = expected > 0 and episode_counter >= expected
    return CheckpointSpec(
        mode=mode,
        design_index=design_index,
        tag=tag,
        prefix=prefix,
        metadata_path=path,
        metadata=metadata,
        complete=complete,
        expected_episodes=expected,
        episode_counter=episode_counter,
        design=resolve_design(metadata),
    )


def discover_specs(args, results_dir):
    metadata_paths = sorted(results_dir.glob("*metadata*.json"))
    specs = []
    for path in metadata_paths:
        if path.name.startswith("best_probe"):
            continue
        try:
            metadata = read_json(path)
        except Exception:
            continue
        mode = str(metadata.get("scale_design_mode", "")).strip().lower()
        tag = str(metadata.get("results_tag", "")).strip()
        if mode not in ("homogeneous", "heterogeneous"):
            continue
        if args.only_mode != "all" and mode != args.only_mode:
            continue
        if args.tag_contains and args.tag_contains not in tag:
            continue
        try:
            spec = make_spec(path, metadata)
        except Exception as exc:
            print(f"Skipping metadata {path.name}: {exc}")
            continue
        if not design_index_matches_tag(spec.tag, spec.design_index):
            continue
        if (
            not spec.complete
            and not args.include_incomplete
            and spec.prefix not in DEFAULT_INCLUDED_PREFIXES
        ):
            continue
        specs.append(spec)

    best_by_group = {}
    for spec in specs:
        key = (spec.mode, spec.tag, spec.design_index)
        current = best_by_group.get(key)
        if current is None or (spec.complete, spec.episode_counter, spec.prefix) > (
            current.complete,
            current.episode_counter,
            current.prefix,
        ):
            best_by_group[key] = spec

    return sorted(
        best_by_group.values(),
        key=lambda spec: (
            0 if spec.mode == "homogeneous" else 1,
            spec.design_index,
            spec.tag,
        ),
    )


def discover_one_spec(args, results_dir):
    prefix = args.checkpoint_prefix.strip()
    candidates = []
    for path in sorted(results_dir.glob(f"{prefix}_metadata*.json")):
        try:
            metadata = read_json(path)
        except Exception:
            continue
        if args.results_tag and metadata.get("results_tag") != args.results_tag:
            continue
        if args.only_mode != "all" and metadata.get("scale_design_mode") != args.only_mode:
            continue
        if args.tag_contains and args.tag_contains not in str(metadata.get("results_tag", "")):
            continue
        candidates.append(make_spec(path, metadata))

    if not candidates:
        raise FileNotFoundError(
            f"No metadata found for checkpoint prefix '{prefix}' in {results_dir}."
        )
    if len(candidates) > 1:
        options = "\n".join(
            f"  {spec.metadata_path.name} tag={spec.tag}" for spec in candidates
        )
        raise RuntimeError(
            "Multiple metadata files matched --checkpoint-prefix. "
            "Pass --results-tag to disambiguate:\n" + options
        )
    spec = candidates[0]
    if not spec.complete and not args.include_incomplete:
        raise RuntimeError(
            f"Checkpoint {spec.prefix} is incomplete "
            f"({spec.episode_counter}/{spec.expected_episodes}). "
            "Pass --include-incomplete to evaluate it anyway."
        )
    return [spec]


def policy_path_for_spec(spec, terrain, results_dir):
    terrain_specific = results_dir / f"ind_policy_{spec.prefix}_{terrain}_{spec.tag}.pt"
    if terrain_specific.exists():
        return terrain_specific

    if str(spec.metadata.get("terrain_model_mode", "separate")).strip().lower() == "separate":
        raise FileNotFoundError(
            "Missing terrain-specific individual policy for separate-terrain checkpoint: "
            f"{terrain_specific}"
        )

    generic = results_dir / f"ind_policy_{spec.prefix}_{spec.tag}.pt"
    if generic.exists():
        return generic
    raise FileNotFoundError(
        f"Missing individual policy for checkpoint {spec.prefix}, terrain {terrain}."
    )


def deterministic_policy_action(policy, state):
    policy.eval()
    obs = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        dist = policy(obs)
        if not hasattr(dist, "mle_estimate"):
            raise RuntimeError(
                "Policy distribution has no mle_estimate(); refusing stochastic evaluation."
            )
        action = dist.mle_estimate()
    action_np = action.squeeze(0).detach().cpu().numpy()
    return np.clip(np.asarray(action_np, dtype=np.float32), -1.0, 1.0)


def deterministic_action_check(policy, obs_dim):
    state = np.zeros(int(obs_dim), dtype=np.float32)
    return deterministic_action_check_for_state(policy, state)


def deterministic_action_check_for_state(policy, state):
    first = deterministic_policy_action(policy, state)
    second = deterministic_policy_action(policy, state)
    max_diff = float(np.max(np.abs(first - second))) if first.size else 0.0
    if max_diff > DETERMINISM_TOLERANCE:
        raise RuntimeError(
            f"Deterministic action self-check failed: max_diff={max_diff}."
        )
    return max_diff


def require_current_terrain(expected_terrain, context):
    actual_terrain = SnakeEnv.get_current_terrain()
    if actual_terrain != expected_terrain:
        raise RuntimeError(
            f"Terrain mismatch during {context}: expected '{expected_terrain}', "
            f"but SnakeEnv.current_terrain is '{actual_terrain}'. Aborting so the "
            "robot is not evaluated on the wrong surface."
        )
    return actual_terrain


def print_current_terrain(expected_terrain, context, spec=None, rollout_idx=None, eval_episodes=None):
    active_terrain = require_current_terrain(expected_terrain, context)
    terrain_id = SnakeEnv.get_terrain_id(active_terrain)
    print("", flush=True)
    print("=== CURRENT TERRAIN ===", flush=True)
    print(
        f"CURRENT TERRAIN: {active_terrain.upper()} "
        f"(terrain_id={terrain_id})",
        flush=True,
    )
    if spec is not None:
        print(
            f"Design: {spec.mode} design {spec.design_index} | checkpoint={spec.prefix}",
            flush=True,
        )
    if rollout_idx is not None and eval_episodes is not None:
        print(f"Rollout: {rollout_idx + 1}/{eval_episodes}", flush=True)
    print("Use this physical surface now.", flush=True)
    print("=======================", flush=True)
    return active_terrain, terrain_id


def load_policy_checked(policy_path, obs_dim):
    policy = safe_torch_load(policy_path, map_location="cpu")
    max_diff = deterministic_action_check(policy, obs_dim)
    return policy, max_diff


def print_install_prompt(spec, terrains, eval_episodes):
    summary = SnakeEnv.design_summary(spec.design)
    first_terrain = terrains[0] if terrains else "UNKNOWN"
    first_terrain_id = (
        SnakeEnv.get_terrain_id(first_terrain)
        if first_terrain in SnakeEnv.terrain_name_to_id
        else "UNKNOWN"
    )
    print("")
    print("=== INSTALL SCALE CONFIGURATION FOR DETERMINISTIC EVALUATION ===")
    print(f"Mode: {spec.mode}")
    print(f"Design index: {spec.design_index}")
    print(f"Checkpoint: {spec.prefix}")
    print(f"Tag: {spec.tag}")
    print(
        f"Scale A: width_ratio={summary['A_Width_Ratio']:.3f}, "
        f"actual_width={summary['A_Actual_Width']:.3f}, "
        f"attack_angle_deg={summary['A_Attack_Angle_Deg']:.2f}"
    )
    print(
        f"Scale B: width_ratio={summary['B_Width_Ratio']:.3f}, "
        f"actual_width={summary['B_Actual_Width']:.3f}, "
        f"attack_angle_deg={summary['B_Attack_Angle_Deg']:.2f}"
    )
    print("Modules 1,3,5,7: Scale A")
    print("Modules 2,4,6,8: Scale B")
    print(f"Terrains: {', '.join(terrains)}")
    print(f"CURRENT TERRAIN / FIRST TERRAIN TO USE: {str(first_terrain).upper()} (terrain_id={first_terrain_id})")
    print(f"Evaluation rollouts per terrain: {eval_episodes}")
    print("================================================================")
    input("Install this scale design, then press Enter to continue.")


def motor_thread(stop_event):
    while not stop_event.is_set():
        SnakeEnv.motorPos()


def opti_thread(stop_event):
    while not stop_event.is_set():
        SnakeEnv.optiPos()


def recover_motor_fault(phase, exc, disable_after_recovery=True):
    print(f"Motor fault during {phase}: {exc}")
    recovered = False
    try:
        recovered = SnakeEnv.recoverMotorFault(
            context=f"{phase}: {exc}",
            force_reboot=True,
        )
    except Exception as recovery_exc:
        print(f"Motor recovery handler raised an exception: {recovery_exc}")

    if not disable_after_recovery:
        return recovered

    try:
        disable_motor_torque(f"{phase} recovery cleanup")
    except Exception as disable_exc:
        print(f"Torque cleanup after recovery failed: {disable_exc}")
        return False
    return recovered


def disable_motor_torque(phase):
    if not SnakeEnv.motorBusAvailable():
        print(f"Cannot disable motor torque during {phase}: DYNAMIXEL USB device is not present.")
        return False
    try:
        torque_disabled = SnakeEnv.disableMotorTorque()
        if torque_disabled:
            return True
        print(f"Failed to disable motor torque during {phase}; forcing DYNAMIXEL reboot.")
    except Exception as disable_exc:
        print(f"Failed to disable motor torque during {phase}: {disable_exc}. Forcing reboot.")

    recovered = SnakeEnv.recoverMotorFault(
        context=f"{phase}: disable torque failed",
        force_reboot=True,
    )
    if not recovered:
        return False
    return bool(SnakeEnv.disableMotorTorque())


def reset_with_recovery(env, seed, phase, reset_prompt, max_attempts=2):
    last_exc = None
    for attempt_idx in range(max_attempts):
        try:
            return env.reset(
                seed=seed,
                options={
                    "reset_prompt": reset_prompt,
                    "interactive_reset": True,
                },
            )
        except MotorFaultError as exc:
            last_exc = exc
            recovered = recover_motor_fault(
                f"{phase} reset attempt {attempt_idx + 1}/{max_attempts}",
                exc,
            )
            if not recovered:
                break
            time.sleep(1.0)
    print(f"Skipping {phase} after failed reset attempts: {last_exc}")
    return None, None


def step_with_recovery(env, action, phase, step_number, max_retries=3):
    max_attempts = max_retries + 1
    for attempt_idx in range(max_attempts):
        try:
            return env.step(action)
        except MotorFaultError as exc:
            if attempt_idx >= max_attempts - 1:
                print(f"Motor fault persisted at step {step_number}.")
                recover_motor_fault(f"{phase} step {step_number}", exc)
                return None
            recovered = recover_motor_fault(
                f"{phase} step {step_number} attempt {attempt_idx + 1}/{max_attempts}",
                exc,
                disable_after_recovery=False,
            )
            if not recovered:
                disable_motor_torque(f"{phase} failed step recovery cleanup")
                return None
            try:
                if not SnakeEnv.enableMotorTorque():
                    disable_motor_torque(f"{phase} torque-enable failure cleanup")
                    return None
            except Exception as enable_exc:
                print(f"Failed to enable motor torque after recovery: {enable_exc}")
                disable_motor_torque(f"{phase} torque-enable exception cleanup")
                return None
    return None


def stable_seed(base_seed, *components):
    text = "|".join([str(base_seed), *[str(component) for component in components]])
    value = 2166136261
    for char in text:
        value ^= ord(char)
        value = (value * 16777619) % (2 ** 32)
    return int(value)


def design_fields(design):
    summary = SnakeEnv.design_summary(design)
    return {
        "A_width_ratio": summary["A_Width_Ratio"],
        "A_attack_angle_deg": summary["A_Attack_Angle_Deg"],
        "A_actual_width": summary["A_Actual_Width"],
        "B_width_ratio": summary["B_Width_Ratio"],
        "B_attack_angle_deg": summary["B_Attack_Angle_Deg"],
        "B_actual_width": summary["B_Actual_Width"],
        "full_design": "|".join(f"{float(value):.6g}" for value in design),
    }


def base_row(spec, terrain, rollout_idx, policy_path, determinism_max_diff):
    row = {
        "checkpoint_prefix": spec.prefix,
        "metadata_file": spec.metadata_path.name,
        "results_tag": spec.tag,
        "design_mode": spec.mode,
        "design_index": spec.design_index,
        "terrain": terrain,
        "rollout_index": rollout_idx,
        "policy_file": policy_path.name,
        "episode_counter": spec.episode_counter,
        "expected_episodes": spec.expected_episodes,
        "checkpoint_complete": spec.complete,
        "resume_checkpoint_prefix": spec.metadata.get("resume_checkpoint_prefix"),
        "run_start_episode": spec.metadata.get("run_start_episode"),
        "resume_as_new_design_run": spec.metadata.get("resume_as_new_design_run"),
        "start_design_counter": spec.metadata.get("start_design_counter"),
        "checkpoint_results_tags": "|".join(spec.metadata.get("checkpoint_results_tags") or []),
        "deterministic_action_max_diff": determinism_max_diff,
    }
    row.update(design_fields(spec.design))
    return row


def run_rollout(env, policy, spec, terrain, rollout_idx, policy_path, determinism_max_diff, args):
    phase = f"{spec.mode} design {spec.design_index} {terrain} rollout {rollout_idx + 1}"
    active_terrain, terrain_id = print_current_terrain(
        terrain,
        phase,
        spec=spec,
        rollout_idx=rollout_idx,
        eval_episodes=args.eval_episodes,
    )
    print("Policy action mode: FULLY DETERMINISTIC distribution MLE.", flush=True)
    reset_prompt = (
        "\n"
        "=== DETERMINISTIC EVALUATION RESET ===\n"
        f"Mode: {spec.mode}\n"
        f"Design index: {spec.design_index}\n"
        f"Checkpoint: {spec.prefix}\n"
        f"TERRAIN TO USE NOW: {active_terrain.upper()}  (terrain_id={terrain_id})\n"
        f"Rollout: {rollout_idx + 1}/{args.eval_episodes}\n"
        f"Design: {SnakeEnv.format_design_for_terminal(spec.design)}\n"
        "Place/reset the robot on THIS terrain, then press Enter."
    )
    seed = stable_seed(args.seed, spec.mode, spec.design_index, spec.prefix, terrain, rollout_idx)
    state, _info = reset_with_recovery(env, seed, phase, reset_prompt)
    row = base_row(spec, terrain, rollout_idx, policy_path, determinism_max_diff)
    row.update({
        "seed": seed,
        "return": np.nan,
        "length": 0,
        "progress_cm": np.nan,
        "terminated": False,
        "truncated": False,
        "reset_failed": state is None,
        "motor_fault": False,
        "status": "reset_failed" if state is None else "started",
    })
    if state is None:
        return row

    state_determinism_max_diff = deterministic_action_check_for_state(policy, state)
    print(
        f"Deterministic action check on reset state for {phase}: "
        f"max_diff={state_determinism_max_diff:.9g}"
    )
    row["deterministic_action_reset_state_max_diff"] = state_determinism_max_diff

    cumulative_reward = 0.0
    progress_cm = 0.0
    steps = 0
    terminated = False
    truncated = False
    motor_fault = False

    try:
        while not (terminated or truncated) and steps < args.max_steps:
            action = deterministic_policy_action(policy, state)
            step_result = step_with_recovery(env, action, phase, steps + 1)
            if step_result is None:
                motor_fault = True
                break
            next_state, reward, terminated, truncated, info = step_result
            cumulative_reward += float(reward)
            progress_cm += float(info.get("distance_progress_cm", 0.0))
            state = next_state
            steps += 1
    finally:
        try:
            disable_motor_torque(f"end of {phase}")
        except Exception as exc:
            print(f"Torque disable at end of rollout raised: {exc}")

    row.update({
        "return": cumulative_reward,
        "length": steps,
        "progress_cm": progress_cm,
        "terminated": bool(terminated),
        "truncated": bool(truncated or steps >= args.max_steps),
        "reset_failed": False,
        "motor_fault": bool(motor_fault),
        "status": "motor_fault" if motor_fault else "complete",
    })
    print(
        f"Completed {phase}: return={cumulative_reward:.4f}, "
        f"length={steps}, progress_cm={progress_cm:.4f}, status={row['status']}"
    )
    return row


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def finite_stats(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
        }
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def build_summary(rows):
    grouped = {}
    for row in rows:
        key = (
            row["design_mode"],
            int(row["design_index"]),
            row["checkpoint_prefix"],
            row["results_tag"],
        )
        grouped.setdefault(key, []).append(row)

    summaries = []
    for key, group_rows in sorted(grouped.items()):
        mode, design_index, prefix, tag = key
        terrain_means = []
        summary = {
            "design_mode": mode,
            "design_index": design_index,
            "checkpoint_prefix": prefix,
            "results_tag": tag,
        }
        summary.update({
            field: group_rows[0].get(field)
            for field in (
                "A_width_ratio",
                "A_attack_angle_deg",
                "A_actual_width",
                "B_width_ratio",
                "B_attack_angle_deg",
                "B_actual_width",
                "full_design",
            )
        })

        for terrain in sorted({row["terrain"] for row in group_rows}):
            terrain_rows = [row for row in group_rows if row["terrain"] == terrain]
            stats = finite_stats([row["return"] for row in terrain_rows])
            length_stats = finite_stats([row["length"] for row in terrain_rows])
            progress_stats = finite_stats([row["progress_cm"] for row in terrain_rows])
            summary[f"{terrain}_rollouts"] = stats["count"]
            summary[f"{terrain}_return_mean"] = stats["mean"]
            summary[f"{terrain}_return_std"] = stats["std"]
            summary[f"{terrain}_return_min"] = stats["min"]
            summary[f"{terrain}_return_max"] = stats["max"]
            summary[f"{terrain}_length_mean"] = length_stats["mean"]
            summary[f"{terrain}_progress_cm_mean"] = progress_stats["mean"]
            if np.isfinite(stats["mean"]):
                terrain_means.append(stats["mean"])

        terrain_means = np.asarray(terrain_means, dtype=float)
        if terrain_means.size:
            summary["mean_across_terrains"] = float(np.mean(terrain_means))
            summary["std_across_terrains"] = float(np.std(terrain_means))
            summary["worst_terrain_return"] = float(np.min(terrain_means))
            summary["robustness_score"] = float(
                np.mean(terrain_means) - 0.5 * np.std(terrain_means)
            )
        else:
            summary["mean_across_terrains"] = np.nan
            summary["std_across_terrains"] = np.nan
            summary["worst_terrain_return"] = np.nan
            summary["robustness_score"] = np.nan
        summaries.append(summary)
    return summaries


def print_selection(specs, terrains, results_dir):
    print("")
    print("=== DETERMINISTIC EVALUATION CHECKPOINT SELECTION ===")
    for spec in specs:
        print(
            f"{spec.mode:13s} design={spec.design_index} "
            f"checkpoint={spec.prefix} ep={spec.episode_counter}/{spec.expected_episodes} "
            f"tag={spec.tag}"
        )
        print(f"  metadata: {spec.metadata_path.name}")
        print(f"  complete: {spec.complete}")
        print(f"  resume_checkpoint_prefix: {spec.metadata.get('resume_checkpoint_prefix')}")
        print(f"  checkpoint_results_tags: {spec.metadata.get('checkpoint_results_tags')}")
        print(f"  design: {SnakeEnv.format_design_for_terminal(spec.design)}")
        for terrain in terrains:
            try:
                path = policy_path_for_spec(spec, terrain, results_dir)
                print(f"  {terrain} policy: {path.name}")
            except FileNotFoundError as exc:
                print(f"  {terrain} policy: MISSING ({exc})")
                raise
    print("=====================================================")
    print("")


def run_dry_policy_checks(specs, terrains, results_dir):
    print("=== DETERMINISTIC POLICY SELF-CHECK ===")
    for spec in specs:
        obs_dim = int(spec.metadata.get("observation_dim", SnakeEnv.base_feature_dim + len(SnakeEnv.config_numpy)))
        for terrain in terrains:
            SnakeEnv.set_current_terrain(terrain)
            active_terrain, _terrain_id = print_current_terrain(
                terrain,
                f"dry-run policy check for {spec.mode} design {spec.design_index}",
                spec=spec,
            )
            path = policy_path_for_spec(spec, terrain, results_dir)
            _policy, max_diff = load_policy_checked(path, obs_dim)
            print(
                f"{spec.mode} design {spec.design_index} {active_terrain}: "
                f"deterministic_action_check max_diff={max_diff:.9g}"
            )
    print("=======================================")
    print("")


def evaluate(specs, terrains, args, results_dir, output_dir):
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    rollout_csv = output_dir / f"{timestamp}_deterministic_eval_rollouts.csv"
    summary_csv = output_dir / f"{timestamp}_deterministic_eval_summary.csv"
    summary_json = output_dir / f"{timestamp}_deterministic_eval_summary.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    stop_event = threading.Event()
    SnakeEnv.passLocksToEnv(threading.Lock(), threading.Lock())
    env = SnakeEnv()
    motor = threading.Thread(target=motor_thread, args=(stop_event,), daemon=True)
    opti = threading.Thread(target=opti_thread, args=(stop_event,), daemon=True)

    rows = []
    try:
        motor.start()
        opti.start()
        for spec in specs:
            SnakeEnv.set_new_design(spec.design)
            print_install_prompt(spec, terrains, args.eval_episodes)
            for terrain in terrains:
                SnakeEnv.set_current_terrain(terrain)
                active_terrain, _terrain_id = print_current_terrain(
                    terrain,
                    f"{spec.mode} design {spec.design_index} terrain setup",
                    spec=spec,
                )
                print("=== NEXT TERRAIN BLOCK ===", flush=True)
                print("This block will use the terrain-specific individual policy file.")
                print("==========================", flush=True)
                policy_path = policy_path_for_spec(spec, terrain, results_dir)
                obs_dim = int(spec.metadata.get("observation_dim", env.observation_space.shape[0]))
                policy, max_diff = load_policy_checked(policy_path, obs_dim)
                print(
                    f"Loaded deterministic policy for {spec.mode} design {spec.design_index} "
                    f"on {active_terrain}: {policy_path.name}; max_diff={max_diff:.9g}"
                )
                for rollout_idx in range(args.eval_episodes):
                    row = run_rollout(
                        env,
                        policy,
                        spec,
                        terrain,
                        rollout_idx,
                        policy_path,
                        max_diff,
                        args,
                    )
                    rows.append(row)
                    write_csv(rollout_csv, rows)
                    summaries = build_summary(rows)
                    write_csv(summary_csv, summaries)
                    with open(summary_json, "w", encoding="utf-8") as f:
                        json.dump(json_safe(summaries), f, indent=2, allow_nan=False)
    finally:
        stop_event.set()
        try:
            disable_motor_torque("deterministic evaluation shutdown")
        except Exception as exc:
            print(f"Shutdown torque disable raised: {exc}")
        motor.join(timeout=2.0)
        opti.join(timeout=2.0)

    print("")
    print("=== DETERMINISTIC EVALUATION COMPLETE ===")
    print(f"Saved rollout CSV: {rollout_csv}")
    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved summary JSON: {summary_json}")
    print("=========================================")


def main():
    gc.collect()
    args = parse_args()
    args.eval_episodes = max(1, int(args.eval_episodes))
    args.max_steps = max(1, int(args.max_steps))
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    terrains = parse_terrains(args.terrains)

    if args.checkpoint_prefix.strip():
        specs = discover_one_spec(args, results_dir)
    else:
        specs = discover_specs(args, results_dir)
    if not specs:
        raise RuntimeError("No checkpoints selected for deterministic evaluation.")

    print_selection(specs, terrains, results_dir)
    run_dry_policy_checks(specs, terrains, results_dir)

    if args.dry_run:
        print("Dry run complete. No robot threads were started.")
        return

    evaluate(specs, terrains, args, results_dir, output_dir)


if __name__ == "__main__":
    main()