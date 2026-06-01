import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


SCRIPT_DIR = Path(__file__).resolve().parent


def safe_member_name(index: int, seed_run_dir: Path) -> str:
    name = seed_run_dir.name.strip() or f"seed_{index}"
    return f"member_{index:02d}_{name}"


def run_member(args: argparse.Namespace, seed_run_dir: Path, member_dir: Path, index: int) -> Dict[str, object]:
    checkpoint_path = member_dir / "checkpoints" / "combined_population_critic.pt"
    replay_path = member_dir / "replay" / "population_replay_latest.npz"
    summary_path = member_dir / "combined_population_critic_summary.json"
    if args.skip_existing and checkpoint_path.exists() and replay_path.exists():
        print(f"[{index}] keeping existing ensemble member: {member_dir}")
    else:
        command: List[str] = [
            sys.executable,
            str(SCRIPT_DIR / "train_combined_population_critic.py"),
        ]
        for run_dir in args.run_dir:
            command.extend(["--run-dir", str(run_dir)])
        command.extend(
            [
                "--seed-run-dir",
                str(seed_run_dir),
                "--output-dir",
                str(member_dir),
                "--updates",
                str(args.updates),
                "--batch-size",
                str(args.batch_size),
                "--log-every",
                str(args.log_every),
                "--seed",
                str(args.seed + index),
            ]
        )
        if args.learning_rate is not None:
            command.extend(["--learning-rate", str(args.learning_rate)])
        if args.gamma is not None:
            command.extend(["--gamma", str(args.gamma)])
        if args.tau is not None:
            command.extend(["--tau", str(args.tau)])
        if args.load_optimizers:
            command.append("--load-optimizers")

        print(f"[{index}] training ensemble member from seed run: {seed_run_dir}")
        subprocess.run(command, check=True)

    return {
        "index": index,
        "seed_run_dir": str(seed_run_dir),
        "output_dir": str(member_dir),
        "checkpoint": str(checkpoint_path),
        "replay": str(replay_path),
        "summary": str(summary_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train an ensemble of combined population critics. Each member uses the same merged replay "
            "sources but starts from a different seed run checkpoint."
        )
    )
    parser.add_argument("--run-dir", action="append", required=True, type=Path)
    parser.add_argument(
        "--seed-run-dir",
        action="append",
        type=Path,
        help="Seed run for one ensemble member. Defaults to every --run-dir.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--load-optimizers", action="store_true")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> Dict[str, object]:
    args = parse_args()
    seed_run_dirs = args.seed_run_dir or args.run_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)

    members = []
    for index, seed_run_dir in enumerate(seed_run_dirs):
        member_dir = args.output_dir / safe_member_name(index, seed_run_dir)
        members.append(run_member(args, seed_run_dir, member_dir, index))

    manifest = {
        "run_dirs": [str(path) for path in args.run_dir],
        "seed_run_dirs": [str(path) for path in seed_run_dirs],
        "updates": int(args.updates),
        "batch_size": int(args.batch_size),
        "members": members,
    }
    manifest_path = args.output_dir / "ensemble_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved ensemble manifest: {manifest_path}")
    print(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    main()
