"""Manual NatNet rigid-body listing tool.

This script is intentionally safe to import so pytest collection does not attempt
hardware/network access.
"""

from __future__ import annotations

import argparse
import os
import time


def _fmt_pos(vec):
    return f"({vec[0]: .4f}, {vec[1]: .4f}, {vec[2]: .4f})"


def test_fmt_pos_formats_three_coordinates():
    assert _fmt_pos((1.0, -2.5, 0.125)) == "( 1.0000, -2.5000,  0.1250)"


def main():
    parser = argparse.ArgumentParser(description="List visible OptiTrack rigid bodies")
    parser.add_argument("--server", default=os.getenv("OPTITRACK_SERVER_IP","10.0.10.4"))
    parser.add_argument(
        "--frames",
        type=int,
        default=100,
        help="number of NatNet frames to read before exiting",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.05,
        help="delay (seconds) between run_once calls",
    )
    args, _ = parser.parse_known_args()

    try:
        import natnet
    except ModuleNotFoundError:
        print("natnet is not installed. Install it to run this connectivity check.")
        return 1

    try:
        client = natnet.Client.connect(args.server)
    except Exception as exc:  # pragma: no cover - network/hardware dependent
        print(f"Failed to connect to NatNet server '{args.server}': {exc}")
        return 1

    print(f"Connected to NatNet server: {args.server}")
    print("Printing visible rigid bodies. Move only the snake to identify its ID.")
    print("Press Ctrl+C to stop.\n")

    frame_count = 0

    def _callback(rigid_bodies, _markers, timing):
        nonlocal frame_count
        frame_count += 1
        timestamp = getattr(timing, "timestamp", time.time())
        print(f"[frame {frame_count:04d}] t={timestamp:.3f}s bodies={len(rigid_bodies)}")
        if not rigid_bodies:
            print("  (no rigid bodies visible)")
            return
        for body in rigid_bodies:
            body_id = getattr(body, "id_", "<unknown>")
            pos = getattr(body, "position", (0.0, 0.0, 0.0))
            orient = getattr(body, "orientation", (0.0, 0.0, 0.0, 0.0))
            print(f"  id={body_id:>3} pos(m)={_fmt_pos(pos)} quat(wxyz)={orient}")

    client.set_callback(_callback)

    try:
        for _ in range(args.frames):
            client.run_once()
            time.sleep(0.005)
    except KeyboardInterrupt:
        print("\nStopped by user.")
        return 0

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
