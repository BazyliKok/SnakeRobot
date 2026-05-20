#!/usr/bin/env python3
"""Simple terminal monitor for OptiTrack position/heading.

Run this in a separate terminal while training/running other scripts to verify
whether NatNet pose updates are changing over time.
"""

import argparse
import os
import sys
import time


def _load_optitrack_class():
    """Import Optitrack lazily so pytest collection stays side-effect free."""
    try:
        from CoadaptationCode.optitrack import Optitrack
        return Optitrack
    except ModuleNotFoundError:
        from optitrack import Optitrack
        return Optitrack


def _changed(curr, prev, eps):
    if prev is None:
        return True
    return any(abs(c - p) > eps for c, p in zip(curr, prev))


def test_changed_detects_initial_and_thresholded_pose_changes():
    assert _changed([0.0, 0.0, 0.0], None, 1e-5)
    assert not _changed([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 1e-5)
    assert not _changed([0.0, 0.0, 0.0], [0.0, 0.0, 0.000001], 1e-5)
    assert _changed([0.0, 0.0, 0.0], [0.0, 0.0, 0.001], 1e-5)


def main():
    parser = argparse.ArgumentParser(description="Monitor OptiTrack pose updates")
    parser.add_argument("--interval", type=float, default=0.1, help="seconds between polls")
    parser.add_argument("--epsilon", type=float, default=1e-5, help="change threshold for stale detection")
    args, _ = parser.parse_known_args()

    rigid_id = 99
    print(f"Starting OptiTrack monitor. OPTITRACK_RIGID_BODY_ID={rigid_id}")
    print("Press Ctrl+C to stop.\n")

    optitrack_cls = _load_optitrack_class()

    # Optitrack also parses argv; keep only this script name so our custom args
    # here do not interfere with its parser.
    old_argv = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        opti = optitrack_cls()
    finally:
        sys.argv = old_argv

    prev_coord = None
    stale_count = 0

    try:
        while True:
            coord, heading = opti.optiTrackGetPos()
            coord = [float(v) for v in coord]
            heading = [float(v) for v in heading]

            is_new = _changed(coord, prev_coord, args.epsilon)
            if is_new:
                stale_count = 0
                status = "UPDATED"
            else:
                stale_count += 1
                status = f"STALE x{stale_count}"

            print(
                f"[{time.strftime('%H:%M:%S')}] {status} "
                f"pos(m)=({coord[0]: .4f}, {coord[1]: .4f}, {coord[2]: .4f}) "
                f"heading(deg)=({heading[0]: .2f}, {heading[1]: .2f}, {heading[2]: .2f})"
            )

            prev_coord = coord
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\nStopped OptiTrack monitor.")


if __name__ == "__main__":
    main()
