#!/usr/bin/env python3
"""Simple terminal monitor for OptiTrack position/heading.

Run this in a separate terminal while training/running other scripts to verify
whether NatNet pose updates are changing over time.
"""

import argparse
import os
import sys
import time

from optitrack import Optitrack


def _changed(curr, prev, eps):
    if prev is None:
        return True
    return any(abs(c - p) > eps for c, p in zip(curr, prev))


def main():
    parser = argparse.ArgumentParser(description="Monitor OptiTrack pose updates")
    parser.add_argument("--interval", type=float, default=0.1, help="seconds between polls")
    parser.add_argument("--epsilon", type=float, default=1e-5, help="change threshold for stale detection")
    args, _ = parser.parse_known_args()

    rigid_id = os.getenv("OPTITRACK_RIGID_BODY_ID", "<unset; first tracked body>")
    print(f"Starting OptiTrack monitor. OPTITRACK_RIGID_BODY_ID={rigid_id}")
    print("Press Ctrl+C to stop.\n")

    # Optitrack also parses argv; keep only this script name so our custom args
    # here do not interfere with its parser.
    old_argv = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        opti = Optitrack()
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
