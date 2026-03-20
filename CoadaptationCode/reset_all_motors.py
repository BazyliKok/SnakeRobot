import argparse
import sys

from motorssynced import MotorsSynced


def _build_targets(args, motor_count):
    if args.positions is not None:
        if len(args.positions) != motor_count:
            raise ValueError(
                f"Expected {motor_count} positions, received {len(args.positions)}."
            )
        return [int(position) for position in args.positions]

    return [int(args.center)] * motor_count


def main():
    parser = argparse.ArgumentParser(
        description="Reset all DYNAMIXEL motors to a known position."
    )
    parser.add_argument(
        "--center",
        type=int,
        default=2048,
        help="single target position applied to every motor (default: 2048)",
    )
    parser.add_argument(
        "--positions",
        type=int,
        nargs="+",
        help="explicit target positions for all motors, e.g. --positions 2048 2048 2048 2048 2048 2048 2048",
    )
    parser.add_argument(
        "--leave-torque-on",
        action="store_true",
        help="leave torque enabled after reset completes",
    )
    args = parser.parse_args()

    motors = None
    try:
        motors = MotorsSynced()
        targets = _build_targets(args, len(motors.DXL_ID))

        print(f"Motor IDs: {motors.DXL_ID}")
        print(f"Reset targets: {targets}")
        motors.reportHardwareErrorStatuses(context="before standalone reset")

        reset_ok = motors.resetMotorPositions(
            targets,
            disable_after_reset=not args.leave_torque_on,
        )

        if not reset_ok:
            print("Standalone motor reset failed.")
            motors.reportHardwareErrorStatuses(context="after standalone reset failure")
            return 1

        print("Standalone motor reset completed successfully.")
        motors.reportHardwareErrorStatuses(context="after standalone reset")
        return 0
    except KeyboardInterrupt:
        print("Standalone motor reset interrupted by user.")
        return 130
    except Exception as exc:
        print(f"Standalone motor reset crashed: {exc}")
        return 1
    finally:
        if motors is not None:
            try:
                motors.portHandler.closePort()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
