"""Manual NatNet connectivity check.

This script is intentionally safe to import so pytest collection does not attempt
hardware/network access.
"""


def main():
    try:
        import natnet
    except ModuleNotFoundError:
        print("natnet is not installed. Install it to run this connectivity check.")
        return 1

    client = natnet.Client.connect("10.0.10.2")
    print("connected")
    client.set_callback(lambda rigid_bodies, markers, timing: print(rigid_bodies))
    client.run_once()
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

