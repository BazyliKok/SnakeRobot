"""Deprecated manual policy runner.

The previous version of this script was hard-coded to an old 6-action policy
and a 6-motor write command. The current project uses train_coadapt.py with the
continuous A/B scale-parameter encoding and 7-motor action space.
"""


if __name__ == '__main__':
    raise RuntimeError(
        "testCoadapt.py is deprecated for the current scale-parameter/7-motor project. "
        "Use train_coadapt.py for fresh scale-parameter training or new-schema resume."
    )
