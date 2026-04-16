"""Deprecated manual policy runner.

The previous version of this script was hard-coded to an old 6-action policy
and a 6-motor write command. The current project uses train_coadapt.py with the
8-module design encoding and 7-motor action space.
"""


if __name__ == '__main__':
    raise RuntimeError(
        "testCoadapt.py is deprecated for the current 8-module/7-motor project. "
        "Use train_coadapt.py for legacy-policy warm-start training."
    )
