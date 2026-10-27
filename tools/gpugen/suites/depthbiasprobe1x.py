"""The pinned depthbiasprobe matrix at 1x, with independent sample readback."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import depthbiasprobe  # noqa: E402


def build(suite):
    depthbiasprobe.build_samples(suite, 1)
