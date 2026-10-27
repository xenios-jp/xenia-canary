"""The pinned depthround matrix at 1x, resolving every sample independently."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import depthround  # noqa: E402


def build(suite):
    depthround.build_samples(suite, 1)
