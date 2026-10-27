"""The sweep suite's rcp binade, [1, 2), as a suite of its own, so a queued
console run needs no GPUGEN_SWEEP in its environment."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep  # noqa: E402


def build(suite):
    os.environ['GPUGEN_SWEEP'] = 'rcp'
    os.environ.pop('GPUGEN_VERTICES', None)
    sweep.build(suite)
