"""log over the binade [0.5, 1), every mantissa, in the sweep suite's
layout. log's fixed point result has 25 fraction bits, and the [1, 2) sweep
loses the last one wherever log2(m) >= 0.5 (its float has 24 bits); here
the result is -(1 - log2(m)) less one fixed point unit, a float of under 0.5
in magnitude there, which keeps it."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep  # noqa: E402
import xenos as x  # noqa: E402


def build(suite):
    sweep.SWEEPS['logh'] = (x.S_LOG, 0.5)
    os.environ['GPUGEN_SWEEP'] = 'logh'
    os.environ.pop('GPUGEN_VERTICES', None)
    sweep.build(suite)
