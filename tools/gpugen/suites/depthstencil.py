"""Depth and stencil cases not in the stencil and msaadepth suites, through
depthcases: clears, conversions of Z and oDepth, out-of-range and special
oDepth, planes, compare functions on Z, polygon offsets and offset slopes.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import depthcases  # noqa: E402


def cases():
    """[(index, group, case)] of depthcases.main()'s this suite draws."""
    return [(i, g, c) for i, (g, c) in enumerate(depthcases.main())
            if not g.startswith(('stencil-', 'two-sided-', 'msaa-')) and
            not (g.startswith('compare-') and g.split('-')[2] == '1')]


def build(suite):
    depthcases.build(suite, cases())
