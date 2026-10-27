"""The depth-bias suite's cases at 2x MSAA, as games draw polygon offset:
through depthcases, reading both samples' raw depth and stencil words.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import depthbias  # noqa: E402
import depthcases  # noqa: E402


def build(suite):
    depthcases.build(suite, depthbias.cases(), samples=2)
