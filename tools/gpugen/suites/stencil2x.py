"""Stencil operations, compares and two-sided state at both 2x samples.

Reuses depthcases' front/back, pass/fail/depth-fail, reference and mask cases.
Each output contains raw depth and stencil, so depth errors are distinguishable
from stencil errors. Every sample starts from the same explicit clear.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import depthcases  # noqa: E402


def cases():
    selected = []
    for i, (group, case) in enumerate(depthcases.main()):
        if group.startswith('two-sided-'):
            selected.append((i, group, case))
            continue
        if not group.startswith('stencil-'):
            continue
        state = case['draws'][0]['stencil']
        masks = state[-1]
        reference, read, write = masks & 255, (masks >> 8) & 255, masks >> 16
        initial = case['clear'] & 255
        if group.startswith('stencil-op-'):
            # Every operation at pass, stencil-fail and depth-fail, including
            # wrap/saturation endpoints. Partial writes use a nontrivial byte.
            keep = reference == 0x5A and (
                (write == 0xFF and initial in (0, 0xFF, 0x5A)) or
                (write in (0x0F, 0xF0) and initial == 0x5A))
        else:
            # Every compare below/equal/above its reference, plus read masks.
            keep = reference == 0x5A and (
                (read == 0xFF and initial in (0x59, 0x5A, 0x5B)) or
                (read in (0, 0x0F, 0xF0) and initial == 0x5B))
        if keep:
            selected.append((i, group, case))
    return selected


def build(suite):
    depthcases.build(suite, cases(), samples=2)
