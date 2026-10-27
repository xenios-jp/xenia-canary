"""Vertex snapping for every PA_SU_VTX_CNTL round_mode and quant_mode, the
combinations snap leaves out: quant_mode 2, 3 and 7 over snap's coarse rows,
and all 32 over wide rows, whose edges move by up to a pixel either way, so
the 1/4, 1/2 and 1 pixel modes and the roundings tell apart.

The layout is snap's: row k holds quad A, from x -0.5 to 8 + d_k, and quad
B, from 40 + d_k to 47.5, with wide d_k = (k - 32) / 32.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pixel  # noqa: E402
import pm4  # noqa: E402
import snap  # noqa: E402

BLOCKS = [('snap coarse', pm4.PRIM_TRIANGLE_LIST, snap.rows(256)),
          ('snap wide', pm4.PRIM_TRIANGLE_LIST, snap.rows(32))]


def build(suite):
    draws = pixel.set_blocks(suite, BLOCKS)
    probe = pixel.Probe()

    def test(block, round_mode, quant_mode):
        probe.test(
            suite, f'{block} round {round_mode} quant {quant_mode}',
            draws[block],
            [('PA_SU_VTX_CNTL', [(round_mode << 1) | (quant_mode << 3)])])

    for round_mode in range(4):
        for quant_mode in (2, 3, 7):
            test('snap coarse', round_mode, quant_mode)
    for round_mode in range(4):
        for quant_mode in range(8):
            test('snap wide', round_mode, quant_mode)
