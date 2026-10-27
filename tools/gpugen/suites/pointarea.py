"""Hold point width fixed while varying area through the other dimension.

Widths 4.125 and 7 each use heights 2,3,4,5,7,8,9,16. The power-of-two
height controls distinguish area-reciprocal precision from general axis
coupling: scaled areas should produce identical X gradients if reciprocal
arithmetic scales with the exponent. Integer/subpixel centers and raw
exports match pointprecision and pointrect. No shader point size or MSAA.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pixel  # noqa: E402
import pm4  # noqa: E402


def build(suite):
    centers = ((16, 16), (40.0625, 40.4375))
    draws = pixel.set_blocks(suite, [('points', pm4.PRIM_POINT_LIST,
              [pixel.vertex(p, (i, p[0], p[1], 1))
               for i, p in enumerate(centers)])])
    probe = pixel.Probe()
    for width in (4.125, 7):
        for height in (2, 3, 4, 5, 7, 8, 9, 16):
            probe.test(suite, f'point area {width:g}x{height:g}',
                       draws['points'],
                       [('PA_SU_POINT_SIZE',
                         [int(height * 8) | (int(width * 8) << 16)]),
                        ('PA_SU_POINT_MINMAX', [0xFFFF0000])])
