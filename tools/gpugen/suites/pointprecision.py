"""Raw point-coordinate precision around power-of-two diameters.

Two isolated points, centered at (16,16) and (40+1/16,40+7/16), export
interpolator (point index, center x, center y, 1) and raw PsParamGen per
covered pixel. The 24 diameters bracket powers of two by one 1/8 register
step, plus odd/fractional controls. No texture fetch or MSAA obscures the
point-coordinate generation. Uncovered pixels keep the sentinel.

This captures the non-power-of-two discrepancy found by pointcoord;
it deliberately does not assert that ideal division is the Xbox result.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pixel  # noqa: E402
import pm4  # noqa: E402

DIAMETERS = (1, 1.5, 1.875, 2, 2.125, 2.5, 3, 3.875, 4, 4.125,
             5, 6, 7, 7.5, 7.875, 8, 8.125, 9, 12, 15, 15.875,
             16, 16.125, 17)


def build(suite):
    centers = ((16, 16), (40.0625, 40.4375))
    draws = pixel.set_blocks(suite, [('points', pm4.PRIM_POINT_LIST,
              [pixel.vertex(p, (i, p[0], p[1], 1))
               for i, p in enumerate(centers)])])
    probe = pixel.Probe()
    for diameter in DIAMETERS:
        fixed = int(diameter * 8)
        probe.test(suite, f'point precision diameter {diameter:g}',
                   draws['points'], [('PA_SU_POINT_SIZE',
                                      [fixed | (fixed << 16)])])
