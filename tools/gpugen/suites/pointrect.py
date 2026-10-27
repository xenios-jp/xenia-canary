"""Rectangular fixed points separate X/Y precision from primitive extent.

For six diameters rejecting the preliminary affine precision model, compare
(width,4), (4,height), and square controls at the same integer and subpixel
centers as pointprecision. All bounds lie inside the 64x64 scissor. Fixed
size comes from PA_SU_POINT_SIZE, whose upper/lower fields are width/height
radii in 12.4 format; no shader point size is exported. MINMAX is explicitly
wide as a control. Raw interpolator and PsParamGen are exported per pixel.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pixel  # noqa: E402
import pm4  # noqa: E402

DIAMETERS = (2.125, 4.125, 7, 8.125, 15.875, 17)


def build(suite):
    centers = ((16, 16), (40.0625, 40.4375))
    draws = pixel.set_blocks(suite, [('points', pm4.PRIM_POINT_LIST,
              [pixel.vertex(p, (i, p[0], p[1], 1))
               for i, p in enumerate(centers)])])
    probe = pixel.Probe()
    for diameter in DIAMETERS:
        for width, height in ((diameter, 4), (4, diameter),
                              (diameter, diameter)):
            probe.test(suite, f'point rectangle {width:g}x{height:g}',
                       draws['points'],
                       [('PA_SU_POINT_SIZE',
                         [int(height * 8) | (int(width * 8) << 16)]),
                        ('PA_SU_POINT_MINMAX', [0xFFFF0000])])
