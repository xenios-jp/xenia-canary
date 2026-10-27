"""Separate subpixel point-coordinate precision from integer translation.

For diameter 3 and 7.5, sweep eight sixteenth-pixel phases on X and Y
independently. Each draw repeats the same phase at centers (8,8), (23,23),
(38,38), (53,53), covering both 2x2 pixel-quad parities and translations.
The unshifted axis is an integer-center control. Raw interpolator contains
(point index, center X, center Y, 1); raw PsParamGen follows per pixel.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pixel  # noqa: E402
import pm4  # noqa: E402

PHASES = (0, 1, 2, 7, 8, 9, 14, 15)


def build(suite):
    blocks = []
    for axis in (0, 1):
        for phase in PHASES:
            vertices = []
            for i, center in enumerate((8, 23, 38, 53)):
                p = [center, center]
                p[axis] += phase / 16
                vertices.append(pixel.vertex(p, (i, p[0], p[1], 1)))
            blocks.append((f'{axis}:{phase}', pm4.PRIM_POINT_LIST, vertices))
    draws = pixel.set_blocks(suite, blocks)
    probe = pixel.Probe()
    for diameter in (3, 7.5):
        fixed = int(diameter * 8)
        for axis in (0, 1):
            for phase in PHASES:
                probe.test(suite,
                           f'point diameter {diameter:g} {"xy"[axis]} '
                           f'phase {phase}/16',
                           draws[f'{axis}:{phase}'],
                           [('PA_SU_POINT_SIZE', [fixed | (fixed << 16)])])
