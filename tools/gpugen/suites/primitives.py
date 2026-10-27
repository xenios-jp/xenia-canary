"""Primitive types games draw that the pixel suites don't: triangle strips,
triangle fans, quad lists and line strips, through the pixel suite's probe
(each pixel exports its interpolated attribute and PsParamGen, so the outputs
are a coverage map too).

Each vertex's attribute is (one-hot weight, its index k): the weight's
component is chosen so a primitive's vertices use different ones, so a
smooth pixel's xyz are its barycentrics and a flat one's w names the
provoking vertex.
- strip: 8 vertices, k at (10 (k // 2) + 0.25, 0.25 or 14.25), component
  k % 3. Also culled front and back: a strip's even triangles wind one way
  and its odd ones the other.
- fan: a center and 6 rim vertices, the center component 0, the rim 1 and 2
  alternately.
- quads: a square, a kite and a dart (concave), component k % 4 with the
  fourth vertex (0, 0, 0): which diagonal splits each quad shows in the
  weights.
- line strip: 5 vertices zigzagging across the window, component k % 2.
Each block runs smooth, flat (provoking vertex first) and flat with the last
vertex provoking.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpugen  # noqa: E402
import pixel  # noqa: E402
import pm4  # noqa: E402

CULL_FRONT, CULL_BACK = 1, 2


def weighted(points, component):
    out = []
    for k, p in enumerate(points):
        weight = [0.0, 0.0, 0.0]
        c = component(k)
        if c < 3:
            weight[c] = 1.0
        out.append(pixel.vertex(p, weight + [float(k)]))
    return out


STRIP = [(10 * (k // 2) + 0.25, 14.25 if k % 2 else 0.25) for k in range(8)]
FAN = [(20.25, 20.25), (40.25, 21.25), (35.25, 37.25), (19.25, 41.25),
       (3.25, 33.75), (1.75, 14.25), (13.25, 1.25)]
QUADS = [(0.25, 0.25), (15.25, 0.25), (15.25, 15.25), (0.25, 15.25),
         (20.25, 0.25), (40.25, 4.25), (36.25, 18.25), (18.25, 14.25),
         (44.25, 0.25), (62.25, 0.25), (53.25, 6.25), (53.25, 18.25)]
LINE_STRIP = [(0.25, 2.25), (20.75, 9.5), (30.25, 1.75), (50.5, 13.25),
              (63.25, 4.75)]
BLOCKS = [
    ('strip', pm4.PRIM_TRIANGLE_STRIP, weighted(STRIP, lambda k: k % 3)),
    ('fan', pm4.PRIM_TRIANGLE_FAN,
     weighted(FAN, lambda k: 0 if k == 0 else 1 + (k - 1) % 2)),
    ('quads', pm4.PRIM_QUAD_LIST, weighted(QUADS, lambda k: k % 4)),
    ('line strip', pm4.PRIM_LINE_STRIP,
     weighted(LINE_STRIP, lambda k: k % 2)),
]


def build(suite):
    draws = pixel.set_blocks(suite, BLOCKS)
    probe = pixel.Probe()
    flat = ('SQ_INTERPOLATOR_CNTL', [0])
    for name, _, _ in BLOCKS:
        probe.test(suite, name, draws[name])
        probe.test(suite, f'{name} flat', draws[name], [flat])
        probe.test(suite, f'{name} flat last', draws[name], [
            flat, ('PA_SU_SC_MODE_CNTL',
                   [gpugen.MODE_CNTL | pixel.PROVOKING_LAST])])
    for cull, bit in (('front', CULL_FRONT), ('back', CULL_BACK)):
        probe.test(suite, f'strip cull {cull}', draws['strip'], [
            ('PA_SU_SC_MODE_CNTL', [gpugen.MODE_CNTL | bit])])
