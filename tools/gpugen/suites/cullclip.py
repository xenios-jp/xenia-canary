"""Face culling, user clip planes and PA_CL_VTE_CNTL 0x400 as titles draw
with them (feature census holes: cull 'back face cw', 8.8% of draws; clip
'ucp 0x1 dx', 'ucp 0x3 dx', 'ucp 0x0 disabled dx', 'ucp 0x3 disabled dx';
vte 0x400), through the pixel suite's probe: each pixel of the 64x64 window
exports its interpolated attribute and PsParamGen, so the outputs are a
coverage map. Every attribute is (X, Y, X + Y, 1) of the vertex's screen
position.
- cull: two triangles, the left (pixels x 0-15, y 0-15) wound like the pixel
  suites' rectangles (top left, top right, bottom left), the right (x 32-47)
  the other way, under PA_SU_SC_MODE_CNTL cull none, front and back with
  face ccw and cw. 'viewport' ones draw them from clip space through
  Direct3D's viewport (PA_CL_VTE_CNTL 0x43F, scale (32, -32), offset (32,
  32), y up), clipping on in Direct3D's clip space (PA_CL_CLIP_CNTL
  0x80000), as titles do.
- clip: a quad over the whole window from clip space through that viewport,
  PA_CL_CLIP_CNTL as titles set it: 0x80001 with plane 0 (0.25, -0.5, 0,
  0.1) (kept where the dot product with the clip-space position is at least
  0), 0x80003 with plane 1 (-0.5, -0.25, 0, 0.2) too, 0x90000 and 0x90003
  (clipping disabled, the planes enabled or not).
- vte 400: PA_CL_VTE_CNTL 0x400 (no viewport, xy and z not marked as
  divided, W as 1/W), clipping disabled (0x90000), as titles draw with it:
  the window quad with W 1, and with X, Y, Z and W all 2 or all 0.5.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpugen  # noqa: E402
import pixel  # noqa: E402
import pm4  # noqa: E402

f2w = gpugen.f2w
CULL = {'none': 0, 'front': 1, 'back': 2}
FACE_CW = 1 << 2
CLIP_DX, CLIP_DISABLE = 1 << 19, 1 << 16
SCALE = 32.0


def vertex(position, screen):
    sx, sy = screen
    return [f2w(v) for v in position] + \
        [f2w(v) for v in (sx, sy, sx + sy, 1.0)]


def screen(points):
    return [vertex((x, y, 0.0, 1.0), (x, y)) for x, y in points]


def clip(points):
    """Clip-space positions the viewport maps to these screen points."""
    return [vertex(((x - SCALE) / SCALE, (SCALE - y) / SCALE, 0.5, 1.0),
                   (x, y)) for x, y in points]


def scaled(points, w):
    return [vertex((x * w, y * w, 0.5 * w, w), (x, y)) for x, y in points]


LEFT = [(-0.5, -0.5), (15.5, -0.5), (-0.5, 15.5)]
RIGHT = [(31.5, 15.5), (47.5, -0.5), (31.5, -0.5)]
QUAD = [(-0.5, -0.5), (63.5, -0.5), (-0.5, 63.5),
        (63.5, -0.5), (63.5, 63.5), (-0.5, 63.5)]
BLOCKS = [
    ('pair', pm4.PRIM_TRIANGLE_LIST, screen(LEFT + RIGHT)),
    ('pair clip', pm4.PRIM_TRIANGLE_LIST, clip(LEFT + RIGHT)),
    ('quad clip', pm4.PRIM_TRIANGLE_LIST, clip(QUAD)),
    ('quad', pm4.PRIM_TRIANGLE_LIST, screen(QUAD)),
    ('quad w2', pm4.PRIM_TRIANGLE_LIST, scaled(QUAD, 2.0)),
    ('quad w0.5', pm4.PRIM_TRIANGLE_LIST, scaled(QUAD, 0.5)),
]
VIEWPORT = [('PA_CL_VTE_CNTL', [0x43F]),
            ('PA_CL_VPORT_XSCALE', [f2w(v) for v in (SCALE, SCALE, -SCALE,
                                                      SCALE, 1.0, 0.0)])]
PLANES = [('PA_CL_UCP_0_X', [f2w(v) for v in (0.25, -0.5, 0.0, 0.1)]),
          ('PA_CL_UCP_1_X', [f2w(v) for v in (-0.5, -0.25, 0.0, 0.2)])]


def mode(cull, cw=False):
    return ('PA_SU_SC_MODE_CNTL', [gpugen.MODE_CNTL | CULL[cull] |
                                   (FACE_CW if cw else 0)])


def build(suite):
    draws = pixel.set_blocks(suite, BLOCKS)
    probe = pixel.Probe()
    for face in ('ccw', 'cw'):
        for cull in ('none', 'front', 'back'):
            probe.test(suite, f'cull {cull} face {face}', draws['pair'],
                       [mode(cull, face == 'cw')])
    for face in ('ccw', 'cw'):
        for cull in ('none', 'back'):
            probe.test(suite, f'viewport cull {cull} face {face}',
                       draws['pair clip'],
                       [mode(cull, face == 'cw'),
                        ('PA_CL_CLIP_CNTL', [CLIP_DX])] + VIEWPORT)
    for cntl in (CLIP_DX | 1, CLIP_DX | 3, CLIP_DX | CLIP_DISABLE,
                 CLIP_DX | CLIP_DISABLE | 3):
        probe.test(suite, f'clip {cntl:x}', draws['quad clip'],
                   [('PA_CL_CLIP_CNTL', [cntl])] + VIEWPORT + PLANES)
    for block in ('quad', 'quad w2', 'quad w0.5'):
        probe.test(suite, f'vte 400 {block}', draws[block],
                   [('PA_CL_CLIP_CNTL', [CLIP_DX | CLIP_DISABLE]),
                    ('PA_CL_VTE_CNTL', [0x400])])
