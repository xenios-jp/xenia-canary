"""Screen extents reports at the pixel edges and for the primitives and depth
cpext didn't draw. cpext found a report's x and y are the covered pixels'
bounding box, >> 3; these pin which pixels count at a fractional edge, and
what lines, points and depth do. As cpext's: an init, a draw, a report of
endianness 8in16 to output + 0.
- e0: a rectangle x 7.3-15.7, y 3.3-11.7 (covered pixels 8-15, 4-11; by its
  vertices floored 7-15, rounded 7-16).
- e1: x 7.6-7.9, y 3.6-3.9, covering no pixel center.
- e2: x 15.4-16.6, y 7.4-8.6, covering the pixel (16, 8) only.
- e3: x -0.5-8.5, y -0.5-0.5, pixels x 0-8 of row 0.
- line: a line list from (8.5, 4.5) to (23.5, 11.5).
- point: a point at (33, 17), PA_SU_POINT_SIZE 0x00080008 (one pixel);
  point 4: size 0x00200020 (4 pixels square, 31-34 and 15-18).
- z r6, z r6 nowrite, z r7: cpext's r6 (depth 0.25 to 0.75 across it) and
  r7 (depth 1) with a D24S8 depth buffer and the depth test on, ALWAYS, with
  and without depth writes: whether the report's z fields follow depth.
- tail: e0 again, a draw after z r7's report (Xenia's trace dump plays a
  frame only to its last draw).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cpext  # noqa: E402
import cpraw as cp  # noqa: E402
import gpugen  # noqa: E402
import pixel  # noqa: E402
import pm4  # noqa: E402

vertex = cpext.vertex
RECTS = [(7.3, 3.3, 15.7, 11.7), (7.6, 3.6, 7.9, 3.9),
         (15.4, 7.4, 16.6, 8.6), (-0.5, -0.5, 8.5, 0.5)]
# RB_DEPTHCONTROL: z enable, ALWAYS, with or without z write.
Z, Z_WRITE, ALWAYS = 2, 4, 7
DEPTH_INFO = 0x100  # D24S8 at tile 256, clear of the 64x64 window's color


def inputs():
    words = []
    for x0, y0, x1, y1 in RECTS:
        words += vertex(x0, y0, 0) + vertex(x1, y0, 0) + vertex(x0, y1, 0)
    words += vertex(8.5, 4.5, 0) + vertex(23.5, 11.5, 0) + vertex(33, 17, 0)
    # cpext's r6 and r7.
    for x0, y0, x1, y1, z in cpext.RECTS[6:8]:
        words += vertex(x0, y0, z[0]) + vertex(x1, y0, z[1]) + \
            vertex(x0, y1, z[2])
    return words


LINE, POINT, R6 = 3 * len(RECTS), 3 * len(RECTS) + 2, 3 * len(RECTS) + 3


def build(suite):
    cp.install(suite)
    suite.set_inputs(inputs(), 8)
    suite.export_slots = cpext.SLOTS
    shader = gpugen.interpolating_vertex_shader(8)

    def test(name, primitive, count, first, registers=()):
        words = (cpext.init() +
                 cp.draw_auto(primitive, count, first=first) +
                 cp.packet(cp.EVENT_WRITE_EXT, cp.SCREEN_EXT_RPT,
                           cp.Ref(output=name, low=1)))
        suite.test(name, shader, program_cntl=pixel.PROGRAM_CNTL,
                   registers=pixel.REGISTERS + list(registers),
                   after_passes=[cp.entry(*words)], draw=cpext.EMPTY)

    for k in range(len(RECTS)):
        test(f'e{k}', pm4.PRIM_RECTANGLE_LIST, 3, 3 * k)
    test('line', pm4.PRIM_LINE_LIST, 2, LINE)
    test('point', pm4.PRIM_POINT_LIST, 1, POINT)
    test('point 4', pm4.PRIM_POINT_LIST, 1, POINT,
         [('PA_SU_POINT_SIZE', [0x00200020])])
    for name, control, first in (('z r6', Z | Z_WRITE, R6),
                                 ('z r6 nowrite', Z, R6),
                                 ('z r7', Z | Z_WRITE, R6 + 3)):
        test(name, pm4.PRIM_RECTANGLE_LIST, 3, first,
             [('RB_DEPTH_INFO', [DEPTH_INFO]),
              ('RB_DEPTHCONTROL', [control | (ALWAYS << 4)])])
    test('tail', pm4.PRIM_RECTANGLE_LIST, 3, 0)
