"""Screen extents reports (feature census: EVENT_WRITE_EXT, 7.9% of the
titles' packets), as predicated tiling issues them: after a draw,
EVENT_WRITE_EXT SCREEN_EXT_RPT (event 26) to an address with endianness
8in16, then EVENT_WRITE SCREEN_EXT_INIT (event 25), both predicated on bin
mask 0x80000000.

Every test starts with SCREEN_EXT_INIT, then draws rectangles into the 64x64
window of the pixel suites (screen-space positions, pixel centers on
integers, edges half a pixel from them, depth from the vertices, color
writes, no depth test) and reports to output + 16 k for report k; the output
starts out the sentinel, so it shows the bytes a report writes.
Rectangles (pixels covered, inclusive):
- r0: x 0-15, y 0-15; r1: x 8-23, y 4-11; r2: x 37-50, y 21-42;
  r3: the pixel (33, 17); r4: x 56-63 of x 56-79, y 40-47, cut by the
  window scissor; r5: x 0-63, y 0-63, from vertices far outside the window;
  r6: x 16-31, y 32-47 at depth 0.25 to 0.75 across it; r7: x 0-7, y 0-7
  at depth 1.
- none: a report with no draw since the init.
- each rN alone.
- r1 r2: two draws, one report. r1 then r2: a report after each, no init
  between. r1 init r2: an init between.
- r1 tri: r1's first triangle wound the other way, as a triangle list;
  r1 tri cull front, r1 tri cull back: culling either face. r1 mask 0: r1 with
  RB_COLOR_MASK 0. r1 scissored: r1 with the window scissor at (32, 32)
  (it covers none of it).
- r1 offset: r1 with PA_SC_WINDOW_OFFSET (16, 8) and
  vtx_window_offset_enable.
- endian 0, endian 2: r1 reported with those.
- predicated pass, predicated skip: the titles' sequence, with bin select
  0x80000003 and 0xC.
- twice: two reports of r1 with no init or draw between.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cpraw as cp  # noqa: E402
import gpugen  # noqa: E402
import pixel  # noqa: E402
import pm4  # noqa: E402

f2w = gpugen.f2w
EMPTY = (pm4.PRIM_POINT_LIST, 0, 0)
SLOTS = 8
CULL_FRONT, CULL_BACK = 1, 2

# (x0, y0, x1, y1, z at the first, second and third vertex)
RECTS = [(-0.5, -0.5, 15.5, 15.5, (0, 0, 0)),
         (7.5, 3.5, 23.5, 11.5, (0, 0, 0)),
         (36.5, 20.5, 50.5, 42.5, (0, 0, 0)),
         (32.5, 16.5, 33.5, 17.5, (0, 0, 0)),
         (55.5, 39.5, 79.5, 47.5, (0, 0, 0)),
         (-1000.5, -1000.5, 1000.5, 1000.5, (0, 0, 0)),
         (15.5, 31.5, 31.5, 47.5, (0.25, 0.5, 0.75)),
         (-0.5, -0.5, 7.5, 7.5, (1, 1, 1))]


def vertex(px, py, z):
    return [f2w(v) for v in (px, py, z, 1.0)] + [0] * 4


def inputs():
    words = []
    for x0, y0, x1, y1, z in RECTS:
        # A rectangle list's three vertices: top left, top right, bottom
        # left (counterclockwise on screen, y down, as the pixel suites').
        words += vertex(x0, y0, z[0]) + vertex(x1, y0, z[1]) + \
            vertex(x0, y1, z[2])
    # r1 again, clockwise, as a triangle.
    x0, y0, x1, y1, _ = RECTS[1]
    words += vertex(x0, y0, 0) + vertex(x0, y1, 0) + vertex(x1, y0, 0)
    return words


CW = len(RECTS)


def draw(k, primitive=pm4.PRIM_RECTANGLE_LIST):
    return cp.draw_auto(primitive, 3, first=3 * k)


def init(predicate=False):
    return cp.packet(pm4.EVENT_WRITE, cp.SCREEN_EXT_INIT, predicate=predicate)


def build(suite):
    cp.install(suite)
    suite.set_inputs(inputs(), 8)
    suite.export_slots = SLOTS
    shader = gpugen.interpolating_vertex_shader(8)

    def test(name, *steps, registers=()):
        reports = 0
        words = init()
        for step in steps:
            if step == 'report' or (isinstance(step, tuple) and
                                    step[0] == 'report'):
                endian, predicate = (step[1], step[2]) if isinstance(
                    step, tuple) else (1, False)
                words += cp.packet(cp.EVENT_WRITE_EXT, cp.SCREEN_EXT_RPT,
                                   cp.Ref(output=name, offset=16 * reports,
                                          low=endian), predicate=predicate)
                reports += 1
            elif step == 'init':
                words += init()
            else:
                words += step
        suite.test(name, shader, program_cntl=pixel.PROGRAM_CNTL,
                   registers=pixel.REGISTERS + list(registers),
                   after_passes=[cp.entry(*words)], draw=EMPTY)

    test('none', 'report')
    for k in range(len(RECTS)):
        test(f'r{k}', draw(k), 'report')
    test('r1 r2', draw(1), draw(2), 'report')
    test('r1 then r2', draw(1), 'report', draw(2), 'report')
    test('r1 init r2', draw(1), 'report', 'init', draw(2), 'report')
    test('r1 tri', draw(CW, pm4.PRIM_TRIANGLE_LIST), 'report')
    for cull, bit in (('front', CULL_FRONT), ('back', CULL_BACK)):
        test(f'r1 tri cull {cull}', draw(CW, pm4.PRIM_TRIANGLE_LIST),
             'report', registers=[('PA_SU_SC_MODE_CNTL',
                                   [gpugen.MODE_CNTL | bit])])
    test('r1 mask 0', draw(1), 'report', registers=[('RB_COLOR_MASK', [0])])
    test('r1 scissored', draw(1), 'report',
         registers=[('PA_SC_WINDOW_SCISSOR_TL', [(32 << 16) | 32])])
    test('r1 offset', draw(1), 'report',
         registers=[('PA_SC_WINDOW_OFFSET', [(8 << 16) | 16])])
    test('endian 0', draw(1), ('report', 0, False))
    test('endian 2', draw(1), ('report', 2, False))
    test('twice', draw(1), 'report', 'report')
    for result, select in (('pass', 0x80000003), ('skip', 0xC)):
        test(f'predicated {result}',
             cp.bins(0xFFFFFFFF, select),
             cp.draw_auto(pm4.PRIM_RECTANGLE_LIST, 3, predicate=True,
                          first=3),
             cp.packet(cp.SET_BIN_MASK_LO, 0x80000000),
             ('report', 1, True), init(True),
             cp.bins(*cp.BINS_DEFAULT))
