"""Depth compares in the states the stencil suite doesn't draw: the depth test
without writes, and without stencil. Over the stencil suite's reset (depth
0.5, stencil 0, color (0.5, 0, 0, 0)), each pixel gets a draw of depth x
(ALWAYS, written, stencil REPLACE 0x10, color (x, 0, 1, 0)), then one of
depth y (color (y, 1, 0, 0)) with the compare function in one of four states,
a band of 8 rows each:

- rows 0-7: depth test only, stencil off;
- rows 8-15: depth test only, stencil ALWAYS, zfail INCR_CLAMP, zpass REPLACE
  0x5A;
- rows 16-23: depth test and write, stencil off;
- rows 24-31: depth test and write, stencil as rows 8-15 (the stencil suite's
  'compare' state, as a check).

Every band draws the same 512 (x, y) pairs, the stencil suite's first for
the format and function, so the states compare pixel for pixel. Each test
runs twice: 'color' resolves the 8_8_8_8 color target, whose green says which
draw won, and 'depth' the raw depth and stencil words, as the stencil suite.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import depth  # noqa: E402
import gpugen  # noqa: E402
import pm4  # noqa: E402
import rb  # noqa: E402
import stencil as st  # noqa: E402

WIDTH, HEIGHT = rb.WIDTH, rb.HEIGHT
BAND = 8
PAIRS = WIDTH * BAND
STENCIL = (st.ALWAYS, st.KEEP, st.INCR_CLAMP, st.REPLACE)
# (depth writes, stencil) per band.
BANDS = ((False, None), (False, STENCIL), (True, None), (True, STENCIL))
COLOR_RESOLVE = gpugen.resolve_registers(WIDTH, HEIGHT, 6, 4)
f2w = gpugen.f2w


def build(suite):
    rects = {}
    inputs = list(rb.RECT)

    def band(y0, y1):
        if (y0, y1) not in rects:
            first = len(inputs) // 2
            for px, py in ((0, y0), (WIDTH, y0), (WIDTH, y1)):
                inputs.extend((f2w(px - 0.5), f2w(py - 0.5)))
            rects[y0, y1] = (pm4.PRIM_RECTANGLE_LIST, first, 3)
        return rects[y0, y1]

    whole = band(0, HEIGHT)
    suite.add_texture('half', rb.fill((0.5, 0.0, 0.0, 0.0)))
    reset = (st.state(st.control(stencil=(st.ALWAYS, st.KEEP, st.KEEP,
                                          st.REPLACE)), st.masks(0)),
             'half', whole)
    pairs = st.compare_pairs()
    for fname, fmt in (('D24S8', depth.D24S8), ('D24FS8', depth.D24FS8)):
        for func in range(8):
            chosen = pairs[fmt, func][:PAIRS]
            first, second = f'{fname} {func} x', f'{fname} {func} y'
            suite.add_texture(first, [w for _ in BANDS for x, _ in chosen
                                      for w in (f2w(x), 0, f2w(1.0), 0)])
            suite.add_texture(second, [w for _ in BANDS for _, y in chosen
                                       for w in (f2w(y), f2w(1.0), 0, 0)])
            draws = [(st.state(st.control(stencil=(st.ALWAYS, st.KEEP,
                                                   st.KEEP, st.REPLACE)),
                               st.masks(0x10)), first, whole)]
            for b, (write, stencil) in enumerate(BANDS):
                draws.append((st.state(st.control(func, write, stencil),
                                       st.masks(0x5A)), second,
                              band(b * BAND, (b + 1) * BAND)))
            for what, resolve in (('color', COLOR_RESOLVE),
                                  ('depth', depth.RESOLVE)):
                passes, fetched = [], None
                for registers, texture, draw in [reset] + draws[:-1]:
                    passes.append((registers, rb.fetch(texture)
                                   if texture != fetched else {}, draw))
                    fetched = texture
                registers, texture, draw = draws[-1]
                rb.test(suite, f'{fname} {func} {what}', texture,
                        rb.registers(depth_format=fmt), resolve, draw=draw,
                        passes=passes, after_passes=registers,
                        pixel=rb.pixel_shader(depth=True),
                        program_cntl=st.PROGRAM_CNTL)
    suite.set_inputs(inputs, 2)
