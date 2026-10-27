"""Depth at 2x and 4x MSAA: the depth suite's D24S8 / D24FS8 buffer and raw
depth Resolve of sample 0, after a draw writing each pixel's depth (test
ALWAYS, no stencil) over depth 0.5 and stencil 0, either from the vertices
('Z': a rectangle per pixel at its Z) or from the pixel shader ('oDepth': the
X of a texture at the pixel, ps_export_mode 3, over Z 0.5). Pixel i's depth is
the i-th of the stencil suite's 40 chosen values for the samples and format,
then more of its depth values.

Vertices are 8 dwords (position, attribute), the Resolve's rectangle first,
drawn by the zdepth suite's interpolating vertex shader.
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import depth  # noqa: E402
import gpugen  # noqa: E402
import pm4  # noqa: E402
import rb  # noqa: E402
import stencil  # noqa: E402
import zdepth  # noqa: E402

WIDTH, HEIGHT = rb.WIDTH, rb.HEIGHT
ALWAYS, KEEP, REPLACE = stencil.ALWAYS, stencil.KEEP, stencil.REPLACE
f2w = gpugen.f2w


def rectangle(x0, y0, x1, y1, z):
    """A rectangle list's vertices over pixels x0..x1, y0..y1 at `z`."""
    return [f2w(v) for x, y in ((x0, y0), (x1, y0), (x1, y1))
            for v in (x - 0.5, y - 0.5, z, 1.0, 0, 0, 0, 1.0)]


def build(suite):
    words = rb.RECT + [0, 0]
    whole = (pm4.PRIM_RECTANGLE_LIST, len(words) // 8, 3)
    words += rectangle(0, 0, WIDTH, HEIGHT, 0.5)
    suite.add_texture('zero', rb.fill((0.0, 0.0, 0.0, 0.0)))
    suite.add_texture('half', rb.fill((0.5, 0.0, 0.0, 0.0)))
    vertex = gpugen.interpolating_vertex_shader(8)
    odepth_cntl = (zdepth.PROGRAM_CNTL & ~(0xF << 27)) | (3 << 27)
    values, _, chosen = stencil.chosen()
    more = random.Random(0x5C1)
    # Depth 0.5 and stencil 0 everywhere, then the test's draw.
    reset = stencil.state(stencil.control(stencil=(ALWAYS, KEEP, KEEP,
                                                   REPLACE)), stencil.masks(0))
    draw = stencil.state(stencil.control(), stencil.masks(0))
    for samples, code in ((2, 1), (4, 2)):
        for fname, fmt in (('D24S8', depth.D24S8), ('D24FS8', depth.D24FS8)):
            zs = list(chosen[samples, fmt])
            zs += [more.choice(values)
                   for _ in range(WIDTH * HEIGHT - len(zs))]
            pixels = (pm4.PRIM_RECTANGLE_LIST, len(words) // 8,
                      3 * len(zs))
            for i, z in enumerate(zs):
                x, y = i % WIDTH, i // WIDTH
                words += rectangle(x, y, x + 1, y + 1, z)
            texture = f'{samples}x {fname}'
            suite.add_texture(texture, [w for z in zs
                                        for w in (f2w(z), 0, 0, 0)])
            state = rb.registers(depth_format=fmt, samples=code,
                                 mode_cntl=gpugen.MODE_CNTL)
            for path, pixel, cntl, tex, geometry in (
                    ('Z', rb.pixel_shader(), zdepth.PROGRAM_CNTL, 'zero',
                     pixels),
                    ('oDepth', rb.pixel_shader(depth=True), odepth_cntl,
                     texture, whole)):
                rb.test(suite, f'{samples}x {fname} {path}', tex, state,
                        depth.RESOLVE, draw=geometry,
                        passes=[(reset, rb.fetch('half'), whole)],
                        after_passes=draw, shader=vertex, pixel=pixel,
                        program_cntl=cntl)
    suite.set_inputs(words, 8)
