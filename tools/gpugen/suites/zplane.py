"""Depth from interpolated Z for the zplane suites: triangles with chosen
vertex positions and Z into the 64x32 D24S8 or D24FS8 depth buffer, cleared
first, depth test ALWAYS with writes, then the depth buffer resolved raw as
the depth suite does (64x32 words, depth << 8 | stencil 0x5A), at a chosen
sample when multisampled.

Vertices go through Direct3D's viewport path (PA_CL_VTE_CNTL 0x43F: X, Y
and Z divided by W, then scaled and offset) with clipping off; a test's
viewport is (x scale, x offset, y scale, y offset, z scale, z offset), by
default the identity, so a vertex (x, y, z, 1) lands at window (x, y) with
depth z. Pixel centers are whole window coordinates (PA_SU_VTX_CNTL 4).

test() takes draws, each a dict of the triangle list's vertices ((x, y, z,
w) floats), and optional polygon offset (front scale, front offset, back
scale, back offset with both enables), cull mode and depth compare. The
draws of a test go in order; one draw's state is set before it.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpugen  # noqa: E402
import pm4  # noqa: E402
import rb  # noqa: E402
import rbdraw  # noqa: E402

D24S8, D24FS8 = 0, 1
FORMATS = ('D24S8', 'D24FS8')
WIDTH, HEIGHT = rb.WIDTH, rb.HEIGHT
VTE_VIEWPORT = 0x43F
IDENTITY = (1.0, 0.0, 1.0, 0.0, 1.0, 0.0)
POLY_OFFSET_ENABLE = 0x1800
# z_enable, z_write_enable, zfunc, stencil ALWAYS REPLACE 0x5A.
DEPTHCONTROL = 0x1 | 0x2 | 0x4 | (7 << 8) | (2 << 14)
# 8 VS and 4 PS registers, vertex and pixel fetches, PsParamGen, one
# interpolator, Direct3D's PS export mode.
PROGRAM_CNTL = 7 | (3 << 8) | (1 << 16) | (1 << 17) | (1 << 18) | (2 << 27)
f2w = gpugen.f2w


def draw(vertices, offset=None, compare=7, cull=0):
    """`offset`: (front scale, front offset, back scale, back offset)."""
    return {'vertices': [tuple(v) for v in vertices], 'offset': offset,
            'compare': compare, 'cull': cull}


class Builder:
    """Collects the suite's vertices: vertex 0 the Resolve's rectangle."""

    def __init__(self, suite):
        self.suite = suite
        self.words = rb.RECT + [0, 0]
        self.ranges = {}
        suite.add_texture('zero', rb.fill((0.0, 0.0, 0.0, 0.0)))
        self.vs = gpugen.interpolating_vertex_shader(8)

    def triangles(self, vertices):
        key = tuple(vertices)
        if key not in self.ranges:
            first = len(self.words) // 8
            for x, y, z, w in vertices:
                self.words += [f2w(v) for v in (x, y, z, w, 0, 0, 0, 1.0)]
            self.ranges[key] = (pm4.PRIM_TRIANGLE_LIST, first, len(vertices))
        return self.ranges[key]

    def test(self, name, fmt, draws, samples=1, sample=0, clear=0x00000000,
             viewport=IDENTITY, resolve=None, dest=(6, 4)):
        """`resolve`: resolve_registers() of a Resolve other than the raw
        depth one, to a `dest` (texture format, bytes) the clear shares."""
        registers = rb.registers(samples={1: 0, 2: 1, 4: 2}[samples],
                                 depth_format=fmt,
                                 mode_cntl=gpugen.MODE_CNTL)
        registers += [('PA_CL_VTE_CNTL', [VTE_VIEWPORT]),
                      ('PA_CL_VPORT_XSCALE', [f2w(v) for v in viewport])]
        states = []
        for d in draws:
            mode = gpugen.MODE_CNTL | d['cull'] | (
                gpugen.MSAA_ENABLE if samples > 1 else 0)
            offset = d['offset'] or (0.0, 0.0, 0.0, 0.0)
            if d['offset']:
                mode |= POLY_OFFSET_ENABLE
            states.append(([
                ('RB_DEPTHCONTROL', [DEPTHCONTROL | d['compare'] << 4]),
                ('PA_SU_SC_MODE_CNTL', [mode]),
                ('PA_SU_POLY_OFFSET_FRONT_SCALE', [f2w(v) for v in offset])],
                self.triangles(d['vertices'])))
        passes = [(key, rb.fetch('zero') if i == 0 else {}, prim)
                  for i, (key, prim) in enumerate(states[:-1])]
        key, prim = states[-1]
        resolve = resolve or gpugen.resolve_registers(
            WIDTH, HEIGHT, 6, 4, command=0, source=4, sample_select=sample)
        rbdraw.test(self.suite, name, 'zero', registers, resolve, draw=prim,
                    passes=passes, after_passes=key, shader=self.vs,
                    pixel=rb.pixel_shader(), program_cntl=PROGRAM_CNTL,
                    clear=gpugen.resolve_registers(WIDTH, HEIGHT, *dest,
                                                   clear=(clear, 0)))

    def finish(self):
        self.suite.set_inputs(self.words, 8)


UNIT = 2.0 ** -28   # a sixteenth of a D24S8 step, the f32 grid below 1/16


def plane_vertices(xy, base, sx, sy, origin=(0, 0), shift=0):
    """Vertices at window `xy` with Z = (base + sx (x - ox) + sy (y - oy) +
    shift) * 2^-28, each exactly a float."""
    out = []
    for x, y in xy:
        n = base + sx * (x - origin[0]) + sy * (y - origin[1]) + shift
        if n != int(n) or not 0 < n < 1 << 24:
            raise ValueError(f'vertex ({x}, {y}): Z {n} * 2^-28 not exact')
        out.append((float(x), float(y), int(n) * UNIT, 1.0))
    return out


def scan(builder, name, fmt, xy, base, sx, sy, origin=(0, 0), shifts=16,
         **kw):
    """The plane drawn `shifts` times, its Z raised a further 2^-28 each
    time: where each pixel's word steps tells its depth to 2^-28."""
    for j in range(shifts):
        builder.test(f'{name} +{j}', fmt, [draw(plane_vertices(
            xy, base, sx, sy, origin, j))], **kw)


def pixel_quads(cases):
    """Vertices drawing pixel i (x + 64 y) of `cases` [(z, w)] as two
    triangles, flat at Z z and W w, their window X and Y times w."""
    out = []
    for i, (z, w) in enumerate(cases):
        x, y = i % WIDTH, i // WIDTH
        corners = [(x - 0.5, y - 0.5), (x + 0.5, y - 0.5),
                   (x - 0.5, y + 0.5), (x + 0.5, y + 0.5)]
        for k in (0, 1, 2, 2, 1, 3):
            cx, cy = corners[k]
            out.append((cx * w, cy * w, z, w))
    return out
