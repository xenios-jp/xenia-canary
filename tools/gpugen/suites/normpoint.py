"""Point sampling at normalized coordinates. Each case point samples a
k_32_FLOAT texture 65, 720 or 1280 texels wide and 16 high, each texel its
x, at one coordinate (u, 0.46875) under one address mode for u (wrap, mirror
or clamp; v clamps): u just either side of where floor and truncation of its
fixed point texel index differ, at wrap and mirror, and 300 random u in
[-2, 2] a width at all three.

Each case is a 2x2 quad with its coordinate uniform over the quad: the pixel
shader fetches the coordinate from a float32 texture, then point samples the
case's texture at it, exporting the texel to a k_32_FLOAT target.
512 cases a test in order, a draw per run of cases of one width and mode.
"""

import math
import os
import random
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpugen  # noqa: E402
import rb  # noqa: E402
import rbdraw  # noqa: E402
import xenos as x  # noqa: E402

WIDTH, HEIGHT = rb.WIDTH, rb.HEIGHT
QUADS_X = WIDTH // 2
CASES_PER_TEST = QUADS_X * (HEIGHT // 2)
WIDTHS = (65, 720, 1280)
TEXTURE_HEIGHT = 16
V = 0.46875
WRAP, MIRROR, CLAMP = 0, 1, 2
SLOT = 2
# k_32_FLOAT targets, resolved as k_32_FLOAT.
RT_32_FLOAT, TEXTURE_32_FLOAT = 14, 36
f2w = gpugen.f2w


def bits(v):
    return struct.unpack('<I', struct.pack('<f', v))[0]


def value(b):
    return struct.unpack('<f', struct.pack('<I', b))[0]


def cases():
    """[(width, mode, u bits, v bits)], sorted."""
    def index(u, w, rounding):
        return math.floor(math.trunc(rounding(u * 2 ** 24) * w * 64 / 2 ** 24)
                          / 64) % w

    out = set()
    for width in WIDTHS:
        for k in range(width):
            center = bits(-(k + 1 / 64) / width)
            for delta in range(-4, 5):
                b = center + delta
                u = value(b)
                if index(u, width, math.floor) != index(u, width, math.trunc):
                    for mode in (WRAP, MIRROR):
                        out.add((width, mode, b, bits(V)))
        rng = random.Random(width)
        for _ in range(300):
            u = rng.uniform(-2, 2)
            for mode in (WRAP, MIRROR, CLAMP):
                out.add((width, mode, bits(u), bits(V)))
    return sorted(out)


def tiled(x_, y, pitch):
    """The byte offset of 32-bit element (x, y) of a tiled 2D texture `pitch`
    elements wide (xenia:src/xenia/gpu/texture_address.h, Tiled2D)."""
    pitch = (pitch + 31) & ~31
    macro = ((x_ >> 5) + (y >> 5) * (pitch >> 5)) << 9
    micro = ((x_ & 7) + ((y & 0xE) << 2)) << 2
    offset = macro + ((micro & ~0xF) << 1) + (micro & 0xF) + ((y & 1) << 4)
    return (((offset & ~0x1FF) << 3) + ((y & 16) << 7) +
            ((offset & 0x1C0) << 2) +
            (((((y & 8) >> 2) + (x_ >> 3)) & 3) << 6) + (offset & 0x3F))


def texture(width):
    """The case's texture, tiled: each texel its x. Its
    fetch constant packs mips, and a level 16 high in a wider texture packs
    16 rows down its tile (xenia:src/xenia/gpu/texture_util.cc,
    GetPackedMipOffset)."""
    pitch = (width + 31) & ~31
    out = [0] * (pitch * 32)
    for y in range(TEXTURE_HEIGHT):
        for x_ in range(width):
            out[tiled(x_, y + 16, width) // 4] = f2w(float(x_))
    return out


def fetch_constant(width, mode):
    """The case's fetch constant but for the base address: tiled, pitch in
    32s of texels, u at `mode` and v clamped, k_32_FLOAT 8-in-32, width x 16,
    the swizzle (x, 0, 0, 1), point filters, 2D."""
    return [2 | mode << 10 | CLAMP << 13 | ((width + 31) // 32) << 22 |
            1 << 31,
            TEXTURE_32_FLOAT | 2 << 6,
            (width - 1) | (TEXTURE_HEIGHT - 1) << 13,
            0x1640, 0, 0xA00]


def pixel_shader():
    """r2 = the case's coordinate, fetched at the pixel (as rb's pixel
    shader); r3 = the tfetch of it; oC0 = r3."""
    s = x.Shader()
    s.exec_([x.alu(vector_op=x.V_ADD, vector_dest=1, vector_mask=0x3,
                   src1=x.r(0, absolute=True), src2=x.c(rb.K, 'xxxx'))])
    s.exec_([x.tfetch(2, 1, 'xy', rb.SLOT, denormalized=True).serialized()])
    s.exec_([x.tfetch(3, 2, 'xyz', SLOT, dimension=1, aniso_filter=0x3F,
                      fetch_valid_only=False, sample_location=1).serialized()])
    s.alloc(x.ALLOC_COLORS, 0)
    s.exec_([x.alu(vector_op=x.V_MAX, vector_dest=0, vector_mask=0xF,
                   export=True, src1=x.r(3), src2=x.r(3))], end=True)
    return s


def tests():
    """[(name, [case index])]: the cases in order, 512 a test."""
    count = len(cases())
    return [(f'cases {i}-{min(i + CASES_PER_TEST, count) - 1}',
             list(range(i, min(i + CASES_PER_TEST, count))))
            for i in range(0, count, CASES_PER_TEST)]


def quad(q):
    """Quad q's pixel rectangle, in rows of 32 quads."""
    x0, y0 = 2 * (q % QUADS_X), 2 * (q // QUADS_X)
    return x0, y0, x0 + 2, y0 + 2


def build(suite):
    all_cases = cases()
    shapes = rbdraw.Shapes()
    for width in WIDTHS:
        suite.add_texture(f'{width} wide', texture(width))
    pixel = pixel_shader()
    state = rb.registers(RT_32_FLOAT)
    resolve = gpugen.resolve_registers(WIDTH, HEIGHT, TEXTURE_32_FLOAT, 4, 7)
    for name, indices in tests():
        coordinates = [0] * (4 * WIDTH * HEIGHT)
        runs = []
        for q, i in enumerate(indices):
            width, mode, u, v = all_cases[i]
            x0, y0, x1, y1 = quad(q)
            for y in range(y0, y1):
                for x_ in range(x0, x1):
                    p = 4 * (x_ + WIDTH * y)
                    coordinates[p:p + 4] = [u, v, f2w(0.0), f2w(1.0)]
            if runs and runs[-1][0] == (width, mode):
                runs[-1][1].append(quad(q))
            else:
                runs.append(((width, mode), [quad(q)]))
        suite.add_texture(f'{name} coordinates', coordinates)
        passes = [([], {SLOT: (fetch_constant(width, mode), f'{width} wide')},
                   shapes.rectangles(rects))
                  for (width, mode), rects in runs]
        _, fetch, draw = passes.pop()
        if passes:
            passes[0][1].update(rb.fetch(f'{name} coordinates'))
        else:
            fetch.update(rb.fetch(f'{name} coordinates'))
        suite.test(name, gpugen.resolve_vertex_shader(),
                   constants=rb.CONSTANTS, pixel_shader=pixel,
                   program_cntl=rb.PROGRAM_CNTL, registers=state,
                   passes=passes, fetch_constants=fetch, draw=draw,
                   resolve=resolve)
    suite.set_inputs(shapes.inputs, 2)
