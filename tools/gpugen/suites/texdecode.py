"""Texture decoding. Each case is a 32x32 texture of 8_8_8_8 texels or 4x4
blocks (DXT1, DXT2_3, DXT4_5, DXN, DXT3A, DXT5A, CTX1 and the AS_16_16_16_16
formats), unsigned, signed or gamma, point sampled at every texel or linear
sampled between texels.

Each case is drawn as a quad in clip space through a viewport over a 32x32
window, the texture coordinates interpolated and fetched with computed LOD,
its texture tiled under the case's fetch constant. The pixel shader fetches
all four components (the swizzle the identity) and exports them at the
pixel's element, x + 32y from PsParamGen, after the elements of the case's
test's earlier cases: 16 cases a test, a draw each, c8.x the draw's first
element.
"""

import os
import random
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gpugen  # noqa: E402
import pm4  # noqa: E402
import xenos as x  # noqa: E402

SIDE = 32
DATA_BYTES = 4096
BASE_DXT1, BASE_DXT3, BASE_DXT5, BASE_DXN, BASE_DXT5A, BASE_8888 = range(6)
UNSIGNED, SIGNED, GAMMA = 0, 1, 3
POINT, LINEAR = 0, 1
# TextureFormat values (xenia:src/xenia/gpu/xenos.h:532-574), the base layout
# they share, and the signs to read them with.
FORMATS = (
    ('8_8_8_8', 6, BASE_8888, (UNSIGNED, SIGNED, GAMMA)),
    ('DXT1', 18, BASE_DXT1, (UNSIGNED, GAMMA)),
    ('DXT2_3', 19, BASE_DXT3, (UNSIGNED, GAMMA)),
    ('DXT4_5', 20, BASE_DXT5, (UNSIGNED, GAMMA)),
    ('DXN', 49, BASE_DXN, (UNSIGNED, SIGNED)),
    ('8_8_8_8_AS_16_16_16_16', 50, BASE_8888, (UNSIGNED, SIGNED)),
    ('DXT1_AS_16_16_16_16', 51, BASE_DXT1, (UNSIGNED,)),
    ('DXT2_3_AS_16_16_16_16', 52, BASE_DXT3, (UNSIGNED,)),
    ('DXT4_5_AS_16_16_16_16', 53, BASE_DXT5, (UNSIGNED,)),
    ('DXT3A', 58, BASE_DXT1, (UNSIGNED, SIGNED)),
    ('DXT5A', 59, BASE_DXT5A, (UNSIGNED, SIGNED)),
    ('CTX1', 60, BASE_DXT1, (UNSIGNED, SIGNED)),
)
TEXTURES = 16
# Texel offsets for linear sampling: halfway, thirds, quarters and the
# smallest steps, across, down and both.
OFFSETS = ((0.5, 0), (0, 0.5), (0.5, 0.5), (0.25, 0), (1 / 3, 0),
           (0.125, 0.625), (1 / 256, 0), (1 / 512, 1 / 512), (255 / 256, 0),
           (0.75, 0.25))


def color_block(rng, edge):
    if edge is None:
        c0, c1 = rng.getrandbits(16), rng.getrandbits(16)
    else:
        c0, c1 = edge
    return [c0, c1, rng.getrandbits(16), rng.getrandbits(16)]


def alpha_block(rng, edge):
    """A DXT5 alpha block: two 8-bit endpoints, 16 3-bit indices."""
    if edge is None:
        edge = rng.getrandbits(8), rng.getrandbits(8)
    a0, a1 = edge
    return [a0 | a1 << 8] + [rng.getrandbits(16) for _ in range(3)]


COLOR_EDGES = [(0, 0), (0xFFFF, 0xFFFF), (0xFFFF, 0), (0, 0xFFFF),
               (0xF800, 0x07E0), (0x07E0, 0xF800), (0x001F, 0xF81F),
               (0x8410, 0x8410), (0x8410, 0x8411), (0x8411, 0x8410),
               (0x0821, 0x0020), (0xFFDF, 0xFFFF)]
ALPHA_EDGES = [(0, 0), (255, 255), (255, 0), (0, 255), (128, 127), (127, 128),
               (1, 0), (0, 1), (254, 255), (255, 254), (37, 200), (200, 37)]


def block(base, rng, n, crafted):
    """Block n's words; crafted textures cycle through the edges."""
    ce = COLOR_EDGES[n % len(COLOR_EDGES)] if crafted else None
    ae = ALPHA_EDGES[n % len(ALPHA_EDGES)] if crafted else None
    ae2 = ALPHA_EDGES[(n + 5) % len(ALPHA_EDGES)] if crafted else None
    if base == BASE_DXT1:
        return color_block(rng, ce)
    if base == BASE_DXT3:
        return [rng.getrandbits(16) for _ in range(4)] + color_block(rng, ce)
    if base == BASE_DXT5:
        return alpha_block(rng, ae) + color_block(rng, ce)
    if base == BASE_DXN:
        return alpha_block(rng, ae) + alpha_block(rng, ae2)
    return alpha_block(rng, ae)


def texture(base, rng, crafted):
    """The texture's bytes, rows of blocks (or texels) in order."""
    if base == BASE_8888:
        if crafted:
            # Every byte in every channel: texel i's channel c holds
            # (i + 64 c + 17 (i >> 8)) mod 256.
            return bytes((i + 64 * c + 17 * (i >> 8)) & 255
                         for i in range(SIDE * SIDE) for c in range(4))
        return bytes(rng.getrandbits(8) for _ in range(SIDE * SIDE * 4))
    words = []
    for n in range(SIDE * SIDE // 16):
        words += block(base, rng, n, crafted)
    return struct.pack(f'>{len(words)}H', *words)


def signs_for(sign):
    """2 bits a component; gamma covers the colour, not the alpha."""
    if sign == GAMMA:
        return GAMMA | GAMMA << 2 | GAMMA << 4
    return sign * 0x55


def case(name, fmt, base, sign, data, filt=POINT, offset=(0, 0)):
    assert len(data) <= DATA_BYTES
    return {'name': name, 'format': fmt, 'base': base,
            'signs': signs_for(sign), 'filter': filt,
            'offset': list(offset), 'data': data}


def cases():
    rng = random.Random(0x58524760)
    out = []
    for name, fmt, base, signs in FORMATS:
        for sign in signs:
            for t in range(TEXTURES):
                out.append(case(f'point-{name}-{sign}-{t}', fmt, base, sign,
                                texture(base, rng, t == 0)))
    for name, fmt, base, signs in FORMATS:
        if fmt not in (6, 18, 20, 59, 49):
            continue
        for sign in signs:
            data = texture(base, rng, False)
            for k, offset in enumerate(OFFSETS):
                out.append(case(f'linear-{name}-{sign}-{k}', fmt, base, sign,
                                data, LINEAR, offset))
    return out


def words(c):
    """The case as 1032 words."""
    head = struct.pack('>4I2f2I', c['format'], c['base'], c['signs'],
                       c['filter'], *c['offset'], 0, 0)
    data = head + c['data'] + bytes(DATA_BYTES - len(c['data']))
    return struct.unpack(f'>{len(data) // 4}I', data)


def tiled(x, y, pitch, log2_bytes):
    """The byte offset of element (x, y) of a tiled 2D texture `pitch`
    elements wide (xenia:src/xenia/gpu/texture_address.h, Tiled2D)."""
    pitch = (pitch + 31) & ~31
    macro = ((x >> 5) + (y >> 5) * (pitch >> 5)) << (log2_bytes + 7)
    micro = ((x & 7) + ((y & 0xE) << 2)) << log2_bytes
    offset = macro + ((micro & ~0xF) << 1) + (micro & 0xF) + ((y & 1) << 4)
    return (((offset & ~0x1FF) << 3) + ((y & 16) << 7) +
            ((offset & 0x1C0) << 2) +
            (((((y & 8) >> 2) + (x >> 3)) & 3) << 6) + (offset & 0x3F))


def element_bytes(base):
    return {BASE_8888: 4, BASE_DXT1: 8, BASE_DXT5A: 8}.get(base, 16)


def memory(c):
    """The texture tiled, in words: its one 32x32-element macro tile,
    elements texels or blocks."""
    size = element_bytes(c['base'])
    side = SIDE if c['base'] == BASE_8888 else SIDE // 4
    out = bytearray(SIDE * SIDE * size)
    for y in range(side):
        for x_ in range(side):
            at = tiled(x_, y, side, size.bit_length() - 1)
            start = (y * side + x_) * size
            out[at:at + size] = c['data'][start:start + size]
    return list(struct.unpack(f'>{len(out) // 4}I', out))


SLOT = 1
ENDIAN_8IN16 = 1
IDENTITY = 0 | 1 << 3 | 2 << 6 | 3 << 9


def fetch_constant(c):
    """The case's fetch constant but for the base address: tiled, pitch in
    32s of texels (blocks 128 texels wide), clamped, the case's signs and
    format, block formats 8-in-16, 32x32, 2D, the sampler's filters."""
    blocks = c['base'] != BASE_8888
    linear = c['filter'] == LINEAR
    return [2 | c['signs'] << 2 | 2 << 10 | 2 << 13 |
            (4 if blocks else 1) << 22 | 1 << 31,
            c['format'] | (ENDIAN_8IN16 if blocks else 0) << 6,
            (SIDE - 1) | (SIDE - 1) << 13,
            IDENTITY << 1 | (1 << 19 | 1 << 21 if linear else 0),
            3 if linear else 0,
            0xA00]


def f32(v):
    return struct.unpack('>f', struct.pack('>f', v))[0]


def quad(offset_u, offset_v):
    """The case's quad: over the target, pixel centres sampling
    texel centres moved by the offset; 8 dwords a vertex, position and (u, v,
    0, 1)."""
    low, high = f32(-0.5 / SIDE), f32((SIDE + 1.5) / SIDE)
    u0 = f32(low + f32(f32(offset_u) / SIDE))
    u1 = f32(high + f32(f32(offset_u) / SIDE))
    v0 = f32(low + f32(f32(offset_v) / SIDE))
    v1 = f32(high + f32(f32(offset_v) / SIDE))
    corners = ((-1, 1, u0, v0), (1, 1, u1, v0), (-1, -1, u0, v1),
               (-1, -1, u0, v1), (1, 1, u1, v0), (1, -1, u1, v1))
    return [gpugen.f2w(v) for px, py, u, v in corners
            for v in (px, py, 0.0, 1.0, u, v, 0.0, 1.0)]


K = gpugen.C_FIRST_FREE  # pixel c8: (first element, SIDE, 0, 0)
PARAM_GEN = 1  # PsParamGen's register, after interpolator 0
# 8 VS and 4 PS registers, vertex fetch, PsParamGen, one interpolator,
# Direct3D's PS export mode.
PROGRAM_CNTL = 7 | (3 << 8) | (1 << 16) | (1 << 18) | (2 << 27)
CASES_PER_TEST = 16
f2w = gpugen.f2w
# The draw state: PsParamGen's register and screen position, sample control
# (centers), interpolation (smooth, at centers), the 32x32 window, clip-space
# positions through a viewport of (-1, -1) to (33, 33) with clipping on as
# Direct3D's.
REGISTERS = [
    ('SQ_CONTEXT_MISC', [1 << 2 | PARAM_GEN << 8 | gpugen.SCREEN_XY]),
    ('SQ_INTERPOLATOR_CNTL', [0xFFFFFFFF]),
    ('PA_SC_WINDOW_SCISSOR_BR', [SIDE << 16 | SIDE]),
    ('RB_COLOR_MASK', [0xF]),
    ('PA_CL_CLIP_CNTL', [1 << 19, 0x10000, 0x43F]),
    ('PA_CL_VPORT_XSCALE', [f2w(v) for v in (17.0, 16.0, -17.0, 16.0, 1.0,
                                             0.0)])]


def pixel_shader():
    """r2 = a tfetch of r0's interpolated coordinates (fetch
    constant's filters, computed LOD, at centers), exported at element
    floor(|PsParamGen.x|) + 32 floor(|PsParamGen.y|) + c8.x."""
    fetch = x.tfetch(2, 0, 'xyz', SLOT, dimension=1, aniso_filter=0x3F,
                     fetch_valid_only=False, sample_location=1)
    index = [x.alu(vector_op=x.V_FLOOR, vector_dest=3, vector_mask=0x3,
                   src1=x.r(PARAM_GEN, absolute=True),
                   src2=x.r(PARAM_GEN, absolute=True)),
             x.alu(vector_op=x.V_MAD, vector_dest=3, vector_mask=0x1,
                   src1=x.r(3, 'y'), src2=x.c(K, 'y'), src3=x.r(3, 'x')),
             x.alu(vector_op=x.V_ADD, vector_dest=3, vector_mask=0x1,
                   src1=x.r(3, 'x'), src2=x.c(K, 'x'))]
    return gpugen.memexport_pixel_shader(index, [[gpugen.export_copy(2)]],
                                         x.r(3, 'x'), fetches=[fetch])


def tests():
    """[(name, [case index])]: the cases in order, 16 a test."""
    count = len(cases())
    return [(f'cases {i}-{min(i + CASES_PER_TEST, count) - 1}',
             list(range(i, min(i + CASES_PER_TEST, count))))
            for i in range(0, count, CASES_PER_TEST)]


def build(suite):
    all_cases = cases()
    inputs, draws = [], {}
    for c in all_cases:
        offset = tuple(c['offset'])
        if offset not in draws:
            draws[offset] = (pm4.PRIM_TRIANGLE_LIST, len(inputs) // 8, 6)
            inputs += quad(*offset)
    suite.set_inputs(inputs, 8)
    suite.export_slots = CASES_PER_TEST * SIDE * SIDE
    textures = {}
    for c in all_cases:
        if c['data'] not in textures:
            textures[c['data']] = c['name']
            suite.add_texture(c['name'], memory(c))
    shader = gpugen.interpolating_vertex_shader(8)
    pixel = pixel_shader()
    for name, indices in tests():
        passes = []
        for k, i in enumerate(indices):
            c = all_cases[i]
            passes.append((
                [('SHADER_CONSTANT_264_X',
                  [f2w(float(k * SIDE * SIDE)), f2w(float(SIDE)), 0, 0])],
                {SLOT: (fetch_constant(c), textures[c['data']])},
                draws[tuple(c['offset'])]))
        last, fetch, draw = passes.pop()
        suite.test(name, shader, constants={256 + K: last[0][1]},
                   pixel_shader=pixel, program_cntl=PROGRAM_CNTL,
                   registers=REGISTERS, passes=passes, after_passes=last,
                   fetch_constants=fetch, draw=draw)
