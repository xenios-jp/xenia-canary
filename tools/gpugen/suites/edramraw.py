"""Raw EDRAM read-back: whether a converting Resolve through a one-sample
k_8_8_8_8 color view (copy command 1, k_8_8_8_8 destination, exp bias 0, no
red/blue swap) copies every 32-bit EDRAM word to a texel unchanged, whatever
format and sample count last wrote it, so a title can dump the EDRAM.

The first test fills all 2048 tiles: a 1280x2048 k_32_FLOAT target at base
0 (16 tiles a row) whose pixel shader outputs 2^23 + x + 2048 y from
PsParamGen, so pixel (x, y) holds 0x4B000000 | y << 11 | x. Each region
test then writes known contents to tiles of its own at an EDRAM base (pitch
80, a tile a row at 1x), all of them first as k_8_8_8_8 bytes 0x33, 0x66,
0x99, 0xCC, then:
- 8_8_8_8 1x, 2x, 4x: texture 'bytes k' (80x32, every byte value in every
  channel) a sample at a time (PA_SC_AA_MASK 0x1111 << k);
- 16_16_16_16 1x: texture 'u16' (unorm16 values), 32_FLOAT 1x: 'f32' (random
  words, no denormal, infinity or NaN);
- D24S8 1x, 4x: oDepth = texture 'depth' X, depth ALWAYS with writes,
  stencil ALWAYS REPLACE 0x5A, then 0xC3 over x < 32, y < 16, RB_COLOR_MASK
  0 (the reinterp suite's depth writer);
- clear: a clearing Resolve of x < 40, y < 16 of color (k_8_8_8_8, tile
  base) and depth (D24S8, base + 1) over what the fill left.
and reads every tile it wrote through the one-sample view, 80 x 16 tiles
rows, to a 96-wide destination: 'convert' as above, 'raw' with copy command
0 (the bits, as the reinterp suite found). The span tests then read all 2048
tiles through the view at pitch 1280, convert and raw, in 4 Resolves of 512
tiles, and last convert in one Resolve of 1280x2048 (10 MiB).
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
import rbdraw  # noqa: E402
import xenos as x  # noqa: E402

f2w = gpugen.f2w
K = gpugen.C_FIRST_FREE
TILE_WIDTH, TILE_HEIGHT = 80, 16
TILES = 2048
SPAN_PITCH = 1280
SPAN_ROWS = TILES * TILE_WIDTH // SPAN_PITCH * TILE_HEIGHT  # 2048
CHUNK_ROWS = 512
WIDTH, HEIGHT = 80, 32  # the region writes
TEXTURE_PITCH = 96
DEST_PITCH = 96
K_8888 = 6
C_8888, C_16_16_16_16, C_32_FLOAT = 0, 5, 14
D24S8 = 0
STENCIL, Z, Z_WRITE = 1, 2, 4
ALWAYS, REPLACE = 7, 2
WRITE = (STENCIL | Z | Z_WRITE | (ALWAYS << 4) | (ALWAYS << 8) |
         (REPLACE << 14))
ODEPTH_CNTL = (rb.PROGRAM_CNTL & ~(0xF << 27)) | (3 << 27)
FILL = (0.2, 0.4, 0.6, 0.8)
DEPTH_CLEAR, COLOR_CLEAR = 0xABCDEF5A, 0xFF336699
# (name, base tile, tiles): regions apart, one at the top of the EDRAM.
REGIONS = {'8_8_8_8 1x': 0, '8_8_8_8 2x': 300, '8_8_8_8 4x': 700,
           '16_16_16_16 1x': 1100, 'D24S8 1x': 1500, 'D24S8 4x': 1800,
           'clear 1x': 1990, '32_FLOAT 1x': 2046}


def texture(texels):
    """An 80x32 k_32_32_32_32_FLOAT texture from 4-word texels, rows padded
    to 96 texels."""
    words = []
    for y in range(HEIGHT):
        for i in range(TEXTURE_PITCH):
            words += texels[y * WIDTH + i] if i < WIDTH else [0] * 4
    return words


def fetch(name):
    return {rb.SLOT: (gpugen.texture_constant(rb.F32_32_32_32, WIDTH, HEIGHT),
                      name)}


def bytes_texels(rng):
    """Every byte value in every channel at least 10 times, shuffled."""
    lanes = []
    for _ in range(4):
        values = [i & 255 for i in range(WIDTH * HEIGHT)]
        rng.shuffle(values)
        lanes.append(values)
    return [[f2w(lanes[c][i] / 255) for c in range(4)]
            for i in range(WIDTH * HEIGHT)]


def u16_texels(rng):
    return [[f2w(rng.randrange(65536) / 65535) for _ in range(4)]
            for _ in range(WIDTH * HEIGHT)]


def f32_texels(rng):
    def word():
        while True:
            w = rng.getrandbits(32)
            if 0 < (w >> 23) & 0xFF < 0xFF:
                return w
    return [[word(), 0, 0, 0] for _ in range(WIDTH * HEIGHT)]


def depth_texels(rng):
    out = []
    for i in range(WIDTH * HEIGHT):
        if i < 64:
            z = rb.SPECIALS[i]
        elif i < 1600:
            z = f2w((rng.randrange(0, 1 << 24) +
                     depth.OFFSETS[i % 8]) / 16777215)
        else:
            z = f2w(rng.uniform(1, 2) * 2.0 ** rng.randint(-24, 0))
        out.append([z, f2w(0.25), f2w(0.5), f2w(0.75)])
    return out


def position_shader():
    """oC0 = 2^23 + |PsParamGen.x| + 2048 |PsParamGen.y|, exact."""
    s = x.Shader()
    s.exec_([x.alu(vector_op=x.V_ADD, vector_dest=1, vector_mask=0x1,
                   src1=x.r(0, 'xxxx', absolute=True), src2=x.c(K, 'yyyy'))])
    s.exec_([x.alu(vector_op=x.V_MAD, vector_dest=1, vector_mask=0x1,
                   src1=x.r(0, 'yyyy', absolute=True), src2=x.c(K, 'zzzz'),
                   src3=x.r(1, 'xxxx'))])
    s.alloc(x.ALLOC_COLORS, 0)
    s.exec_([x.alu(vector_op=x.V_MAX, vector_dest=0, vector_mask=0xF,
                   export=True, src1=x.r(1, 'xxxx'), src2=x.r(1, 'xxxx'))],
            end=True)
    return s


def surface(base, fmt=C_8888, samples=0, pitch=TILE_WIDTH, rows=HEIGHT,
            depth_base=None):
    """Every register of a draw's render target, so nothing carries over:
    color `fmt` at `base`, a D24S8 buffer at `depth_base` (else `base`),
    depth and writes off."""
    return [('SQ_CONTEXT_MISC', [gpugen.SCREEN_XY]),
            ('RB_SURFACE_INFO', [pitch | (samples << 16)]),
            ('RB_COLOR_INFO', [base | (fmt << 16)]),
            ('RB_DEPTH_INFO', [(base if depth_base is None else depth_base) |
                               (D24S8 << 16)]),
            ('RB_COLOR_MASK', [0xF]),
            ('RB_DEPTHCONTROL', [0]),
            ('RB_STENCILREFMASK', [0x00FFFF00]),
            ('PA_SC_WINDOW_SCISSOR_BR', [(rows << 16) | pitch]),
            ('PA_SC_AA_CONFIG', [gpugen.AA_CONFIG[samples]]),
            ('PA_SC_AA_MASK', [0xFFFF]),
            ('PA_SU_SC_MODE_CNTL', [gpugen.MODE_CNTL |
                                    (gpugen.MSAA_ENABLE if samples else 0)])]


def build(suite):
    suite.raw = True
    rng = random.Random(0xED7A)
    for k in range(4):
        suite.add_texture(f'bytes {k}', texture(bytes_texels(rng)))
    suite.add_texture('u16', texture(u16_texels(rng)))
    suite.add_texture('f32', texture(f32_texels(rng)))
    suite.add_texture('depth', texture(depth_texels(rng)))
    suite.add_texture('fill', rb.fill(FILL))
    shapes = rbdraw.Shapes()
    write_rect = shapes.rectangles([(0, 0, WIDTH, HEIGHT)])
    stencil_rect = shapes.rectangles([(0, 0, 32, 16)])
    clear_rect = shapes.rectangles([(0, 0, 40, 16)])
    nothing = (pm4.PRIM_RECTANGLE_LIST, 0, 0)

    def read(base, rows, command, pitch=TILE_WIDTH, dest_pitch=DEST_PITCH):
        """The Resolve of `rows` rows of the one-sample k_8_8_8_8 view at
        `base`, `pitch` wide."""
        rect = shapes.rectangles([(0, 0, pitch, rows)])
        return gpugen.resolve_registers(
            pitch, rows, K_8888, 4, command=command, rect=2 * rect[1],
            dest=(dest_pitch, rows), registers=surface(base, pitch=pitch,
                                                       rows=rows))

    # Every tile: 0x4B000000 | y << 11 | x at pitch 1280.
    span = surface(0, C_32_FLOAT, pitch=SPAN_PITCH, rows=SPAN_ROWS)
    suite.test('span fill 32_FLOAT position', gpugen.resolve_vertex_shader(),
               constants={256 + K: [f2w(0.5), f2w(2.0 ** 23), f2w(2048.0),
                                    0]},
               pixel_shader=position_shader(), program_cntl=rb.PROGRAM_CNTL,
               registers=span,
               draw=shapes.rectangles([(0, 0, SPAN_PITCH, SPAN_ROWS)]),
               resolve=read(0, 32, 0, SPAN_PITCH, SPAN_PITCH))

    def region(name, tiles, writes, odepth=False, clear=None):
        """The region's fill then `writes` (registers, texture, draw), the
        last the test's own, read convert and raw."""
        base = REGIONS[name]
        rows = TILE_HEIGHT * tiles
        fill_rect = shapes.rectangles([(0, 0, TILE_WIDTH, rows)])
        for command, how in ((1, 'convert'), (0, 'raw')):
            if clear is None:
                passes = [(surface(base, rows=rows), rb.fetch('fill'),
                           fill_rect)] + list(writes)
                registers = []
            else:
                passes = []
                registers = surface(base, depth_base=base + 1)
            after, fetch_constants, draw = (passes.pop() if passes else
                                            ([], {}, nothing))
            suite.test(
                f'{name} {how}', gpugen.resolve_vertex_shader(),
                constants=rb.CONSTANTS,
                pixel_shader=rb.pixel_shader(depth=odepth),
                program_cntl=ODEPTH_CNTL if odepth else rb.PROGRAM_CNTL,
                registers=registers, passes=passes, after_passes=after,
                fetch_constants=fetch_constants, draw=draw,
                resolve=read(base, rows, command), clear=clear)

    def color(name, fmt, samples, tiles, textures):
        base = REGIONS[name]
        writes = [((surface(base, fmt, samples) if k == 0 else []) +
                   [('PA_SC_AA_MASK', [0x1111 << k if samples else 0xFFFF])],
                   fetch(texture_name), write_rect)
                  for k, texture_name in enumerate(textures)]
        region(name, tiles, writes)

    color('8_8_8_8 1x', C_8888, 0, 2, ['bytes 0'])
    color('8_8_8_8 2x', C_8888, 1, 4, ['bytes 0', 'bytes 1'])
    color('8_8_8_8 4x', C_8888, 2, 8, [f'bytes {k}' for k in range(4)])
    color('16_16_16_16 1x', C_16_16_16_16, 0, 4, ['u16'])
    color('32_FLOAT 1x', C_32_FLOAT, 0, 2, ['f32'])

    for name, samples, tiles in (('D24S8 1x', 0, 2), ('D24S8 4x', 2, 8)):
        base = REGIONS[name]
        state = surface(base, samples=samples) + [
            ('RB_COLOR_MASK', [0]), ('RB_DEPTHCONTROL', [WRITE]),
            ('RB_STENCILREFMASK', [0x00FFFF00 | 0x5A])]
        region(name, tiles, [
            (state, fetch('depth'), write_rect),
            ([('RB_STENCILREFMASK', [0x00FFFF00 | 0xC3])], {},
             stencil_rect)], odepth=True)

    region('clear 1x', 2, (), clear=gpugen.resolve_registers(
        40, 16, K_8888, 4, clear=(DEPTH_CLEAR, COLOR_CLEAR), clear_enable=3,
        rect=2 * clear_rect[1], dest=(64, 16)))

    # All 2048 tiles, 512 a Resolve, then in one.
    for k in range(TILES // (CHUNK_ROWS * SPAN_PITCH // TILE_WIDTH //
                             TILE_HEIGHT)):
        base = k * CHUNK_ROWS * SPAN_PITCH // TILE_WIDTH // TILE_HEIGHT
        for command, how in ((1, 'convert'), (0, 'raw')):
            suite.test(f'span chunk {k} {how}', gpugen.resolve_vertex_shader(),
                       pixel_shader=rb.pixel_shader(),
                       program_cntl=rb.PROGRAM_CNTL, draw=nothing,
                       resolve=read(base, CHUNK_ROWS, command, SPAN_PITCH,
                                    SPAN_PITCH))
    suite.test('span whole convert', gpugen.resolve_vertex_shader(),
               pixel_shader=rb.pixel_shader(), program_cntl=rb.PROGRAM_CNTL,
               draw=nothing,
               resolve=read(0, SPAN_ROWS, 1, SPAN_PITCH, SPAN_PITCH))
    suite.set_inputs(shapes.inputs, 2)
