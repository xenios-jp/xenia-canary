"""EDRAM reinterpretation: what one surface format reads from the EDRAM
bits another wrote, as titles reuse EDRAM tiles across formats (the pairs
traces of titles reinterpret), in parts of the size the ring takes:
1 at 1x, 2 at 2x and 4x, 3 where samples sit (layout()).

Every test first fills the tiles at EDRAM base 0 that either surface
reaches (80 pixels wide, 32 rows, at the larger sample count) with k_8_8_8_8
bytes 0x33, 0x66, 0x99, 0xCC, then writes the rb suite's 64x32 surface there
in one format, then, with nothing cleared, reads the same tiles as another:
- a color writer: the rb suite's draw of 'ramp' (its pixel shader's oC0 =
  the texel) into a render target of the format, the depth test off;
- a depth writer: the depth suite's draw of its 'depth' texture (oDepth =
  the texel's X) into a depth buffer of the format, depth ALWAYS with
  writes, stencil ALWAYS REPLACE 0x5A, then 0xC3 over x < 32, y < 16, with
  RB_COLOR_MASK 0.
The reader is a Resolve under the other format's RB_COLOR_INFO or
RB_DEPTH_INFO (and RB_SURFACE_INFO's samples): 'raw' copy command 0 of the
color target to a texture of its pixel size, or of the depth buffer as
Direct3D's (raw to k_8_8_8_8), of sample k at 2x and 4x; or a converting one
of a color target to its resolve format (the rb suite's); or 'depth test': a
draw at depth 0.5 (oDepth) with LESS and writes, stencil ALWAYS zfail
INCR_CLAMP, then the raw depth Resolve.
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import depth  # noqa: E402
import gpugen  # noqa: E402
import rb  # noqa: E402
import rbdraw  # noqa: E402

WIDTH, HEIGHT = rb.WIDTH, rb.HEIGHT
D24S8, D24FS8 = 0, 1
FORMATS = {f[1]: f for f in rb.FORMATS}  # ColorRenderTargetFormat -> entry
STENCIL, Z, Z_WRITE = 1, 2, 4
LESS, ALWAYS = 1, 7
REPLACE, INCR_CLAMP = 2, 3
WRITE = (STENCIL | Z | Z_WRITE | (ALWAYS << 4) | (ALWAYS << 8) |
         (REPLACE << 14))
TEST = STENCIL | Z | Z_WRITE | (LESS << 4) | (ALWAYS << 8) | (INCR_CLAMP << 17)
# What every test first writes over whole tiles, as k_8_8_8_8: bytes 0x33,
# 0x66, 0x99, 0xCC.
FILL = (0.2, 0.4, 0.6, 0.8)
TILE_WIDTH = 80
# Direct3D's PS export mode with oDepth.
ODEPTH_CNTL = (rb.PROGRAM_CNTL & ~(0xF << 27)) | (3 << 27)
# The pairs titles reinterpret, (writer, reader, samples): 'c' color
# formats, 'd' depth formats.
PAIRS = [
    (('d', 1), ('c', 1), 2), (('c', 2), ('d', 1), 2), (('c', 1), ('c', 2), 2),
    (('c', 0), ('d', 1), 2), (('c', 3), ('d', 1), 1), (('d', 1), ('c', 3), 1),
    (('c', 10), ('d', 1), 2), (('c', 1), ('c', 10), 2),
    (('c', 0), ('d', 0), 2), (('d', 0), ('c', 0), 2), (('d', 1), ('c', 0), 1),
    (('c', 0), ('d', 1), 1), (('c', 12), ('d', 1), 1),
    (('d', 1), ('c', 12), 1), (('d', 0), ('c', 2), 2),
    (('d', 1), ('c', 0), 2), (('d', 0), ('c', 1), 2),
    (('c', 10), ('c', 0), 2), (('c', 12), ('c', 10), 0),
    (('c', 1), ('d', 1), 2), (('d', 1), ('c', 2), 2),
    (('c', 0), ('c', 10), 2), (('c', 3), ('c', 12), 0),
    (('c', 0), ('c', 7), 0), (('c', 7), ('c', 1), 0),
    (('c', 2), ('c', 10), 0), (('d', 1), ('c', 3), 2),
    (('c', 0), ('d', 0), 0), (('c', 0), ('d', 1), 0), (('d', 1), ('c', 0), 0),
]


def name_of(surface):
    kind, fmt = surface
    if kind == 'd':
        return ('D24S8', 'D24FS8')[fmt]
    return FORMATS[fmt][0]


def view(surface, samples):
    """The registers making `surface` the one at EDRAM base 0."""
    kind, fmt = surface
    info = ('RB_DEPTH_INFO' if kind == 'd' else 'RB_COLOR_INFO', [fmt << 16])
    return [info, ('RB_SURFACE_INFO', [80 | (samples << 16)]),
            ('PA_SC_AA_CONFIG', [gpugen.AA_CONFIG[samples]])]


def unique(sample):
    """A 32_FLOAT texture of a word for every pixel and sample: 1 + (i +
    2048 sample) / 8192 for texel i."""
    return [gpugen.f2w(v) for i in range(WIDTH * HEIGHT)
            for v in (1 + (i + 2048 * sample) / 8192, 0.0, 0.0, 0.0)]


def wide(sample):
    """A 32_32_32_32_FLOAT texture 80 texels wide, so 96 a row (pitches are
    whole 32s), its red and green 1 and 2 + (i + 3072 sample) / 16384 for
    texel i = x + 96 y."""
    return [gpugen.f2w(v) for i in range(96 * HEIGHT)
            for n in [(i + 96 * HEIGHT * sample) / 16384]
            for v in (1 + n, 2 + n, 0.0, 0.0)]


def layout(suite, rects, fill_rect):
    """Where each sample of each pixel sits: a 32_FLOAT target written a
    sample at a time (PA_SC_AA_MASK 0x1111 << k, sample k at each pixel of
    the 2x2 quad the mask's nibbles are for; texture 'sample k', a word of
    its own for every pixel and sample), then raw reads of every tile it
    covers through a one-sample color view of the same pitch (80 pixels a
    row, 16 rows a tile), of each sample through its own view, and of each
    through the D24S8 view; a one-sample target read as 2x and 4x; at 4x a
    mask of a sample for each quad position, 0x8421; and a 64bpp target 80
    pixels wide (texture 'wide k') at each sample count through a 32bpp
    view."""
    for k in range(4):
        suite.add_texture(f'sample {k}', unique(k))
    for samples in (2, 1, 0):
        count = 1 << samples
        # Every tile a 4x view of the target spans.
        fill = rb.registers(0, 2, mode_cntl=gpugen.MODE_CNTL) + [
            ('PA_SC_WINDOW_SCISSOR_BR', [(HEIGHT << 16) | TILE_WIDTH]),
            ('RB_DEPTHCONTROL', [0])]
        state = rb.registers(14, samples, mode_cntl=gpugen.MODE_CNTL)
        masks = [0x1111 << k for k in range(count)]

        def test(name, resolve, masks=masks):
            passes = [([], rb.fetch('fill'), fill_rect)] + [
                ((state if k == 0 else []) + [('PA_SC_AA_MASK', [mask])],
                 rb.fetch(f'sample {k}'), rb.RECT_DRAW)
                for k, mask in enumerate(masks)]
            after, fetch, draw = passes.pop()
            suite.test(name, gpugen.resolve_vertex_shader(),
                       constants=rb.CONSTANTS, pixel_shader=rb.pixel_shader(),
                       program_cntl=rb.PROGRAM_CNTL, registers=fill,
                       passes=passes, after_passes=after,
                       fetch_constants=fetch, draw=draw, resolve=resolve)
        if samples == 0:
            # A one-sample target through 2x and 4x views.
            for read in (1, 2):
                for k in range(1 << read):
                    test(f'32_FLOAT 1x read {1 << read}x sample {k}',
                         gpugen.resolve_registers(
                             WIDTH, HEIGHT, 6, 4, command=0, sample_select=k,
                             registers=view(('c', 14), read)))
            continue
        # Every tile through a one-sample view: 80 pixels a row, to a
        # destination 96 wide (one not a multiple of 32 tiles otherwise).
        rows = HEIGHT * count
        whole = rects.rectangles([(0, 0, TILE_WIDTH, rows)])
        test(f'32_FLOAT {count}x read 1x', gpugen.resolve_registers(
            TILE_WIDTH, rows, 6, 4, command=0, rect=2 * whole[1],
            dest=(96, rows),
            registers=view(('c', 14), 0) + [
                ('PA_SC_WINDOW_SCISSOR_BR', [(rows << 16) | TILE_WIDTH])]))
        for k in range(count):
            test(f'32_FLOAT {count}x sample {k}', gpugen.resolve_registers(
                WIDTH, HEIGHT, 6, 4, command=0, sample_select=k))
        if samples == 2:
            for k in range(count):
                test(f'32_FLOAT 4x mask 0x8421 sample {k}',
                     gpugen.resolve_registers(WIDTH, HEIGHT, 6, 4, command=0,
                                              sample_select=k),
                     masks=[0x8421])
        for k in range(count):
            test(f'32_FLOAT {count}x as D24S8 sample {k}',
                 gpugen.resolve_registers(
                     WIDTH, HEIGHT, 6, 4, command=0, source=4,
                     sample_select=k, registers=view(('d', 0), samples)))

    # 64bpp: a 32_32_FLOAT target 80 pixels wide, a sample at a time, its
    # every tile through a one-sample 32bpp view.
    for k in range(4):
        suite.add_texture(f'wide {k}', wide(k))
    scissor = ('PA_SC_WINDOW_SCISSOR_BR', [(HEIGHT << 16) | TILE_WIDTH])
    fill = rb.registers(15, 2, mode_cntl=gpugen.MODE_CNTL) + [
        scissor, ('RB_DEPTHCONTROL', [0])]
    for samples in (0, 1, 2):
        count = 1 << samples
        state = rb.registers(15, samples, mode_cntl=gpugen.MODE_CNTL) + [
            scissor]
        passes = [([], rb.fetch('fill'), fill_rect)] + [
            ((state if k == 0 else []) + [('PA_SC_AA_MASK', [0x1111 << k])],
             {rb.SLOT: (gpugen.texture_constant(rb.F32_32_32_32, TILE_WIDTH,
                                                HEIGHT), f'wide {k}')},
             fill_rect) for k in range(count)]
        after, fetch, draw = passes.pop()
        rows = 2 * HEIGHT * count
        whole = rects.rectangles([(0, 0, TILE_WIDTH, rows)])
        suite.test(
            f'32_32_FLOAT {count}x read 32bpp 1x',
            gpugen.resolve_vertex_shader(), constants=rb.CONSTANTS,
            pixel_shader=rb.pixel_shader(), program_cntl=rb.PROGRAM_CNTL,
            registers=fill, passes=passes, after_passes=after,
            fetch_constants=fetch, draw=draw,
            resolve=gpugen.resolve_registers(
                TILE_WIDTH, rows, 6, 4, command=0, rect=2 * whole[1],
                dest=(96, rows), registers=view(('c', 14), 0) + [
                    ('PA_SC_WINDOW_SCISSOR_BR', [(rows << 16) | TILE_WIDTH])]))


def build(suite, part):
    rng = random.Random(0x4B0)
    suite.add_texture('ramp', rb.ramp_texture(rng))
    suite.add_texture('depth', depth.depth_texture(random.Random(0xD3)))
    suite.add_texture('half', rb.fill((0.5, 0.25, 0.5, 0.75)))
    suite.add_texture('fill', rb.fill(FILL))
    rects = rbdraw.Shapes()
    stencil_rect = rects.rectangles([(0, 0, 32, 16)])
    fill_rect = rects.rectangles([(0, 0, TILE_WIDTH, HEIGHT)])
    suite.set_inputs(rects.inputs, 2)

    def write(writer, samples):
        """The writer's draws as passes, and its pixel shader."""
        kind, fmt = writer
        if kind == 'c':
            state = rb.registers(fmt, samples, mode_cntl=gpugen.MODE_CNTL) + [
                ('RB_DEPTHCONTROL', [0])]
            return [(state, rb.fetch('ramp'), rb.RECT_DRAW)], False
        state = rb.registers(0, samples, mode_cntl=gpugen.MODE_CNTL) + [
            ('RB_DEPTH_INFO', [fmt << 16]), ('RB_COLOR_MASK', [0]),
            ('RB_DEPTHCONTROL', [WRITE]),
            ('RB_STENCILREFMASK', [0x00FFFF00 | 0x5A])]
        return [(state, rb.fetch('depth'), rb.RECT_DRAW),
                ([('RB_STENCILREFMASK', [0x00FFFF00 | 0xC3])], {},
                 stencil_rect)], True

    def test(name, writer, reader, samples, how, sample=0,
             read_samples=None):
        """`how`: 'raw', 'convert' or 'depth test'."""
        if read_samples is None:
            read_samples = samples
        # Every tile either view reaches filled first, so nothing an earlier
        # title or test left shows.
        fill_samples = max(samples, read_samples)
        fill = rb.registers(0, fill_samples, mode_cntl=gpugen.MODE_CNTL) + [
            ('PA_SC_WINDOW_SCISSOR_BR', [(HEIGHT << 16) | TILE_WIDTH]),
            ('RB_DEPTHCONTROL', [0])]
        passes, odepth = write(writer, samples)
        passes = [([], rb.fetch('fill'), fill_rect)] + passes
        kind, fmt = reader
        registers = view(reader, read_samples)
        if how == 'depth test':
            odepth = True
            after, fetch, draw = registers + [
                ('RB_COLOR_MASK', [0]), ('RB_DEPTHCONTROL', [TEST]),
                ('RB_STENCILREFMASK', [0x00FFFF00])], rb.fetch('half'), \
                rb.RECT_DRAW
            resolve = gpugen.resolve_registers(
                WIDTH, HEIGHT, 6, 4, command=0, source=4,
                sample_select=sample)
        else:
            # The writer's last draw is the test's own.
            after, fetch, draw = passes.pop()
            if kind == 'd':
                resolve = gpugen.resolve_registers(
                    WIDTH, HEIGHT, 6, 4, command=0, source=4,
                    sample_select=sample, registers=registers)
            else:
                _, _, bpp, dest_format, number = FORMATS[fmt]
                if how == 'raw':
                    resolve = gpugen.resolve_registers(
                        WIDTH, HEIGHT, 6 if bpp == 4 else 26, bpp, command=0,
                        sample_select=sample, registers=registers)
                else:
                    resolve = gpugen.resolve_registers(
                        WIDTH, HEIGHT, dest_format,
                        rb.TEXEL_BYTES[dest_format], number,
                        sample_select=sample, registers=registers)
        suite.test(name, gpugen.resolve_vertex_shader(),
                   constants=rb.CONSTANTS,
                   pixel_shader=rb.pixel_shader(depth=odepth),
                   program_cntl=ODEPTH_CNTL if odepth else rb.PROGRAM_CNTL,
                   registers=fill, passes=passes, after_passes=after,
                   fetch_constants=fetch, draw=draw, resolve=resolve)

    def label(writer, reader, how, samples, sample=None, read_samples=None):
        text = f'{name_of(writer)} as {name_of(reader)} {how}'
        if read_samples is not None and read_samples != samples:
            text += f' {1 << samples}x read {1 << read_samples}x'
        elif samples:
            text += f' {1 << samples}x'
        if sample is not None:
            text += f' sample {sample}'
        return text

    if part == 3:
        layout(suite, rects, fill_rect)
        suite.set_inputs(rects.inputs, 2)
        return
    colors = [f[1] for f in rb.FORMATS]
    if part == 1:
        # The bits every format leaves, through a 32-bit color view (and the
        # 64-bit one for 64-bit formats) and the D24S8 view.
        writers = [('c', f) for f in colors] + [('d', D24S8), ('d', D24FS8)]
        for writer in writers:
            raw = [('c', 0)]
            if writer[0] == 'c' and FORMATS[writer[1]][2] == 8:
                raw.append(('c', 5))
            for reader in raw + [('d', D24S8)]:
                test(label(writer, reader, 'raw', 0), writer, reader, 0,
                     'raw')
        # Whether the view's format changes a raw copy.
        for fmt in colors:
            if fmt not in (0, 5):
                test(label(('c', 0), ('c', fmt), 'raw', 0), ('c', 0),
                     ('c', fmt), 0, 'raw')
        for writer in (('c', 0), ('d', D24S8)):
            test(label(writer, ('d', D24FS8), 'raw', 0), writer,
                 ('d', D24FS8), 0, 'raw')
        test(label(('d', D24FS8), ('d', D24FS8), 'raw', 0), ('d', D24FS8),
             ('d', D24FS8), 0, 'raw')
        # The pairs at 1x, and every 32-bit color format's bits as
        # D24FS8 in a depth test.
        pairs = [(w, r) for w, r, s in PAIRS if s == 0]
        for fmt in (0, 1, 2, 3, 10, 12, 14):
            if (('c', fmt), ('d', 1)) not in pairs:
                pairs.append((('c', fmt), ('d', 1)))
        for writer, reader in pairs:
            how = 'depth test' if reader[0] == 'd' else 'convert'
            test(label(writer, reader, how, 0), writer, reader, 0, how)
        return
    # Multisampled: the bits of each sample through the 32-bit color and
    # depth views.
    for samples, writers in ((2, [('c', 0), ('c', 1), ('c', 2), ('c', 3),
                                  ('c', 10), ('c', 12), ('d', 0), ('d', 1)]),
                             (1, [('c', 0), ('c', 3), ('c', 12), ('d', 1)])):
        for writer in writers:
            for reader in (('c', 0), ('d', D24S8)):
                for sample in range(1 << samples):
                    test(label(writer, reader, 'raw', samples, sample),
                         writer, reader, samples, 'raw', sample)
    # The pairs multisampled.
    for writer, reader, samples in PAIRS:
        if samples:
            how = 'depth test' if reader[0] == 'd' else 'convert'
            test(label(writer, reader, how, samples), writer, reader,
                 samples, how)
    # Written at one sample count, read at another.
    for writer, reader, samples, read_samples in (
            (('c', 0), ('c', 0), 0, 2), (('c', 0), ('c', 0), 2, 0),
            (('d', 1), ('d', 0), 2, 0), (('d', 0), ('d', 1), 0, 2),
            (('c', 0), ('d', 0), 0, 2), (('d', 0), ('c', 12), 2, 0)):
        test(label(writer, reader, 'raw', samples, None, read_samples),
             writer, reader, samples, 'raw', read_samples=read_samples)
