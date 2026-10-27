"""The output merger, in three case sets (main, mask and negative): the alpha
test against RB_ALPHA_REF, alpha to mask at its dither offsets, one to four
color targets of k_8_8_8_8 and k_2_10_10_10 each blended and masked its own
way, PA_SC_AA_MASK, and depth written under a polygon offset, at 1, 2 and 4
samples.

Each case is a 2x2 quad. Cases with the same targets (and depth clear, when
they test depth) share tests of up to 512 quads in a 64x32 window, grouped by
state. The first test of each draws them: each target's clear, the word a
clearing Resolve writes, stored raw by a draw of its bytes (byte + 1/4) / 255
with the target's format k_8_8_8_8 (as the color7e3 suite stores its words),
then the cases' quads, a draw of each run of quads alike, in clip space through
a viewport over the window (two triangles at the case's z). Their pixel shader
exports oC0..oC3 with MAX, its colors fetched from a 64x32 k_32_32_32_32_FLOAT
texture per target. Depth cases clear depth with the clearing Resolve of their
test. Each target's samples and their average, then depth (sample 0, as
k_8_8_8_8), are then resolved, a test each, the tests after the first drawing
nothing. The tests fill three suites (merger1.py, merger2.py, merger3.py), each
as much as the ring holds.
"""

import os
import random
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpugen  # noqa: E402
import pm4  # noqa: E402
import rb  # noqa: E402
import rbdraw  # noqa: E402
import xenos as x  # noqa: E402

NEVER, LESS, EQUAL, LEQUAL, GREATER, NOTEQUAL, GEQUAL, ALWAYS = range(8)
T8888, T2101010 = 0, 2
ADD, SUBTRACT, MIN, MAX, REVSUB = range(5)
ZERO, ONE = 0, 1
SRC_COLOR, ONE_MINUS_SRC_COLOR, SRC_ALPHA, ONE_MINUS_SRC_ALPHA = 4, 5, 6, 7
DST_COLOR, ONE_MINUS_DST_COLOR, DST_ALPHA, ONE_MINUS_DST_ALPHA = 8, 9, 10, 11
FLAG_DEPTH = 1
# RB_COLORCONTROL (xenia:src/xenia/gpu/registers.h:682-731); libxenon's
# offsets.
TEST_ENABLE, ALPHA_TO_MASK = 1 << 3, 1 << 4
DITHER, SYMMETRIC = 0x87, 0xAA
AA_MASK_ALL = 0xFFFF


def bits(f):
    return struct.unpack('>I', struct.pack('>f', f))[0]


def from_bits(b):
    return struct.unpack('>f', struct.pack('>I', b))[0]


def ulp(f, n):
    return from_bits(bits(f) + n)


INF, NAN, NEG_NAN = 0x7F800000, 0x7FC00000, 0xFFC00000
# Float bit patterns, so NaNs and signed zeros go as they are.
ALPHAS = [bits(f) for f in (
    0.0, -0.0, 2.0 ** -24, 0.5 / 255, 1 / 255, 0.25, ulp(0.5, -1), 0.5,
    ulp(0.5, 1), 0.499, 0.501, 127 / 255, 128 / 255, 0.75, ulp(1.0, -1),
    1.0, ulp(1.0, 1), 2.0, -0.25)] + [INF, INF | 1 << 31, NAN, NEG_NAN, 1]
REFS = [bits(f) for f in (
    0.0, -0.0, 0.5, ulp(0.5, 1), 0.501, 127 / 255, 128 / 255, ulp(1.0, -1),
    1.0, 2.0, -1.0)] + [INF, NAN, 1]
NEAR_THIRDS = [bits(f) for f in (
    0.3, 1 / 3, ulp(1 / 3, 1), 0.34, 0.5, 0.6, 2 / 3, ulp(2 / 3, -1), 0.7)]
OFFSETS = (SYMMETRIC, DITHER, 0x00, 0xFF, 0x1B, 0xE4)
COVER_ALPHAS = [bits(k / 32) for k in range(33)] + [
    bits(-0.5), bits(1.5), NAN, NEG_NAN, INF, bits(2.0 ** -24)]


def control(func=ALWAYS, test=False, cover=False, offsets=DITHER):
    return (func | (TEST_ENABLE if test else 0) |
            (ALPHA_TO_MASK if cover else 0) | offsets << 24)


def blend(op=ADD, source=ONE, dest=ZERO, alpha_op=None, alpha_source=None,
          alpha_dest=None, mask=15):
    return [op, source, dest, op if alpha_op is None else alpha_op,
            source if alpha_source is None else alpha_source,
            dest if alpha_dest is None else alpha_dest, mask]


NO_BLEND = blend()


def case(group, name, samples=1, formats=(T8888,), color_control=None,
         alpha_ref=0, flags=0, depth_compare=LESS, aa_mask=AA_MASK_ALL,
         z=0.5, bias=0.0, slope=0.0, depth_clear=1.0, blends=None,
         sources=None, clears=None):
    count = len(formats)
    blends = blends or [NO_BLEND] * count
    sources = sources or [[1.0, 0.5, 0.25, 1.0]] * count
    clears = clears or [[0.0, 1.0, 0.0, 0.0]] * count
    assert len(blends) == len(sources) == len(clears) == count
    return {'group': group, 'name': name, 'samples': samples,
            'formats': list(formats),
            'color_control': (control() if color_control is None
                              else color_control),
            'alpha_ref': alpha_ref, 'flags': flags,
            'depth_compare': depth_compare, 'aa_mask': aa_mask, 'z': z,
            'bias': bias, 'slope': slope, 'depth_clear': depth_clear,
            'blends': blends, 'sources': sources, 'clears': clears}


def with_alpha(source, alpha_bits):
    """A source whose alpha is a float bit pattern."""
    return source[:3] + [{'bits': alpha_bits}]


def word(f):
    if isinstance(f, dict):
        return f['bits']
    return bits(f)


def test_cases():
    out = []
    base = [1.0, 0.5, 0.25]
    for func in range(8):
        for a in ALPHAS:
            for r in REFS:
                out.append(case('test', f'test-{func}-{a:08X}-{r:08X}',
                                color_control=control(func, True),
                                alpha_ref=r, sources=[with_alpha(base, a)]))
    for func in (LESS, EQUAL, GREATER):
        for a in NEAR_THIRDS:
            for r in NEAR_THIRDS:
                out.append(case('test-2101010',
                                f'test10-{func}-{a:08X}-{r:08X}',
                                formats=(T2101010,),
                                color_control=control(func, True),
                                alpha_ref=r, sources=[with_alpha(base, a)]))
    return out


def cover_cases():
    out = []
    base = [0.75, 0.25, 1.0]
    for samples in (1, 2, 4):
        for offsets in OFFSETS:
            for a in COVER_ALPHAS:
                out.append(case(
                    'cover', f'cover-{samples}-{offsets:02X}-{a:08X}',
                    samples=samples,
                    color_control=control(cover=True, offsets=offsets),
                    sources=[with_alpha(base, a)]))
    for offsets in (SYMMETRIC, DITHER):
        for func in (LESS, GREATER):
            for a in COVER_ALPHAS:
                out.append(case(
                    'cover-test', f'cover-test-{offsets:02X}-{func}-{a:08X}',
                    samples=4,
                    color_control=control(func, True, True, offsets),
                    alpha_ref=bits(0.5), sources=[with_alpha(base, a)]))
    for samples, masks in ((2, (0x1, 0x2)), (4, (0x3, 0x5, 0xA, 0xE))):
        for mask in masks:
            for k in range(9):
                a = bits(k / 8)
                out.append(case(
                    'cover-mask', f'cover-mask-{samples}-{mask:X}-{a:08X}',
                    samples=samples, aa_mask=mask,
                    color_control=control(cover=True, offsets=SYMMETRIC),
                    sources=[with_alpha(base, a)]))
    return out


BLENDS = [
    NO_BLEND,
    blend(ADD, SRC_ALPHA, ONE_MINUS_SRC_ALPHA),
    blend(ADD, ONE, ONE),
    blend(REVSUB, ONE, ONE),
    blend(SUBTRACT, SRC_COLOR, DST_COLOR),
    blend(MAX, ONE, ONE, alpha_op=MIN),
    blend(ADD, DST_ALPHA, ONE_MINUS_DST_ALPHA, alpha_source=ZERO,
          alpha_dest=ONE),
    blend(ADD, ONE_MINUS_DST_COLOR, SRC_ALPHA),
]


def mrt_cases(rng):
    out = []

    def value():
        return rng.randrange(65) / 64

    for n in range(160):
        samples = (1, 2, 4)[n % 3]
        count = 2 + n % 3
        formats = [rng.choice((T8888, T2101010)) for _ in range(count)]
        blends = []
        for _ in range(count):
            b = list(rng.choice(BLENDS))
            b[6] = rng.choice((15, 15, 0, 1, 6, 8, 9, 14))
            blends.append(b)
        sources = [[value() for _ in range(4)] for _ in range(count)]
        clears = [[value() for _ in range(4)] for _ in range(count)]
        mode = n // 3 % 3
        if mode == 0:
            # oC0 alone passes or fails the test, the others' alphas both
            # ways.
            sources[0][3] = (0.25, 0.75)[n // 9 % 2]
            cc = control(GREATER, True)
        elif mode == 1:
            sources[0][3] = rng.randrange(9) / 8
            cc = control(cover=True, offsets=SYMMETRIC)
        else:
            cc = control()
        name = f'mrt-{n}'
        out.append(case('mrt', name, samples=samples, formats=formats,
                        color_control=cc, alpha_ref=bits(0.5), blends=blends,
                        sources=sources, clears=clears))
        for t in range(count):
            out.append(case('mrt-alone', f'{name}-{t}', samples=samples,
                            formats=formats[t:t + 1], blends=[blends[t]],
                            sources=[sources[t]], clears=[clears[t]]))
    return out


def depth_cases():
    out = []
    base = [1.0, 0.5, 0.25]
    tests = {'off': (control(), 0.25), 'pass': (control(GREATER, True), 0.75),
             'fail': (control(GREATER, True), 0.25)}
    biases = ((0.0, 0.0), (1.0, 0.0), (-1.0, 0.0), (256.0, 0.0), (0.5, 0.0),
              (0.0, 1.0))
    for samples in (1, 4):
        for test, (cc, a) in tests.items():
            for z in (0.25, 1 / 3, 0.5, 0.75):
                for bias, slope in biases:
                    for clear in (1.0, 0.5):
                        out.append(case(
                            'depth',
                            f'depth-{samples}-{test}-{z}-{bias}-{slope}-'
                            f'{clear}',
                            samples=samples, color_control=cc,
                            alpha_ref=bits(0.5), flags=FLAG_DEPTH, z=z,
                            bias=bias, slope=slope, depth_clear=clear,
                            sources=[with_alpha(base, bits(a))]))
    for offsets in (SYMMETRIC, DITHER):
        for k in range(9):
            out.append(case(
                'depth-cover', f'depth-cover-{offsets:02X}-{k}', samples=4,
                color_control=control(cover=True, offsets=offsets),
                flags=FLAG_DEPTH, z=0.25,
                sources=[with_alpha(base, bits(k / 8))]))
    return out


NEGATIVES = [bits(f) for f in (
    -0.0, -2.0 ** -24, -0.25, -0.5, -0.75, ulp(-1.0, -1), -1.0, ulp(-1.0, 1),
    -2.0, -1e30, -3.4028234663852886e38, 0.0, 0.25)] + [
        0x80000001, INF | 1 << 31, NEG_NAN]


def negative_cases():
    base = [1.0, 0.5, 0.25]
    return [case('test-negative', f'neg-{func}-{a:08X}-{r:08X}',
                 color_control=control(func, True), alpha_ref=r,
                 sources=[with_alpha(base, a)])
            for func in range(8) for a in NEGATIVES for r in NEGATIVES]


def mask_cases():
    out = []
    for samples in (1, 2, 4):
        masks = [1 << b for b in range(16)] + [
            0xF << 4 * n for n in range(4)] + [0x1248, 0x8421]
        for mask in masks:
            out.append(case('mask-bits', f'mask-{samples}-{mask:04X}',
                            samples=samples, aa_mask=mask))
    return out


def cases():
    rng = random.Random(0x58524760)
    return test_cases() + cover_cases() + mrt_cases(rng) + depth_cases()


# The case sets, by name, and their cases.
SETS = (('output-merger', cases), ('output-merger-mask', mask_cases),
            ('output-merger-negative', negative_cases))

WIDTH, HEIGHT = rb.WIDTH, rb.HEIGHT
QUADS_X = WIDTH // 2
QUADS = QUADS_X * (HEIGHT // 2)
# samples -> (RB_SURFACE_INFO and PA_SC_AA_CONFIG code, the average's
# resolve sample select)
SAMPLES = {1: (0, 0), 2: (1, 4), 4: (2, 6)}
# EDRAM tiles a 64x32 target takes at most (at 4x); the depth buffer after
# the fourth.
TARGET_TILES = 8
DEPTH_BASE = 4 * TARGET_TILES
# Target format -> its readbacks' (resolve texture format, endianness) in
# RB_COPY_DEST_INFO.
DEST = {T8888: (6, 0), T2101010: (7, 2)}
DEPTH_DEST = (6, 2)
DEPTH_SOURCE = 4
NO_BLEND_CONTROL = 0x00010001
COLORCONTROL = 0x87000007
# RB_DEPTHCONTROL: the compare, the test and write when a case tests depth.
Z_ENABLE, Z_WRITE_ENABLE = 1 << 1, 1 << 2
POLY_OFFSET_ENABLE = 3 << 11  # front and back
F32_32_32_32 = 38
K = gpugen.C_FIRST_FREE  # pixel c8: (0.5, ...)
PARAM_GEN = 1  # PsParamGen's register, after interpolator 0
# 8 VS and 8 PS registers, vertex and pixel fetches, PsParamGen, one
# interpolator, PS export mode 8.
PROGRAM_CNTL = 7 | (7 << 8) | (1 << 16) | (1 << 17) | (1 << 18) | (8 << 27)
f2w = gpugen.f2w


def blend_control(b):
    """A target's RB_BLENDCONTROLn."""
    op, source, dest, alpha_op, alpha_source, alpha_dest, _ = b
    return (source | op << 5 | dest << 8 | alpha_source << 16 |
            alpha_op << 21 | alpha_dest << 24)


def key(c):
    """What the cases sharing tests share: samples, target formats, whether
    depth is tested and its clear."""
    depth = c['flags'] & FLAG_DEPTH
    return (c['samples'], tuple(c['formats']), depth,
            c['depth_clear'] if depth else None)


def state(c):
    """What a case's draw sets: its z and registers."""
    depth = c['flags'] & FLAG_DEPTH
    offset = c['bias'] or c['slope']
    blends = [blend_control(b) for b in c['blends']]
    blends += [NO_BLEND_CONTROL] * (4 - len(blends))
    mask = sum(b[6] << 4 * t for t, b in enumerate(c['blends']))
    mode = gpugen.MODE_CNTL | (POLY_OFFSET_ENABLE if offset else 0)
    if c['samples'] > 1:
        mode |= gpugen.MSAA_ENABLE
    scale = f2w(c['slope'] * 16)
    return (c['z'], (
        ('RB_DEPTHCONTROL', (c['depth_compare'] << 4 |
                             (Z_ENABLE | Z_WRITE_ENABLE if depth else 0),
                             blends[0], c['color_control'])),
        ('RB_BLENDCONTROL1', tuple(blends[1:])),
        ('RB_COLOR_MASK', (mask,)),
        ('RB_ALPHA_REF', (c['alpha_ref'],)),
        ('PA_SC_AA_MASK', (c['aa_mask'],)),
        ('PA_SU_SC_MODE_CNTL', (mode,)),
        ('PA_SU_POLY_OFFSET_FRONT_SCALE',
         (scale, f2w(c['bias']), scale, f2w(c['bias'])))))


def clear_word(c, t):
    """The word a clearing Resolve writes target t's samples with: its
    clear's channels rounded, red low."""
    rgba = c['clears'][t]
    if c['formats'][t] == T8888:
        return sum(round(v * 255) << 8 * k for k, v in enumerate(rgba))
    return (sum(round(v * 1023) << 10 * k for k, v in enumerate(rgba[:3])) |
            round(rgba[3] * 3) << 30)


def depth_word(clear):
    """A case's RB_DEPTH_CLEAR: D24S8, stencil 0."""
    return min(round(clear * 0xFFFFFF), 0xFFFFFF) << 8


def tests():
    """[(name, key, [case]), ...]: each set's cases sharing tests, in
    order, up to 512 a test, grouped by state."""
    out = []
    for name, generate in SETS:
        groups = {}
        for c in generate():
            groups.setdefault(key(c), []).append(c)
        for k, group in groups.items():
            samples, formats, depth, clear = k
            label = (f'{name} {samples}x '
                    f'{"-".join(str(f) for f in formats)}'
                    f'{f" depth {clear}" if depth else ""}')
            for part in range(0, len(group), QUADS):
                chunk = group[part:part + QUADS]
                chunk = [chunk[i] for i in rbdraw.grouped(
                    [state(c) for c in chunk])]
                out.append((f'{label} {part // QUADS}', k, chunk))
    return out


def readbacks(k):
    """[(name, target or None for depth, sample select)]: the readbacks of
    a test's cases, in order."""
    samples, formats, depth, _ = k
    average = SAMPLES[samples][1]
    out = [(f't{t} s{s}', t, s if s < samples else average)
           for t in range(len(formats)) for s in range(samples + 1)]
    if depth:
        out.append(('depth', None, 0))
    return out


def resolve(fmt, sample_select, source=0):
    dest_format, endian = fmt
    return gpugen.resolve_registers(WIDTH, HEIGHT, dest_format, 4,
                                    endian=endian, source=source,
                                    sample_select=sample_select)


def target_registers(samples, formats):
    """The window, its targets and depth buffer, PsParamGen and the
    viewport."""
    code = SAMPLES[samples][0]
    infos = [t * TARGET_TILES | f << 16 for t, f in enumerate(formats)]
    infos += [0] * (4 - len(infos))
    return [
        ('SQ_CONTEXT_MISC', [1 << 2 | PARAM_GEN << 8 | gpugen.SCREEN_XY]),
        ('SQ_INTERPOLATOR_CNTL', [0xFFFFFFFF]),
        ('RB_SURFACE_INFO', [80 | code << 16]),
        ('RB_COLOR_INFO', [infos[0], DEPTH_BASE, *infos[1:]]),
        ('PA_SC_WINDOW_SCISSOR_BR', [HEIGHT << 16 | WIDTH]),
        ('PA_SC_AA_CONFIG', [gpugen.AA_CONFIG[code]]),
        ('PA_CL_CLIP_CNTL', [1 << 19, gpugen.MODE_CNTL, 0x43F]),
        ('PA_CL_VPORT_XSCALE', [f2w(v) for v in (
            WIDTH / 2, WIDTH / 2, -HEIGHT / 2, HEIGHT / 2, 1.0, 0.0)])]


def pixel_shader():
    """oC0..oC3 = texels (|PsParamGen.xy| + 0.5), unnormalized, of fetch
    constants 1..4, exported with MAX."""
    s = x.Shader()
    s.exec_([x.alu(vector_op=x.V_ADD, vector_dest=2, vector_mask=0x3,
                   src1=x.r(PARAM_GEN, absolute=True),
                   src2=x.c(K, 'xxxx'))])
    s.exec_([x.tfetch(3 + t, 2, 'xy', 1 + t, denormalized=True).serialized()
             for t in range(4)])
    s.alloc(x.ALLOC_COLORS, 0)
    s.exec_([x.alu(vector_op=x.V_MAX, vector_dest=t, vector_mask=0xF,
                   export=True, src1=x.r(3 + t), src2=x.r(3 + t))
             for t in range(4)], end=True)
    return s


class Quads:
    """Inputs of 8 dwords a vertex (position, then interpolator 0 (0, 0, 0,
    1)): the Resolve's rectangle (rb.RECT) in the first, then each run of
    quads q (at pixel 2 (q % 32), 2 (q // 32)) as rows of two triangles in
    clip space, at a z, as a case's quad, their edges as rbdraw.Shapes';
    each added once."""

    def __init__(self):
        self.inputs = rb.RECT + [0, 0]
        self.draws = {}

    def draw(self, first, count, z):
        rows = []
        q = first
        while q < first + count:
            n = min(QUADS_X - q % QUADS_X, first + count - q)
            x0, y0 = 2 * (q % QUADS_X), 2 * (q // QUADS_X)
            rows.append((x0, y0, x0 + 2 * n, y0 + 2))
            q += n
        k = (tuple(rows), z)
        if k not in self.draws:
            start = len(self.inputs) // 8
            for x0, y0, x1, y1 in rows:
                # Edges half a pixel before the pixel centers (at whole
                # pixels, as PA_SU_VTX_CNTL's), so every sample of a pixel
                # is inside its quad.
                left, right = ((v - 0.5) / (WIDTH / 2) - 1 for v in (x0, x1))
                top, bottom = (1 - (v - 0.5) / (HEIGHT / 2) for v in (y0, y1))
                for px, py in ((left, top), (right, top), (left, bottom),
                               (left, bottom), (right, top),
                               (right, bottom)):
                    self.inputs += [f2w(v) for v in (px, py, z, 1.0, 0.0, 0.0,
                                                     0.0, 1.0)]
            self.draws[k] = (pm4.PRIM_TRIANGLE_LIST, start, 6 * len(rows))
        return self.draws[k]


def texture(chunk, texel):
    """A 64x32 k_32_32_32_32_FLOAT texture of texel(case) at each of the
    chunk's quads' 4 pixels, zeros past them."""
    words = [0] * (4 * WIDTH * HEIGHT)
    for q, c in enumerate(chunk):
        x0, y0 = 2 * (q % QUADS_X), 2 * (q // QUADS_X)
        for y in (y0, y0 + 1):
            for x_ in (x0, x0 + 1):
                i = 4 * (x_ + WIDTH * y)
                words[i:i + 4] = texel(c)
    return words


def fetch(name):
    return (gpugen.texture_constant(F32_32_32_32, WIDTH, HEIGHT), name)


def add_tests(suite, quads, name, k, chunk):
    """A chunk's tests: the one drawing its cases, then its other
    readbacks."""
    samples, formats, depth, clear = k
    registers = target_registers(samples, formats)
    shader = gpugen.interpolating_vertex_shader(8)
    pixel = pixel_shader()
    constants = {256 + K: [f2w(0.5), 0, 0, 0]}
    sources, stores = {}, {}
    for t in range(len(formats)):
        suite.add_texture(f'{name} source {t}', texture(
            chunk, lambda c: [word(v) for v in c['sources'][t]]))
        sources[1 + t] = fetch(f'{name} source {t}')
        suite.add_texture(f'{name} store {t}', texture(chunk, lambda c: [
            f2w(((clear_word(c, t) >> 8 * b & 0xFF) + 0.25) / 255)
            for b in range(4)]))
        stores[t] = fetch(f'{name} store {t}')
    # Fetch constants past the targets' fetch target 0's source.
    for slot in range(1 + len(formats), 5):
        sources[slot] = sources[1]
    whole = quads.draw(0, QUADS, 0.5)
    mode = gpugen.MODE_CNTL | (gpugen.MSAA_ENABLE if samples > 1 else 0)
    passes = []
    for t in range(len(formats)):
        passes.append((
            [('RB_COLOR_INFO', [t * TARGET_TILES]),
             ('RB_DEPTHCONTROL', [0, NO_BLEND_CONTROL, COLORCONTROL]),
             ('RB_COLOR_MASK', [0xF]),
             ('PA_SU_SC_MODE_CNTL', [mode])],
            {**sources, 1: stores[t]} if t == 0 else {1: stores[t]},
            whole))
    # The cases: target 0 and its source back, then each run's registers
    # that change.
    last = None
    for (z, regs), first, count in rbdraw.runs([state(c) for c in chunk]):
        writes, fetches = [], {}
        if last is None:
            writes.append(('RB_COLOR_INFO', [formats[0] << 16]))
            fetches = {1: sources[1]}
            last = {}
        for register, values in regs:
            if last.get(register) != values:
                writes.append((register, list(values)))
        last = dict(regs)
        passes.append((writes, fetches, quads.draw(first, count, z)))
    after, fetches, draw = passes.pop()
    clears = None
    if depth:
        clears = gpugen.resolve_registers(
            WIDTH, HEIGHT, 6, 4, sample_select=SAMPLES[samples][1],
            clear=(depth_word(clear), 0))
    for i, (readback, target, select) in enumerate(readbacks(k)):
        if target is None:
            res = resolve(DEPTH_DEST, select, DEPTH_SOURCE)
        else:
            res = resolve(DEST[formats[target]], select)
        if i == 0:
            suite.test(f'{name} {readback}', shader, constants=constants,
                       pixel_shader=pixel, program_cntl=PROGRAM_CNTL,
                       registers=registers, passes=passes,
                       after_passes=after, fetch_constants=fetches,
                       draw=draw, resolve=res, clear=clears)
            continue
        # Nothing drawn: the target to read in RB_COLOR_INFO.
        info = (target * TARGET_TILES | formats[target] << 16
                if target is not None else 0)
        suite.test(f'{name} {readback}', shader, constants=constants,
                   pixel_shader=pixel, program_cntl=PROGRAM_CNTL,
                   registers=registers + [('RB_COLOR_INFO', [info]),
                                          ('PA_SU_SC_MODE_CNTL', [mode])],
                   draw=quads.draw(0, 1, 0.5), resolve=res)


# The ring's words for the tests, less the stream's start and end.
RING_WORDS = gpugen.RING_SIZE // 4 - 1024


def parts():
    """The tests' chunks split into suites fitting the ring, in order: [[(name,
    key, chunk)]]."""
    out, words = [[]], 0
    for test in tests():
        suite = gpugen.Suite('size')
        quads = Quads()
        add_tests(suite, quads, *test)
        suite.set_inputs(quads.inputs, 8)
        layout = {'input': 0, 'dummy': 0, 'textures': {
            n: 0 for n in suite.textures}, 'outputs': {
                t.name: 0 for t in suite.tests}}
        empty = gpugen.Suite('empty')
        size = (len(gpugen.build_stream(suite, layout).words) -
                len(gpugen.build_stream(empty, layout).words))
        if words + size > RING_WORDS:
            out.append([])
            words = 0
        out[-1].append(test)
        words += size
    return out


def build(suite, part):
    """Part `part` (from 1) of the tests (merger1.py, ...)."""
    quads = Quads()
    for test in parts()[part - 1]:
        add_tests(suite, quads, *test)
    suite.set_inputs(quads.inputs, 8)
