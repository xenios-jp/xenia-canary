"""k_2_10_10_10_FLOAT (7e3) and k_2_10_10_10_FLOAT_AS_16_16_16_16 render
targets: encodes through RB_COLOR_INFO's exponent bias, blending, and
resolves decoding, averaging and converting through RB_COPY_DEST_INFO's, at
1, 2 and 4 samples. A 7e3 channel is 3
exponent bits above 7 mantissa bits; sources are float32 bit patterns, so
NaNs keep their payloads.

Each case is a pixel, as the gamma suites draw them: its destination word,
stored raw by a draw of its bytes (byte + 1/4) / 255 with the target's
format k_8_8_8_8 (whose R is the word's low byte), then with the target's
format and exponent bias its draws of its source (from a texture through the
rb suite's pixel shader), each draw a draw of its own. The cases resolving
alike share tests of up to 2048 in order, the pixels holding them grouped by
state. Each test resolves under its format, k_8_8_8_8 for raw readbacks, a
draw writing nothing setting it when the test's last draw left another.
"""

import math
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpugen  # noqa: E402
import rb  # noqa: E402
import rbdraw  # noqa: E402

F7E3, F7E3_AS_16 = 3, 12
RAW, HALF4, RGB10A2, RGBA8, R32F, R16G16F = range(6)
ADD, SUBTRACT, MIN, MAX, REVERSE_SUBTRACT = 0, 1, 2, 3, 4
ZERO, ONE, SRC_ALPHA, ONE_MINUS_SRC_ALPHA = 0, 1, 6, 7
# RB_COPY_DEST_INFO, less the exponent bias: the format at bit 7, endianness
# low, the number format at bit 13.
ENDIAN_8IN16, ENDIAN_8IN32 = 1, 2
DEST_8888 = 6 << 7
DEST_8888_UINT = 6 << 7 | 2 << 13
DEST_2_10_10_10 = 7 << 7 | ENDIAN_8IN32
DEST_16_16_16_16 = 26 << 7 | ENDIAN_8IN16
# 6-bit signed exponent biases, as RB_COLOR_INFO and RB_COPY_DEST_INFO hold
# them: the 4, 5, 2 and -1 titles' targets use and the -5, -4, -3 and 1
# their resolves do, and more.
BIASES = (0, 1, 2, 4, 5, 24, 31, 32, 39, 60, 62, 63)
RESOLVE_BIASES = (0, 1, 2, 59, 60, 61, 63, 39)


def signed(bias):
    return bias - 64 if bias >= 32 else bias


def bits(f):
    return struct.unpack('>I', struct.pack('>f', f))[0]


def flt(u):
    return struct.unpack('>f', struct.pack('>I', u))[0]


def value(code):
    """The 7e3 code's value."""
    e, m = code >> 7, code & 0x7F
    return m / 512.0 if e == 0 else 2.0 ** (e - 3) * (1 + m / 128.0)


def next_up(u):
    return u + 1 if u < 0x80000000 else u - 1


def next_down(u):
    return u - 1 if 0 < u < 0x80000000 else u + 1


def word(r, g, b, a):
    return r | g << 10 | b << 20 | a << 30


def case(source=(0, 0, 0, 0), fmt=F7E3, samples=1, color_bias=0,
         resolve_bias=0, readback=RAW, coverage=0, draws=1,
         color=(ADD, ONE, ZERO), alpha=(ADD, ONE, ZERO), clear=0,
         dest_info=0, constant=(0, 0, 0, 0)):
    """`source` and `constant` float32 bit patterns, `clear` a raw word."""
    return {'format': fmt, 'samples': samples, 'color_bias': color_bias,
            'resolve_bias': resolve_bias, 'readback': readback,
            'coverage': coverage, 'draws': draws, 'color': color,
            'alpha': alpha, 'clear': clear, 'dest_info': dest_info,
            'source': list(source), 'constant': list(constant)}


def words(c):
    """The case as 32 words."""
    return [c['format'], c['samples'], c['color_bias'], c['resolve_bias'],
            c['readback'], c['coverage'], c['draws'], 15, *c['color'],
            *c['alpha'], 1, c['clear'], c['clear'], c['dest_info'], 0, 0,
            *c['source'], 0, 0, 0, 0, *c['constant']]


def alpha_bits(j):
    return bits((j % 1025) / 1024.0)


SPECIALS = [0x00000000, 0x80000000, 0x80000001, bits(-1e-30), bits(-1.0),
            0xFF800000, 0x7F800000, 0x7FC00000, 0xFFC00000, 0x7F800001,
            0x7FBFFFFF, 0x00000001, 0x00800000, bits(2.0 ** -10),
            bits(2.0 ** -11), bits(2.0 ** -12), bits(31.875), bits(31.9375),
            bits(31.94), bits(32.0), bits(33.0), bits(64.0), bits(1e30),
            0x7F7FFFFF, bits(0.25), next_down(bits(0.25)),
            bits(0.25 - 1 / 1024.0), next_down(bits(1 / 1024.0)),
            bits(1 / 1024.0), next_up(bits(1 / 1024.0))]
ALPHA_SPECIALS = [0x00000000, 0x80000000, bits(-1.0), 0xFF800000, 0x7F800000,
                  0x7FC00000, 0xFFC00000, bits(1.0), bits(1.5), bits(1e30)]
for k in (1, 3, 5):
    t = bits(k / 6.0)
    ALPHA_SPECIALS += [next_down(t), t, next_up(t)]
for k in (1, 2):
    t = bits(k / 3.0)
    ALPHA_SPECIALS += [next_down(t), t, next_up(t)]


def scaled(u, bias):
    """u's float divided by 2 to the bias, which the target's bias undoes, as
    float32 leaves it."""
    f = flt(u)
    if f != f or math.isinf(f) or f == 0:
        return u
    g = f * 2.0 ** -signed(bias)
    return bits(g) if abs(g) <= flt(0x7F7FFFFF) else u


def encode_cases():
    out = []
    j = 0
    for bias in BIASES:
        step = 1 if bias == 0 else 8
        for k in range(0, 1023, step):
            tie = bits((value(k) + value(k + 1)) / 2)
            v = bits(value(k))
            for rgb in ((v, tie, next_up(tie)),
                        (next_down(tie), next_up(v), next_down(v))):
                out.append(case((*(scaled(u, bias) for u in rgb),
                                 alpha_bits(j)), color_bias=bias))
                j += 1
        if bias in (0, 2, 4, 5, 63, 39):
            for n in range(0, len(SPECIALS), 3):
                rgb = (SPECIALS[n:n + 3] + [0, 0])[:3]
                out.append(case((*(scaled(u, bias) for u in rgb),
                                 alpha_bits(j)), color_bias=bias))
                j += 1
            for a in ALPHA_SPECIALS:
                out.append(case((0, 0, 0, a), color_bias=bias))
    return out


def clear_words():
    """Every code, three a word, alpha cycling through its four."""
    return [word(k, min(k + 342, 1023), min(k + 684, 1023), k & 3)
            for k in range(342)]


# The conversions out of 7e3 targets titles resolve: (format, samples,
# resolve biases, readback, RB_COPY_DEST_INFO).
CONVERSIONS = (
    (F7E3, 1, (0, 59), HALF4, DEST_16_16_16_16),
    (F7E3, 4, (59,), HALF4, DEST_16_16_16_16),
    (F7E3, 1, (0,), RGBA8, DEST_8888),
    (F7E3, 1, (0,), RGBA8, DEST_8888_UINT),
    (F7E3, 2, (0,), RGBA8, DEST_8888_UINT),
    (F7E3_AS_16, 1, (0, 59), RGB10A2, DEST_2_10_10_10),
    (F7E3_AS_16, 1, (0, 60), RGBA8, DEST_8888),
    (F7E3_AS_16, 2, (0,), RGBA8, DEST_8888),
    (F7E3_AS_16, 4, (0,), RGBA8, DEST_8888),
    (F7E3_AS_16, 1, (0,), RGBA8, DEST_8888_UINT),
    (F7E3_AS_16, 2, (0,), RGBA8, DEST_8888_UINT))


def resolve_cases():
    out = []
    clears = clear_words()
    for k in range(0, 342, 23):
        out.append(case(clear=clears[k], draws=0))
    for fmt in (F7E3, F7E3_AS_16):
        for samples in (1, 2, 4):
            biases = RESOLVE_BIASES if samples == 1 else (0, 60, 61, 1)
            for bias in biases:
                for w in clears:
                    out.append(case(fmt=fmt, samples=samples,
                                    resolve_bias=bias, readback=HALF4,
                                    clear=w, draws=0))
    for fmt, samples, biases, readback, dest in CONVERSIONS:
        for bias in biases:
            for w in clears:
                out.append(case(fmt=fmt, samples=samples, resolve_bias=bias,
                                readback=readback, clear=w, draws=0,
                                dest_info=dest))
    return out


GRID = tuple(sorted(set(range(0, 1024, 32)) | {1, 127, 129, 1023}))


def blend_cases():
    out = []
    alphas = [bits(x) for x in (0.25, 0.5, 0.75, 0.3)]
    ops = (((ADD, ONE, ONE), (ADD, ONE, ONE)),
           ((ADD, SRC_ALPHA, ONE_MINUS_SRC_ALPHA), (ADD, ONE, ZERO)),
           ((MAX, ONE, ONE), (MAX, ONE, ONE)),
           ((REVERSE_SUBTRACT, ONE, ONE), (REVERSE_SUBTRACT, ONE, ONE)))
    j = 0
    for color, alpha in ops:
        for bias in (0, 4, 63):
            for s in GRID:
                for d in GRID:
                    src = (bits(value(min(s + 3, 1023))) if s % 2 == 0
                           else bits(value(s)))
                    rgb = (src, bits(value(1023 - s)), bits(value(s)))
                    out.append(case(
                        (*(scaled(u, bias) for u in rgb), alphas[j % 4]),
                        color_bias=bias, color=color, alpha=alpha,
                        clear=word(d, 1023 - d, (d * 7) % 1024, j % 4)))
                    j += 1
    for draws in (2, 3, 4, 8):
        for s in GRID:
            out.append(case((bits(value(s)), bits(value(s) / 3),
                             bits(value(s) / 7), bits(0.25)), draws=draws,
                            color=(ADD, ONE, ONE), alpha=(ADD, ONE, ONE),
                            clear=word(s, s // 2, s // 4, 0)))
    return out


# Alpha covering n samples with alpha to coverage, between the thresholds
# (4n - 2) / (4N).
COVERAGE_ALPHAS = {2: {1: 0.5}, 4: {1: 0.25, 2: 0.5, 3: 0.75}}


def msaa_cases():
    out = []
    for samples, alphas in COVERAGE_ALPHAS.items():
        for a in alphas.values():
            for s in GRID:
                for d in GRID:
                    out.append(case(
                        (bits(value(s)), bits(value(1023 - s)),
                         bits(value(s ^ 0x55)), bits(a)),
                        samples=samples, readback=HALF4, coverage=1,
                        clear=word(d, (d + 300) % 1024, 1023 - d, 3)))
    for s in GRID:
        for d in GRID:
            out.append(case((bits(value(s)), bits(value(1023 - s)),
                             bits(value(s ^ 0x55)), bits(0.5)),
                            samples=4, resolve_bias=39, readback=HALF4,
                            coverage=1,
                            clear=word(d, (d + 300) % 1024, 1023 - d, 3)))
    for samples in (2, 4):
        for k in range(0, 1023, 4):
            tie = bits((value(k) + value(k + 1)) / 2)
            out.append(case((bits(value(k)), tie, next_up(tie),
                             alpha_bits(k)), samples=samples,
                            readback=HALF4))
    return out


def cases():
    return encode_cases() + resolve_cases() + blend_cases() + msaa_cases()


WIDTH, HEIGHT = rb.WIDTH, rb.HEIGHT
NO_BLEND = 0x00010001
# RB_COLORCONTROL: Direct3D's, and with alpha to mask at symmetric offsets.
COLORCONTROL = 0x87000007
COVERAGE = 0xAA000017
# samples -> (RB_SURFACE_INFO and PA_SC_AA_CONFIG code, resolve sample
# select: the average of all of them)
SAMPLES = {1: (0, 0), 2: (1, 4), 4: (2, 6)}
# The default readback, k_16_16_16_16_FLOAT.
DEST_HALF4 = 32 << 7 | 7 << 13 | ENDIAN_8IN16
DEST_RAW = 6 << 7
TEXEL_BYTES = {6: 4, 7: 4, 26: 8, 32: 8}
f2w = gpugen.f2w


def control(c):
    """The case's RB_BLENDCONTROL0."""
    (color_op, color_src, color_dest), (alpha_op, alpha_src, alpha_dest) = (
        c['color'], c['alpha'])
    return (color_src | color_op << 5 | color_dest << 8 | alpha_src << 16 |
            alpha_op << 21 | alpha_dest << 24)


def state(c):
    """What the case's draws set: (RB_COLOR_INFO's exponent bias,
    RB_BLENDCONTROL0, RB_BLEND_RED..ALPHA, alpha to mask, draws)."""
    return (c['color_bias'], control(c), tuple(c['constant']),
            c['coverage'], c['draws'])


def readback(c):
    """(format, samples, RB_COPY_DEST_INFO, raw): what the case's resolve
    takes, raw through k_8_8_8_8."""
    if c['readback'] == RAW:
        assert c['samples'] == 1 and not c['resolve_bias']
        return F7E3, 1, DEST_RAW, True
    assert not c['color_bias']
    info = c['dest_info'] or DEST_HALF4
    return c['format'], c['samples'], info | c['resolve_bias'] << 16, False


def resolve(samples, info):
    dest_format = info >> 7 & 0x3F
    return gpugen.resolve_registers(
        WIDTH, HEIGHT, dest_format, TEXEL_BYTES[dest_format],
        number=info >> 13 & 7, endian=info & 7,
        sample_select=SAMPLES[samples][1], exp_bias=info >> 16 & 0x3F)


def tests():
    """[(name, format, samples, RB_COPY_DEST_INFO, raw, [case index])]: the
    cases resolving alike, up to 2048 a test, in order."""
    keys = {}
    for i, c in enumerate(cases()):
        keys.setdefault(readback(c), []).append(i)
    out = []
    for (fmt, samples, info, raw), indices in keys.items():
        for part in range(0, len(indices), WIDTH * HEIGHT):
            out.append((f'{"raw" if raw else fmt} {samples}x {info:06X} '
                        f'{part // (WIDTH * HEIGHT)}', fmt, samples, info,
                        raw, indices[part:part + WIDTH * HEIGHT]))
    return out


def order(chunk):
    """The chunk's pixels' cases, grouped by state."""
    return [chunk[i] for i in rbdraw.grouped([state(c) for c in chunk])]


def build(suite):
    shapes = rbdraw.Shapes()
    whole = rb.RECT_DRAW
    all_cases = cases()
    for name, fmt, samples, info, raw, indices in tests():
        code = SAMPLES[samples][0]
        registers = rb.registers(fmt, code, mode_cntl=gpugen.MODE_CNTL)
        pixels = order([all_cases[i] for i in indices])
        source, dest = [0] * (4 * WIDTH * HEIGHT), [0] * (4 * WIDTH * HEIGHT)
        for pixel, c in enumerate(pixels):
            source[4 * pixel:4 * pixel + 4] = c['source']
            dest[4 * pixel:4 * pixel + 4] = [
                f2w(((c['clear'] >> (8 * k) & 0xFF) + 0.25) / 255)
                for k in range(4)]
        suite.add_texture(f'{name} source', source)
        suite.add_texture(f'{name} dest', dest)
        draws = [([('RB_COLOR_INFO', [0]),
                   ('RB_BLENDCONTROL0', [NO_BLEND]),
                   ('RB_COLORCONTROL', [COLORCONTROL])],
                  rb.fetch(f'{name} dest'), whole)]
        fetched = False
        for (bias, blend, constant, coverage, n), first, count in rbdraw.runs(
                [state(c) for c in pixels]):
            for _ in range(n):
                draws.append((
                    [('RB_COLOR_INFO', [fmt << 16 | bias << 20]),
                     ('RB_BLENDCONTROL0', [blend]),
                     ('RB_BLEND_RED', list(constant)),
                     ('RB_COLORCONTROL', [COVERAGE if coverage
                                          else COLORCONTROL])],
                    {} if fetched else rb.fetch(f'{name} source'),
                    shapes.pixels(first, count)))
                fetched = True
        color_info = 0 if raw else fmt << 16
        if dict(draws[-1][0])['RB_COLOR_INFO'] != [color_info]:
            draws.append(([('RB_COLOR_INFO', [color_info]),
                           ('RB_COLOR_MASK', [0])], {}, whole))
        last, _, draw = draws[-1]
        rb.test(suite, name, f'{name} dest' if len(draws) == 1
                else f'{name} source', registers, resolve(samples, info),
                draw=draw, passes=draws[:-1], after_passes=last)
    suite.set_inputs(shapes.inputs, 2)
