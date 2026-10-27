"""Blend constants: how RB_BLEND_RED..ALPHA become blend factors, per target
format. The rb suite's draw: a first draw stores 'dest' unblended, then 64
draws, one per state s, each over 32 pixels (32 s to 32 s + 31) with
RB_BLENDCONTROL0 = CONSTANT_COLOR + INV_CONSTANT_COLOR (alpha
CONSTANT_ALPHA + INV_CONSTANT_ALPHA) and RB_BLEND_RED..ALPHA = CONSTANTS[4 s
.. 4 s + 3], store 'source'. Of a state's pixels j = 0..31, j < 16 blend a
source S(j) over 0 (the term source x constant alone) and j >= 16 a source
0 over the destination S(j - 16) (destination x (1 - constant) alone). Read
back raw (copy command 0).

S(j) per format:
- 2_10_10_10_FLOAT: R the 7e3 code 7 << 7 | 8j + 7, G 7 << 7 | 8j + 3, B
  6 << 7 | 127 - 8j, A 1.
- 16_16_16_16: R 16 + j, G -(8 + j / 2), B 31.99 - j, A 1 + j / 16.
- 16_16_16_16_FLOAT: R 1, G 2, B -1.5, A 0.5 + j / 64.
- 8_8_8_8 and 2_10_10_10: R 1, G (255 - j) / 255, B (128 + j) / 255, A 1.

CONSTANTS: k/64 (k = 0..64), the blend suite's 0.3 0.6 0.9 0.1, k/255,
(k + 0.5)/1024, k/1023 for a few k, negatives, values above 1, NaN, Inf,
-0, denormals, then random 24-bit fractions.
"""

import os
import random
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpugen  # noqa: E402
import rb  # noqa: E402
import rbdraw  # noqa: E402

STATES, PIXELS = 64, 32
CONSTANT_COLOR, INV_CONSTANT_COLOR = 12, 13
CONSTANT_ALPHA, INV_CONSTANT_ALPHA = 14, 15
ADD = 0
NO_BLEND = 0x00010001
CONTROL = (CONSTANT_COLOR | ADD << 5 | INV_CONSTANT_COLOR << 8 |
           CONSTANT_ALPHA << 16 | ADD << 21 | INV_CONSTANT_ALPHA << 24)


def bits(v):
    return struct.unpack('>I', struct.pack('>f', v))[0]


def value_7e3(c):
    e, m = c >> 7, c & 127
    return m / 512 if e == 0 else (128 + m) / 128 * 2.0 ** (e - 3)


def constants():
    rng = random.Random(0xB1E4)
    out = [bits(k / 64) for k in range(65)]
    out += [bits(v) for v in (0.3, 0.6, 0.9, 0.1)]
    out += [bits(k / 255) for k in (1, 2, 3, 7, 85, 127, 128, 170, 253, 254)]
    out += [bits((k + 0.5) / 1024) for k in (0, 1, 306, 307, 613, 614, 921,
                                             1022)]
    out += [bits(k / 1023) for k in (1, 307, 614, 921)]
    out += [bits(v) for v in (-0.25, -1.0, -2.0 ** -20, 1.0001, 1.5, 2.0,
                              31.875, 65504.0, 2.0 ** -17, 2.0 ** -16,
                              3 * 2.0 ** -17, 0.5 + 2.0 ** -24,
                              1 - 2.0 ** -24, 1 / 3, 2 / 3)]
    out += [0x7FC00000, 0xFFC00000, 0x7F800000, 0xFF800000, 0x80000000,
            0x00000001, 0x00800000]
    while len(out) < 4 * STATES:
        out.append(bits(rng.getrandbits(24) / 2 ** 24))
    return out


def sources(fmt):
    out = []
    for j in range(16):
        if fmt == 3:
            out.append([bits(value_7e3(7 << 7 | (8 * j + 7))),
                        bits(value_7e3(7 << 7 | (8 * j + 3))),
                        bits(value_7e3(6 << 7 | (127 - 8 * j))), bits(1.0)])
        elif fmt == 5:
            out.append([bits(v) for v in (16 + j, -(8 + j / 2), 31.99 - j,
                                          1 + j / 16)])
        elif fmt == 7:
            out.append([bits(v) for v in (1.0, 2.0, -1.5, 0.5 + j / 64)])
        else:
            out.append([bits(v) for v in (1.0, (255 - j) / 255,
                                          (128 + j) / 255, 1.0)])
    return out


def textures(fmt):
    src = sources(fmt)
    source, dest = [], []
    for i in range(rb.WIDTH * rb.HEIGHT):
        j = i % PIXELS
        if j < 16:
            source += src[j]
            dest += [0, 0, 0, 0]
        else:
            source += [0, 0, 0, 0]
            dest += src[j - 16]
    return source, dest


# (name, ColorRenderTargetFormat, bytes per pixel)
FORMATS = [('2_10_10_10_FLOAT', 3, 4), ('16_16_16_16', 5, 8),
           ('16_16_16_16_FLOAT', 7, 8), ('8_8_8_8', 0, 4),
           ('2_10_10_10', 2, 4)]


def build(suite):
    shapes = rbdraw.Shapes()
    k = constants()
    for name, fmt, bpp in FORMATS:
        source, dest = textures(fmt)
        suite.add_texture(f'{name} source', source)
        suite.add_texture(f'{name} dest', dest)
        draws = [([('RB_BLENDCONTROL0', [NO_BLEND])], rb.fetch(f'{name} dest'),
                  rb.RECT_DRAW)]
        for s in range(STATES):
            draws.append(([('RB_BLENDCONTROL0', [CONTROL]),
                           ('RB_BLEND_RED', k[4 * s:4 * s + 4])],
                          rb.fetch(f'{name} source') if s == 0 else {},
                          shapes.pixels(PIXELS * s, PIXELS)))
        registers, _, draw = draws[-1]
        rb.test(suite, f'{name} raw', f'{name} source', rb.registers(fmt),
                gpugen.resolve_registers(rb.WIDTH, rb.HEIGHT,
                                         6 if bpp == 4 else 26, bpp,
                                         command=0),
                draw=draw, passes=draws[:-1], after_passes=registers)
    suite.set_inputs(shapes.inputs, 2)
