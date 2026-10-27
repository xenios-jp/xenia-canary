"""Texture formats and fetch scales games use that no suite pinned: 24_8,
24_8_FLOAT, 2_10_10_10_AS_16_16_16_16, the EXPAND family and exp_adjust,
point and bilinear, from vertex shaders.

Every texture is 256x4 texels, linear, read with endian none (texels
little-endian). Vertex v (1024), x = v % 256, row = v // 256:
- r1 = (x + .5, row + .5): point sampled at texel (x, row), unnormalized;
- r2 = (x + .5 + row / 4, 1.5): between texels x and x + 1 of row 1 (the
  weight 0, 1/4, 1/2 and 3/4 by row), no vertical blend;
- r3 = (x + .5 + row / 4, 1.5 + (x % 2 + 1) / 4 + row / 8): both axes.
Each test exports the fetched float4.

Texel data:
- 24_8: depth << 8 | stencil, depths of 0, 1, 2^k, 2^k - 1, 0x555555 ...,
  stencils 0-255;
- 24_8_FLOAT: the same words (a 20e4 depth: a 4-bit exponent over a 20-bit
  mantissa) with every exponent over 0, 1, half and all-ones mantissas;
- 2_10_10_10_AS_16_16_16_16: every 10-bit field value in turn;
- the others (16_16_16_16_EXPAND, 16_16_EXPAND, 16_EXPAND, 8_8_8_8,
  32_FLOAT): seeded random words with the 16-bit and byte extremes in the
  first texels.
exp_adjust sweeps all 64 values over 8_8_8_8.
"""

import os
import random
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gpugen  # noqa: E402
import pm4  # noqa: E402
import xenos as x  # noqa: E402

SLOT = 1
WIDTH, ROWS = 256, 4
VERTICES = WIDTH * ROWS
UNSIGNED, SIGNED = 0, 1
POINT, LINEAR = (0, 0, 0), (1, 1, 0)
f2w = gpugen.f2w


def inputs():
    words = []
    for v in range(VERTICES):
        col, row = v % WIDTH, v // WIDTH
        words += [f2w(col + .5), f2w(row + .5), 0, 0,
                  f2w(col + .5 + row / 4), f2w(1.5), 0, 0,
                  f2w(col + .5 + row / 4),
                  f2w(1.5 + (col % 2 + 1) / 4 + row / 8), 0, 0]
    return words


def pack(values, fmt):
    return b''.join(struct.pack(fmt, v) for v in values)


def depth_words(rng):
    depths = [0, 1, 2, 0x7FFFFF, 0x800000, 0x800001, 0xFFFFFF, 0xFFFFFE,
              0x555555, 0xAAAAAA, 0x123456, 0xABCDEF, 0x00FFFF, 0x010000,
              0xFF0000, 0x00FF00]
    depths += [1 << k for k in range(24)] + [(1 << k) - 1 for k in range(2, 25)]
    words = []
    for i in range(WIDTH * ROWS):
        depth = depths[(i * 7 + i // 256) % len(depths)] if i % 3 else \
            rng.getrandbits(24)
        words.append(depth << 8 | (i * 37 + i // 256) % 256)
    return pack(words, '<I')


def float_depth_words(rng):
    words = []
    mantissas = [0, 1, 0x80000, 0xFFFFF, 0x7FFFF, 0x40000]
    for i in range(WIDTH * ROWS):
        if i < 16 * len(mantissas):
            depth = (i // len(mantissas)) << 20 | mantissas[i % len(mantissas)]
        else:
            depth = rng.getrandbits(24)
        words.append(depth << 8 | (i * 29) % 256)
    return pack(words, '<I')


def fields_words():
    words = []
    for i in range(WIDTH * ROWS):
        r, g, b = i % 1024, (i * 3 + 1) % 1024, (1023 - i) % 1024
        words.append(r | g << 10 | b << 20 | (i // 256 % 4) << 30)
    return pack(words, '<I')


def random_bytes(rng, texel_bytes):
    data = bytearray(rng.getrandbits(8) for _ in range(WIDTH * ROWS *
                                                       texel_bytes))
    half = texel_bytes // 2 if texel_bytes in (2, 4, 8) else texel_bytes
    patterns = [b'\x00\xFF', b'\xFF\x7F', b'\x00\x80', b'\x01\x00',
                b'\xFF\xFF', b'\x00\x00', b'\xFF\x80', b'\x00\x7F']
    for i in range(0, min(len(data), 16 * 2 * 8), 2):
        data[i:i + 2] = patterns[(i // 2) % len(patterns)]
    del half
    return bytes(data)


def words_of(data):
    data += bytes(-len(data) % 4)
    return list(struct.unpack(f'>{len(data) // 4}I', data))


def build(suite):
    rng = random.Random(0x7E1F0)
    suite.set_inputs(inputs(), 12)
    textures = {
        '24_8': depth_words(rng), '24_8_FLOAT': float_depth_words(rng),
        '2_10_10_10': fields_words(),
        '16_16_16_16': random_bytes(rng, 8), '16_16': random_bytes(rng, 4),
        '16': random_bytes(rng, 2), '8_8_8_8': random_bytes(rng, 4),
        '32_FLOAT': pack([rng.choice([0x3F800000, 0x40000000, 0x3F000000,
                                      0x41200000, rng.getrandbits(32)])
                          for _ in range(WIDTH * ROWS)], '<I'),
    }
    for name, data in textures.items():
        suite.add_texture(name, words_of(data))
    move = gpugen.export_copy(3)
    vertex = gpugen.fetch_inputs(12)

    def test(name, texture, fmt, filters=POINT, coordinates=0, **constant):
        """coordinates 0: point (r1), 1: horizontal blend (r2), 2: both (r3)"""
        words = gpugen.texture_constant(
            fmt, WIDTH, ROWS, filters=filters, pitch=WIDTH,
            endian=pm4.ENDIAN_NONE, **constant)
        fetch = x.tfetch(3, 1 + coordinates, 'xy', SLOT,
                         denormalized=True).serialized()
        suite.test(name, gpugen.memexport_shader(
            12, [], [[move]], fetches=vertex + [fetch]),
            fetch_constants={SLOT: (words, texture)})

    # 24_8 and 24_8_FLOAT (unsigned, as games read depth).
    for fname, fmt in (('24_8', 22), ('24_8_FLOAT', 23)):
        test(f'{fname} point', fname, fmt)
        test(f'{fname} bilinear', fname, fmt, LINEAR, 1)
        test(f'{fname} bilinear 2d', fname, fmt, LINEAR, 2)
    # 2_10_10_10_AS_16_16_16_16.
    test('2_10_10_10_AS_16_16_16_16 point', '2_10_10_10', 54)
    test('2_10_10_10_AS_16_16_16_16 exp_adjust 3', '2_10_10_10', 54,
         exp_adjust=3)
    test('2_10_10_10_AS_16_16_16_16 bilinear', '2_10_10_10', 54, LINEAR, 1)
    # EXPAND, signed as games read 16_16_16_16_EXPAND, and unsigned.
    for fname, fmt, texture in (('16_16_16_16_EXPAND', 29, '16_16_16_16'),
                                ('16_16_EXPAND', 28, '16_16'),
                                ('16_EXPAND', 27, '16')):
        for sname, sign in (('signed', SIGNED), ('unsigned', UNSIGNED)):
            test(f'{fname} {sname} bilinear', texture, fmt, LINEAR, 1,
                 sign=sign)
            test(f'{fname} {sname} bilinear 2d', texture, fmt, LINEAR, 2,
                 sign=sign)
    test('16_16_16_16_EXPAND signed exp_adjust 2', '16_16_16_16', 29,
         sign=SIGNED, exp_adjust=2)
    # exp_adjust, every value.
    for adjust in range(-32, 32):
        test(f'8_8_8_8 exp_adjust {adjust}', '8_8_8_8', 6, exp_adjust=adjust)
    test('8_8_8_8 exp_adjust 2 bilinear', '8_8_8_8', 6, LINEAR, 1,
         exp_adjust=2)
    test('8_8_8_8 exp_adjust 2 bilinear 2d', '8_8_8_8', 6, LINEAR, 2,
         exp_adjust=2)
    test('8_8_8_8 exp_adjust 2 signed', '8_8_8_8', 6, sign=SIGNED,
         exp_adjust=2)
    test('8_8_8_8 exp_adjust 2 signed bilinear', '8_8_8_8', 6, LINEAR, 1,
         sign=SIGNED, exp_adjust=2)
    test('32_FLOAT exp_adjust 2', '32_FLOAT', 36, exp_adjust=2)
    test('32_FLOAT exp_adjust 2 bilinear', '32_FLOAT', 36, LINEAR, 1,
         exp_adjust=2)
    test('16_16_16_16 signed exp_adjust 2', '16_16_16_16', 26, sign=SIGNED,
         exp_adjust=2)
