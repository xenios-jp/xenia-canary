"""24_8 and 24_8_FLOAT point sampled over 65536 depths each, with the fraction
(0) and integer (1) number formats (games set 1 for 24_8_FLOAT): the texel's
float, from vertex shaders.

Each texture is 256x256 texels, linear, endian none (texels little-endian,
depth << 8 | stencil). Vertex v (65536) point samples texel (v % 256,
v // 256), unnormalized, and the test exports the float4 fetched.

Depths, in order: the powers of two and their neighbours (24 bits, + 0, - 1,
+ 1, - 2), the top and bottom 64 values, 0x7FFFFF-0x800010, then random:
a random bit length with a random mantissa, a third of them with the top
bits all ones (24_8; 24_8_FLOAT randoms are an exponent 0-15 over all
mantissas, and over the all-zero, all-one and half mantissas first).
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
WIDTH = 256
VERTICES = WIDTH * WIDTH
f2w = gpugen.f2w


def inputs():
    words = []
    for v in range(VERTICES):
        words += [f2w(v % WIDTH + .5), f2w(v // WIDTH + .5), 0, 0]
    return words


def unorm_depths(rng):
    depths = []
    for k in range(24):
        for delta in (0, -1, 1, -2):
            d = (1 << k) + delta
            if 0 <= d < 1 << 24:
                depths.append(d)
    depths += list(range(64)) + list(range(0x1000000 - 64, 0x1000000))
    depths += list(range(0x7FFFFF - 16, 0x800010))
    depths += [(1 << k) - 1 for k in range(1, 25)]
    while len(depths) < VERTICES:
        bits = rng.randint(1, 24)
        d = rng.getrandbits(bits) | 1 << (bits - 1)
        if rng.random() < .33 and bits > 3:
            ones = rng.randint(1, bits - 1)
            d |= (((1 << ones) - 1) << (bits - ones))
        depths.append(d)
    return depths[:VERTICES]


def float_depths(rng):
    depths = []
    for e in range(16):
        for m in (0, 1, 0x80000, 0xFFFFF, 0x7FFFF, 0x40000, 0xFFFFE):
            depths.append(e << 20 | m)
    while len(depths) < VERTICES:
        depths.append(rng.getrandbits(24) if rng.random() < .5 else
                      rng.randint(0, 15) << 20 | rng.getrandbits(20))
    return depths[:VERTICES]


def texture(depths):
    data = b''.join(struct.pack('<I', d << 8 | i % 251)
                    for i, d in enumerate(depths))
    return list(struct.unpack(f'>{len(data) // 4}I', data))


def build(suite):
    rng = random.Random(0xDE97)
    suite.set_inputs(inputs(), 4)
    move = gpugen.export_copy(3)
    vertex = gpugen.fetch_inputs(4)
    fetch = x.tfetch(3, 1, 'xy', SLOT, denormalized=True).serialized()
    shader = gpugen.memexport_shader(4, [], [[move]],
                                     fetches=vertex + [fetch])
    suite.add_texture('24_8', texture(unorm_depths(rng)))
    suite.add_texture('24_8_FLOAT', texture(float_depths(rng)))
    for fname, fmt in (('24_8', 22), ('24_8_FLOAT', 23)):
        for num_format in (0, 1):
            words = gpugen.texture_constant(
                fmt, WIDTH, WIDTH, pitch=WIDTH, endian=pm4.ENDIAN_NONE,
                num_format=num_format)
            suite.test(f'{fname} point num_format {num_format}', shader,
                       fetch_constants={SLOT: (words, fname)})
