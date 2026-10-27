"""Which source component fills the components a texture format lacks:
the four floats fetched from point sampled 32x8 textures of the formats of
1, 2 and 3 components (and 4 as a control), with the identity swizzle and
endian none (texels little-endian), random texels.

Vertex v (256) point samples texel (v % 32, v // 32), unnormalized, and the
test exports the float4 fetched.
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
WIDTH, ROWS = 32, 8
VERTICES = WIDTH * ROWS
f2w = gpugen.f2w

# name -> (TextureFormat, bytes a texel)
FORMATS = {
    '8': (2, 1), '8_A': (8, 1), '8_B': (9, 1), '16': (24, 2), '16_FLOAT': (30, 2),
    '32_FLOAT': (36, 4), '24_8': (22, 4),
    '8_8': (10, 2), '16_16': (25, 4), '16_16_FLOAT': (31, 4),
    '32_32_FLOAT': (37, 8),
    '5_6_5': (4, 2), '6_5_5': (5, 2), '10_11_11': (16, 4),
    '11_11_10': (17, 4),
    '1_5_5_5': (3, 2), '4_4_4_4': (15, 2), '8_8_8_8': (6, 4),
    '2_10_10_10': (7, 4), '16_16_16_16': (26, 8),
}
BLOCK = {'DXN': (49, 16), 'CTX1': (60, 8), 'DXT5A': (59, 8)}


def inputs():
    words = []
    for v in range(VERTICES):
        words += [f2w(v % WIDTH + .5), f2w(v // WIDTH + .5), 0, 0]
    return words


def build(suite):
    rng = random.Random(0x5112)
    suite.set_inputs(inputs(), 4)
    move = gpugen.export_copy(3)
    vertex = gpugen.fetch_inputs(4)
    fetch = x.tfetch(3, 1, 'xy', SLOT, denormalized=True).serialized()
    shader = gpugen.memexport_shader(4, [], [[move]],
                                     fetches=vertex + [fetch])
    for name, (fmt, size) in FORMATS.items():
        data = bytes(rng.getrandbits(8) for _ in range(WIDTH * ROWS * size))
        data += bytes(-len(data) % 4)
        suite.add_texture(name, list(struct.unpack(f'>{len(data) // 4}I',
                                                   data)))
        words = gpugen.texture_constant(
            fmt, WIDTH, ROWS, pitch=WIDTH, endian=pm4.ENDIAN_NONE)
        suite.test(name, shader, fetch_constants={SLOT: (words, name)})
    for name, (fmt, size) in BLOCK.items():
        # 32x8 texels = 8x2 blocks (CTX1/DXN/DXT5A): fetch inside them with
        # the same coordinates.
        blocks = (WIDTH // 4) * (ROWS // 4)
        data = bytes(rng.getrandbits(8) for _ in range(blocks * size * 4))
        suite.add_texture(name, list(struct.unpack(f'>{len(data) // 4}I',
                                                   data)))
        words = gpugen.texture_constant(
            fmt, WIDTH, ROWS, pitch=WIDTH, endian=pm4.ENDIAN_NONE)
        suite.test(name, shader, fetch_constants={SLOT: (words, name)})
