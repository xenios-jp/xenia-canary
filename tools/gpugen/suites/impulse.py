"""Bilinear filtering of an impulse: a 16x16 texture, 1 at texel (8, 7) and 0
elsewhere, sampled linearly (clamped, base level) from vertex shaders at
normalized U = (7.5 + v / 1024) / 16, V = 7.5 / 16 for v = 0 to 1024: a
phase sweep across the texel's horizontal neighbours.
- 'R8': k_8 bytes.
- 'BC3 alpha': k_DXT4_5, every block's alpha endpoints 255 and 0, index 1
  (0) everywhere but the impulse's texel (255), endian 8in16 as Direct3D's;
  the swizzle puts alpha in X.
Rows are 256 texels (R8) and 32 blocks (BC3) apart, pitches whose rows are
whole multiples of 256 bytes and of 32 blocks, the pitch saying so.
"""

import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gpugen  # noqa: E402
import pm4  # noqa: E402
import xenos as x  # noqa: E402

SLOT = 1
SIDE = 16
QUERIES = 1025
IMPULSE = (8, 7)
LINEAR = (1, 1, 0)
BC3_ROW_BYTES = 32 * 16
f2w = gpugen.f2w


def r8_texture():
    return gpugen.texture_words(*[bytes(255 if (tx, ty) == IMPULSE else 0
                                        for tx in range(SIDE))
                                  for ty in range(SIDE)])


def bc3_texture():
    """Rows of 4 blocks, big-endian 16-bit words: endpoints 255 and 0, then
    the 48-bit indices; the color half zero."""
    rows = []
    for by in range(SIDE // 4):
        words = []
        for bx in range(SIDE // 4):
            indices = 0
            for p in range(16):
                if (bx * 4 + p % 4, by * 4 + p // 4) != IMPULSE:
                    indices |= 1 << (3 * p)
            words += [255, indices & 0xFFFF, (indices >> 16) & 0xFFFF,
                      indices >> 32, 0, 0, 0, 0]
        row = struct.pack(f'>{len(words)}H', *words)
        rows.append(row + bytes(BC3_ROW_BYTES - len(row)))
    return gpugen.texture_words(*rows)


def build(suite):
    words = []
    for v in range(QUERIES):
        words += [f2w((7.5 + v / 1024) / SIDE), f2w(7.5 / SIDE), 0, 0]
    suite.set_inputs(words, 4)
    suite.add_texture('R8', r8_texture())
    suite.add_texture('BC3 alpha', bc3_texture())
    fetch = x.tfetch(2, 1, 'xy', SLOT).serialized()
    shader = gpugen.memexport_shader(
        4, [], [[gpugen.export_copy(2)]],
        fetches=gpugen.fetch_inputs(4) + [fetch])
    alpha = 3 | (3 << 3) | (3 << 6) | (5 << 9)
    for name, fmt, endian, swizzle, pitch in (
            ('R8', 2, pm4.ENDIAN_NONE, gpugen.SWIZZLE_XYZW, 256),
            ('BC3 alpha', 20, pm4.ENDIAN_8IN16, alpha, 128)):
        suite.test(name, shader, fetch_constants={SLOT: (
            gpugen.texture_constant(fmt, SIDE, SIDE, endian=endian,
                                    filters=LINEAR, swizzle=swizzle,
                                    pitch=pitch), name)})
