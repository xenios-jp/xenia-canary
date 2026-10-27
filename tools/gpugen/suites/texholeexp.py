"""The EXPAND texture formats over every 16-bit value: 16_EXPAND,
16_16_EXPAND and 16_16_16_16_EXPAND (games read the last signed), unsigned
and signed, point sampled from vertex shaders.

Each texture is 8192x2 texels, linear, endian none (texels little-endian).
Vertex v (16384) point samples texel (v % 8192, v // 8192), unnormalized, and
the test exports the float4 fetched.

Test q (0-3) holds in component 0 the 16-bit values 4 i + q, i = 0-16383 the
texel's number, so four tests cover all 65536; in the others, 16_16 and
16_16_16_16 hold a pseudo-random bijection of them (component k of texel i:
(4 i + q) * (2 k + 1) * 0x9E37 mod 65536 for 16_16 and 16_16_16_16's k = 1-3,
rotated by 5 k bits).
"""

import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gpugen  # noqa: E402
import pm4  # noqa: E402
import xenos as x  # noqa: E402

SLOT = 1
WIDTH, ROWS = 8192, 2
VERTICES = WIDTH * ROWS
UNSIGNED, SIGNED = 0, 1
f2w = gpugen.f2w


def inputs():
    words = []
    for v in range(VERTICES):
        words += [f2w(v % WIDTH + .5), f2w(v // WIDTH + .5), 0, 0]
    return words


def component(i, q, k):
    value = 4 * i + q
    if k:
        value = value * (2 * k + 1) * 0x9E37 & 0xFFFF
        value = (value << 5 * k | value >> 16 - 5 * k) & 0xFFFF
    return value


def texture(components, q):
    data = b''.join(struct.pack('<%dH' % components,
                                *(component(i, q, k)
                                  for k in range(components)))
                    for i in range(VERTICES))
    return list(struct.unpack(f'>{len(data) // 4}I', data))


def build(suite):
    suite.set_inputs(inputs(), 4)
    move = gpugen.export_copy(3)
    vertex = gpugen.fetch_inputs(4)
    fetch = x.tfetch(3, 1, 'xy', SLOT, denormalized=True).serialized()
    shader = gpugen.memexport_shader(4, [], [[move]],
                                     fetches=vertex + [fetch])

    def test(name, texture_name, fmt, sign):
        words = gpugen.texture_constant(
            fmt, WIDTH, ROWS, sign=sign, pitch=WIDTH, endian=pm4.ENDIAN_NONE)
        suite.test(name, shader,
                   fetch_constants={SLOT: (words, texture_name)})

    for fname, fmt, components in (('16_EXPAND', 27, 1),
                                   ('16_16_EXPAND', 28, 2),
                                   ('16_16_16_16_EXPAND', 29, 4)):
        for q in range(4):
            suite.add_texture(f'{fname} q{q}', texture(components, q))
            test(f'{fname} signed q{q}', f'{fname} q{q}', fmt, SIGNED)
            if components == 1 or q == 0:
                test(f'{fname} unsigned q{q}', f'{fname} q{q}', fmt,
                     UNSIGNED)
