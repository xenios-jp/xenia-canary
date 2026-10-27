"""cwround's boundary textures with the offsets it meant to have: cwround's
step() moves a float64 by its own last bit, which the float32 store drops,
so all its d rows repeat d = 0. Here d (-4 to 3, pixel i // 256) moves the
float32 itself by d bits (toward +inf). The same draws, formats and raw
Resolves as cwround, for the formats whose rounding the offsets decide.

Each test: '<format> <texture> ulp raw'.
"""

import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpugen  # noqa: E402
import rb  # noqa: E402
import cwround  # noqa: E402


def step32(w, d):
    """Float32 bits w moved d float32 bits toward +inf (finite w)."""
    for _ in range(abs(d)):
        up = d > 0
        if w & 0x7FFFFFFF == 0:
            w = 0x00000001 if up else 0x80000001
        elif (w >> 31) == 0:
            w = w + 1 if up else w - 1
        else:
            w = w - 1 if up else w + 1
            if w == 0x80000000 and up:
                w = 0
    return w


cwround.step = step32

TESTS = [t for t in cwround.TESTS
         if (t[0], t[3], t[4]) in {('8_8_8_8', 'u8', 0),
                                    ('8_8_8_8_GAMMA', 'u10', 0),
                                    ('2_10_10_10', 'u10', 0),
                                    ('2_10_10_10_AS_10_10_10_10', 'u10', 0),
                                    ('16_16_16_16', 'fx', 0),
                                    ('16_16_16_16_FLOAT', 'h', 0),
                                    ('2_10_10_10_FLOAT', '7e3', 0),
                                    ('2_10_10_10_FLOAT_AS_16_16_16_16',
                                     '7e3', 0)}]


def build(suite):
    suite.set_inputs(rb.RECT, 2)
    textures = cwround.textures()
    used = {t[3] for t in TESTS}
    for name in sorted(used):
        suite.add_texture(name, textures[name])
    for name, fmt, bpp, texture, bias in TESTS:
        rb.test(suite, f'{name} {texture} ulp raw', texture,
                rb.registers(fmt),
                gpugen.resolve_registers(rb.WIDTH, rb.HEIGHT,
                                         6 if bpp == 4 else 26, bpp,
                                         command=0))
