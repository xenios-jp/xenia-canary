"""Per-vertex depth: one flat pixel per case, so its word is the depth the
vertex stage made of the vertex's Z, W and the viewport's Z scale and
offset (zplane.pixel_quads; X and Y are multiplied by W as well).

- 'w': Z = c W for c in [0.5, 1) (D24S8 there keeps every float), W over
  2^-6 .. 2^20, powers of two, near them and random, and c over [0, 2)
  below and above.
- 'viewport': W 1, Z over [0, 1], through Z scale and offset pairs.
- 'both': the 'w' cases through a Z scale and offset.
Each in D24S8 and D24FS8.
"""

import os
import random
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zplane  # noqa: E402

N = zplane.WIDTH * zplane.HEIGHT


def f32(v):
    return struct.unpack('>f', struct.pack('>f', v))[0]


def w_cases(rng):
    out = []
    for i in range(N):
        if i < 512:
            w = 2.0 ** rng.randint(-6, 20)
            w = f32(w * (1 + rng.choice((0, 2 ** -23, -2 ** -24, 2 ** -22,
                                         3 * 2 ** -23))))
        else:
            w = f32(rng.uniform(1, 2) * 2.0 ** rng.randint(-6, 20))
        if i % 8 == 7:
            c = rng.uniform(0, 0.5)
        elif i % 8 == 6:
            c = rng.uniform(1, 2)
        else:
            c = rng.uniform(0.5, 1)
        out.append((f32(c * w), w))
    return out


def unit_cases(rng):
    return [(f32(rng.random()), 1.0) for _ in range(N)]


VIEWPORTS = [(0.5, 0.5), (-1.0, 1.0), (0.25, 0.75), (1.0, 2.0 ** -24),
             (f32(0.999), f32(0.0005)), (f32(1 / 3), f32(1 / 3))]


def build(suite):
    rng = random.Random(0x2E7)
    b = zplane.Builder(suite)
    ws, units = w_cases(rng), unit_cases(rng)
    for fmt, fname in enumerate(zplane.FORMATS):
        b.test(f'{fname} w', fmt, [zplane.draw(zplane.pixel_quads(ws))])
        for scale, offset in VIEWPORTS:
            b.test(f'{fname} viewport {scale!r} {offset!r}', fmt,
                   [zplane.draw(zplane.pixel_quads(units))],
                   viewport=(1.0, 0.0, 1.0, 0.0, scale, offset))
        b.test(f'{fname} both', fmt, [zplane.draw(zplane.pixel_quads(ws))],
               viewport=(1.0, 0.0, 1.0, 0.0, 0.5, 0.25))
    b.finish()
