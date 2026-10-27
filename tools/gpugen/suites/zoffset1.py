"""Constant polygon offsets on flat Z: column c of the target is one draw
with front and back offset O_c (scales 0), row r a pixel at Z z_r
(zplane.pixel_quads), so each word is z_r + O_c as the depth stage adds
them.

- 'fine': O = k 2^-28 for k -32 .. 31; rows 0-15 Z in [1/32, 1/16) on
  every sixteenth of a step (floor(Z 2^28) mod 16 = r), rows 16-31 in
  [0.5, 1).
- 'mantissa': O random over 2^-26 .. 2^-4 of both signs; Z random.
- 'wrap': O near and past +-8, +-16 and beyond; Z random.
Each in D24S8 and D24FS8.
"""

import os
import random
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zplane  # noqa: E402

W, H = zplane.WIDTH, zplane.HEIGHT


def f32(v):
    return struct.unpack('>f', struct.pack('>f', v))[0]


def column(c, zs):
    out = []
    for r, z in enumerate(zs):
        quad = zplane.pixel_quads([(z, 1.0)])
        out += [(x + c, y + r, z, w) for x, y, z, w in quad]
    return out


def fine_rows(rng):
    rows = []
    for r in range(16):
        n = (1 << 23) + (rng.randrange(1 << 19) << 4) + r
        rows.append(n * 2.0 ** -28)
    rows += [f32(rng.uniform(0.5, 1)) for _ in range(16)]
    return rows


WRAP = [8.0, -8.0, f32(8 - 2 ** -20), f32(-8 + 2 ** -20), f32(8 + 2 ** -19),
        f32(-8 - 2 ** -19), 16.0, -16.0, f32(16 - 2 ** -19),
        f32(-16 + 2 ** -19), f32(16 + 2 ** -18), f32(-16 - 2 ** -18), 24.0,
        -24.0, 32.0, -32.0, 48.0, -48.0, 64.0, -64.0, 1024.0, -1024.0,
        f32(1e5), f32(-1e5), 2.0 ** 24, -2.0 ** 24, 2.0 ** 30, -2.0 ** 30,
        2.0 ** 31, -2.0 ** 31, 2.0 ** 40, -2.0 ** 40, f32(7.5), f32(-7.5),
        f32(15.5), f32(-15.5), f32(-16.25), f32(-15.75), f32(-17.0),
        f32(-31.0), f32(-12.0), f32(-4.0), f32(-1.0), f32(-0.5),
        f32(-16 + 2 ** -24 * 3), f32(-16 + 2 ** -23), f32(-16 + 2 ** -22),
        f32(-16 + 2 ** -21), f32(-32 + 2 ** -18), f32(-8 + 2 ** -21),
        f32(-8 + 2 ** -22), f32(-8 + 2 ** -23), f32(-4 + 2 ** -22),
        f32(-4 + 2 ** -23), f32(-4 + 2 ** -24), f32(-2 + 2 ** -23),
        f32(-2 + 2 ** -24), f32(-2 + 2 ** -25), f32(-1 + 2 ** -24),
        f32(-1 + 2 ** -25), f32(-1 + 2 ** -26), f32(-0.5 + 2 ** -26),
        f32(-0.5 + 2 ** -27), f32(-0.25 + 2 ** -28)]


def build(suite):
    rng = random.Random(0x0FF5E7)
    b = zplane.Builder(suite)
    sets = []
    sets.append(('fine', [k * 2.0 ** -28 for k in range(-32, 32)],
                 fine_rows(rng)))
    sets.append(('mantissa', [f32(rng.choice((1, -1)) * rng.uniform(1, 2) *
                                  2.0 ** rng.randint(-26, -4))
                              for _ in range(W)],
                 [f32(rng.uniform(0.02, 0.98)) for _ in range(H)]))
    assert len(WRAP) == W
    sets.append(('wrap', WRAP, [f32(rng.uniform(0.02, 0.98))
                                for _ in range(H)]))
    for name, offsets, zs in sets:
        for fmt, fname in enumerate(zplane.FORMATS):
            draws = [zplane.draw(column(c, zs), offset=(0.0, o, 0.0, o))
                     for c, o in enumerate(offsets)]
            b.test(f'{fname} {name}', fmt, draws)
    b.finish()
