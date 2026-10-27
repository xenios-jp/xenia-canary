"""Depth Resolves to other destination formats: the same 2048 flat pixels
of known Z (zplane.pixel_quads) in D24S8 and D24FS8, then the depth buffer
resolved raw (copy command 0) and converting (1) to k_8_8_8_8 (Direct3D's),
k_24_8, k_24_8_FLOAT, k_32_FLOAT, k_16, k_16_16, k_16_16_FLOAT and
k_32_32_FLOAT, float formats with number format 7 too.
"""

import os
import random
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpugen  # noqa: E402
import zplane  # noqa: E402

# (name, texture format, bytes, number formats)
DESTS = [('8_8_8_8', 6, 4, (0,)), ('24_8', 22, 4, (0,)),
         ('24_8_FLOAT', 23, 4, (0, 7)), ('32_FLOAT', 36, 4, (0, 7)),
         ('16', 24, 2, (0,)), ('16_16', 25, 4, (0,)),
         ('16_16_FLOAT', 31, 4, (0, 7)), ('32_32_FLOAT', 37, 8, (0, 7))]


def f32(v):
    return struct.unpack('>f', struct.pack('>f', v))[0]


def zs(rng):
    out = [0.0, 1.0, 0.5, 0.25, 2.0 ** -14, 2.0 ** -15, 2.0 ** -24,
           f32(1 / 3), f32(2 / 3), f32(0.75), f32(1 - 2 ** -24), 2.0 ** -34]
    while len(out) < zplane.WIDTH * zplane.HEIGHT:
        out.append(f32(rng.random()) if len(out) % 2 else
                   f32(2.0 ** -rng.uniform(0, 24)))
    return out


def build(suite):
    rng = random.Random(0x24F8)
    b = zplane.Builder(suite)
    quads = zplane.pixel_quads([(z, 1.0) for z in zs(rng)])
    for fmt, fname in enumerate(zplane.FORMATS):
        for command in (0, 1):
            for dname, dest, bpp, numbers in DESTS:
                for number in numbers:
                    resolve = gpugen.resolve_registers(
                        zplane.WIDTH, zplane.HEIGHT, dest, bpp, number,
                        command=command, source=4)
                    b.test(f'{fname} command {command} {dname} number '
                           f'{number}', fmt, [zplane.draw(quads)],
                           resolve=resolve, dest=(dest, bpp))
    b.finish()
