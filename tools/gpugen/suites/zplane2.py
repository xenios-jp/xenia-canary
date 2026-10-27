"""Interpolated depth to 2^-28, part 2 (zplane1's scans): fractional and
negative slopes, the part 1 plane through its triangle's vertices in other
orders, through a triangle with its right angle top right, and through two
triangles splitting the target.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zplane  # noqa: E402
import zplane1  # noqa: E402

T = zplane1.TRIANGLE
ROTATED = (T[1], T[2], T[0])
REVERSED = (T[0], T[2], T[1])
MIRRORED = ((-72, -8), (72, -8), (72, 72))
SPLIT = ((-8, -8), (72, -8), (-8, 40), (-8, 40), (72, -8), (72, 40))


def build(suite):
    b = zplane.Builder(suite)
    o = (-8, -8)
    for name, xy, base, sx, sy in (
            ('fraction', T, 1 << 22, 4097.5, 0),
            ('negative', T, 3 << 22, -32769, 0),
            ('general', T, 1 << 22, 1777761 / 144, -262718 / 112),
            ('rotated', ROTATED, 1 << 22, 32769, 0),
            ('reversed', REVERSED, 1 << 22, 32769, 0),
            ('mirrored', MIRRORED, 1 << 23, 32769, 0),
            ('split', SPLIT, 1 << 22, 32769, 0)):
        zplane.scan(b, f'D24S8 {name}', zplane.D24S8, xy, base, sx, sy,
                    origin=o)
    b.finish()
