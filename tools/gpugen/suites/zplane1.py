"""Interpolated depth to 2^-28, part 1: planes of chosen slopes (in 2^-28
a pixel, so sixteenths of a D24S8 step) over one triangle covering the
target, Z below 1/16 so every vertex Z is exact, each drawn 16 times raised
2^-28 a time (zplane.scan): x slopes either side of 15 significant bits, a
y slope and both.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zplane  # noqa: E402

# A triangle over the whole 64x32 target, its right angle top left.
TRIANGLE = ((-8, -8), (136, -8), (-8, 104))
BASE = 1 << 22
PLANES = [(32769, 0), (32771, 0), (65537, 0), (65538, 0), (65539, 0),
          (0, 32769), (32769, 32769)]


def build(suite):
    b = zplane.Builder(suite)
    for sx, sy in PLANES:
        zplane.scan(b, f'D24S8 {sx} {sy}', zplane.D24S8, TRIANGLE, BASE, sx,
                    sy, origin=(-8, -8))
    b.finish()
