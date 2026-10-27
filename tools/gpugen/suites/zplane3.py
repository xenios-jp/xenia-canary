"""Interpolated depth to 2^-28 at 2x and 4x (zplane1's scans): one plane
sloped in X and Y, each sample read back by the Resolve's sample select.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zplane  # noqa: E402
import zplane1  # noqa: E402

PLANE = (32769, 4097)


def build(suite):
    b = zplane.Builder(suite)
    sx, sy = PLANE
    for samples in (2, 4):
        for sample in range(samples):
            zplane.scan(b, f'D24S8 {samples}x sample {sample}', zplane.D24S8,
                        zplane1.TRIANGLE, zplane1.BASE, sx, sy,
                        origin=(-8, -8), samples=samples, sample=sample)
    b.finish()
