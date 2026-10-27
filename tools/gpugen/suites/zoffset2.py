"""Slope-scaled polygon offsets: planes of chosen slopes over zplane1's
triangle, each drawn without an offset and with front and back scales S
(offsets 0), so the difference per pixel is the offset the slope makes:
which slope measure, its unit (S in 1/16 pixel), signs and faces.

- 'scale': 7 planes x S in 16, 1, 7, 100, -16, 0.5, 1600 (D24S8).
- 'face': one plane in both vertex orders, front scale 16 and back 64,
  then front 64 and back 16.
- D24FS8: 2 planes x S 16 and 100.
"""

import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zplane  # noqa: E402
import zplane1  # noqa: E402

T = zplane1.TRIANGLE
PLANES = [(0.5, 0.002, 0.001), (0.5, 0.001, 0.002), (0.6, -0.002, 0.001),
          (0.4, 0.0015, -0.0015), (0.3, 0.003, 0.0), (0.7, 0.0, -0.003),
          (0.45, 1e-5, 3e-6)]
SCALES = [16.0, 1.0, 7.0, 100.0, -16.0, 0.5, 1600.0]


def f32(v):
    return struct.unpack('>f', struct.pack('>f', v))[0]


def vertices(plane, xy=T):
    z0, a, b = plane
    return [(float(x), float(y), f32(z0 + a * (x - 28) + b * (y - 16)), 1.0)
            for x, y in xy]


def build(suite):
    b = zplane.Builder(suite)
    for p, plane in enumerate(PLANES):
        v = vertices(plane)
        b.test(f'D24S8 plane {p}', zplane.D24S8, [zplane.draw(v)])
        for s in SCALES:
            b.test(f'D24S8 plane {p} scale {s:g}', zplane.D24S8,
                   [zplane.draw(v, offset=(s, 0.0, s, 0.0))])
    v = vertices(PLANES[0])
    for order, xy in (('cw', v), ('ccw', [v[0], v[2], v[1]])):
        for front, back in ((16.0, 64.0), (64.0, 16.0)):
            b.test(f'D24S8 face {order} front {front:g} back {back:g}',
                   zplane.D24S8,
                   [zplane.draw(xy, offset=(front, 0.0, back, 0.0))])
        b.test(f'D24S8 face {order} none', zplane.D24S8, [zplane.draw(xy)])
    for p in (0, 3):
        v = vertices(PLANES[p])
        b.test(f'D24FS8 plane {p}', zplane.D24FS8, [zplane.draw(v)])
        for s in (16.0, 100.0):
            b.test(f'D24FS8 plane {p} scale {s:g}', zplane.D24FS8,
                   [zplane.draw(v, offset=(s, 0.0, s, 0.0))])
    b.finish()
