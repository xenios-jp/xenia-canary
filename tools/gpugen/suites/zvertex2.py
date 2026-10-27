"""Perspective depth as games draw it: vertices with different W, so the
depth each pixel gets depends on the per-vertex divide, the setup and the
interpolation together. Clip Z = A W - B, a near 1, far 1000 projection
(window depth A - B / W), the shape a game's vertex shader gives; the
vertices are given post-transform (passthrough vertex shader, no shader
arithmetic). Depth test LEQUAL over a target cleared to the farthest
depth, as Metro: Last Light's scene pass draws; raw depth readback.

- 'big': one triangle over the whole target, W 1 .. 64 per vertex.
- 'mesh': 8x4 cells of 8x8 pixels, two triangles each, sharing vertices
  jittered off the 1/16 grid, W 1 .. 64 per vertex.
- 'mesh flat': the mesh at W 1 (no perspective), the same window depths.
Each in D24S8 and D24FS8.
"""

import os
import random
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zplane  # noqa: E402

NEAR, FAR = 1.0, 1000.0
A = FAR / (FAR - NEAR)
B = FAR * NEAR / (FAR - NEAR)
LEQUAL = 3
FARTHEST = 0xFFFFFF00


def f32(v):
    return struct.unpack('>f', struct.pack('>f', v))[0]


def vertex(x, y, w):
    """Window (x, y) at W w: clip (x w, y w, A w - B, w) as floats."""
    w = f32(w)
    return (f32(x * w), f32(y * w), f32(f32(A * w) - B), w)


def big(rng):
    xy = [(rng.uniform(-24, -8), rng.uniform(-24, -8)),
          (rng.uniform(136, 160), rng.uniform(-24, -8)),
          (rng.uniform(-24, -8), rng.uniform(104, 128))]
    rng.shuffle(xy)
    return [vertex(x, y, rng.uniform(1, 64)) for x, y in xy]


def mesh(rng, flat):
    cols, rows, size = 8, 4, 8
    grid = {}
    for j in range(rows + 1):
        for i in range(cols + 1):
            x = i * size - 0.5 + (rng.uniform(-2, 2) if 0 < i < cols else
                                  (-2 if i == 0 else 2))
            y = j * size - 0.5 + (rng.uniform(-2, 2) if 0 < j < rows else
                                  (-2 if j == 0 else 2))
            w = rng.uniform(1, 64)
            if flat:
                # The same window depth at W 1.
                grid[i, j] = (f32(x), f32(y), f32(A - B / f32(w)), 1.0)
            else:
                grid[i, j] = vertex(x, y, w)
    out = []
    for j in range(rows):
        for i in range(cols):
            a, b_, c, d = (grid[i, j], grid[i + 1, j], grid[i, j + 1],
                           grid[i + 1, j + 1])
            out += [a, b_, c, c, b_, d] if (i + j) % 2 else [a, b_, d, a, d, c]
    return out


def build(suite):
    b = zplane.Builder(suite)
    for fmt, fname in enumerate(zplane.FORMATS):
        rng = random.Random(0x2E72)
        for k in range(8):
            b.test(f'{fname} big {k}', fmt,
                   [zplane.draw(big(rng), compare=LEQUAL)], clear=FARTHEST)
        for k in range(4):
            state = rng.getstate()
            b.test(f'{fname} mesh {k}', fmt,
                   [zplane.draw(mesh(rng, False), compare=LEQUAL)],
                   clear=FARTHEST)
            rng.setstate(state)
            b.test(f'{fname} mesh flat {k}', fmt,
                   [zplane.draw(mesh(rng, True), compare=LEQUAL)],
                   clear=FARTHEST)
    b.finish()
