"""Stacked 2D and 3D textures from vertex shaders: which slice and texel a
3D-dimension tfetch reads for each w, normalized and not, at slice edges,
with clamping and wrapping in w, and how linear filtering blends slices.

The textures describe their own memory: 8x8 texels, 8 slices, pitch 32
texels, over 8 blocks of 4 KiB (Xenia's slice stride for this size), where
dword d holds
- 'addr' (k_32_FLOAT, point sampled): d;
- 'tags' (k_8_8_8_8): bytes (8 (d % 32), 8 (d // 32 % 32), 32 (d // 1024),
  0xFF): column, row and 4 KiB block.

Vertex v (1024):
- r1 = (u, v, w, 0) normalized: v < 512: w = (v // 4 - 16) / 96 at
  u = 0.3 + 0.1 (v % 4), v = 0.55; v < 768: slice edges w = n / 8 +
  (-2^-21, 0, 2^-21, 2^-12)[v % 4], n = (v - 512) // 4 % 10 - 1, at
  (0.4, 0.4); else w = 0.5, u = (v - 768) / 128 - 0.5, v = 0.3.
- r2 = (x, y, z, 0) unnormalized: x = 2.5 + v % 4, y = 3.5,
  z = -1 + (v // 4) / 24.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gpugen  # noqa: E402
import xenos as x  # noqa: E402

SLOT = 1
VERTICES = 1024
SIDE = 8
SLICES = 8
PITCH = 32
BLOCK_DWORDS = 1024
F32, F8888 = 36, 6
POINT, LINEAR = (0, 0, 0), (1, 1, 0)
DIMENSION_3D = 2 << 9
STACKED = 1 << 10  # dword 1
EDGE = (-2.0 ** -21, 0.0, 2.0 ** -21, 2.0 ** -12)
f2w = gpugen.f2w


def inputs():
    words = []
    for v in range(VERTICES):
        if v < 512:
            u, t, w = 0.3 + 0.1 * (v % 4), 0.55, (v // 4 - 16) / 96
        elif v < 768:
            n = (v - 512) // 4 % 10 - 1
            u, t, w = 0.4, 0.4, n / SLICES + EDGE[v % 4]
        else:
            u, t, w = (v - 768) / 128 - 0.5, 0.3, 0.5
        words += [f2w(u), f2w(t), f2w(w), 0,
                  f2w(2.5 + v % 4), f2w(3.5), f2w(-1 + (v // 4) / 24), 0]
    return words


def addr():
    return [f2w(float(d)) for d in range(SLICES * BLOCK_DWORDS)]


def tags():
    return [8 * (d % 32) | 8 * (d // 32 % 32) << 8 | 32 * (d // 1024) << 16 |
            0xFF << 24 for d in range(SLICES * BLOCK_DWORDS)]


def constant(kind, fmt, filters, clamp, volume_linear=False):
    words = gpugen.texture_constant(fmt, SIDE, SIDE, filters=filters,
                                    clamp=clamp, pitch=PITCH)
    if kind == 'stacked':
        words[1] |= STACKED
        words[2] |= (SLICES - 1) << 26
    else:
        words[2] = (SIDE - 1) | (SIDE - 1) << 11 | (SLICES - 1) << 22
        words[5] = (words[5] & ~(3 << 9)) | DIMENSION_3D
    if volume_linear:
        words[4] |= 3  # vol_mag_filter, vol_min_filter
    return words


def build(suite):
    suite.set_inputs(inputs(), 8)
    suite.add_texture('addr', addr())
    suite.add_texture('tags', tags())
    vertex = gpugen.fetch_inputs(8)

    def test(name, kind, texture, fmt, filters, clamp, source=1,
             denormalized=False, volume_linear=False):
        fetch = x.tfetch(5, source, 'xyz', SLOT, dimension=2,
                         denormalized=denormalized).serialized()
        suite.test(name, gpugen.memexport_shader(
            8, [], [[gpugen.export_copy(5)]], fetches=vertex + [fetch]),
            fetch_constants={SLOT: (constant(kind, fmt, filters, clamp,
                                             volume_linear), texture)})

    for kind in ('stacked', '3d'):
        test(f'{kind} point', kind, 'addr', F32, POINT, 2)
        test(f'{kind} point unnormalized', kind, 'addr', F32, POINT, 2,
             source=2, denormalized=True)
        test(f'{kind} point wrap', kind, 'addr', F32, POINT, (2, 2, 0))
        test(f'{kind} linear', kind, 'tags', F8888, LINEAR, 2,
             volume_linear=True)
