"""Mipmapped textures from vertex shaders: which level and texel a fetch
reads at an explicit LOD (setTexLOD) or from explicit gradients
(setGradientH/V), and how mip filtering, the fetch constant's mip range,
packed mip tails, LOD biases, tri_clamp and anisotropic filtering blend them.

The textures describe their own memory, so the results show the layout the
GPU reads as well as the levels: under the base address and the mip address
alike, dword d of the texture's blob holds
- 'addr' (k_32_FLOAT, point sampled): d, plus 1048576 under the mip
  address;
- 'tags' (k_8_8_8_8): bytes (d & 0xFF, (d >> 8) & 0xFF, d >> 16 | 0x40
  under the base or 0xC0 under the mip address, 0xFF) in x, y, z, w order,
  so a filtered x is a fraction between horizontally adjacent texels.

Vertex v (256): r1 = (u, v, lod, 0), lod = -1 + (v % 96) / 8 at the point
(0.3, 0.6), (0.53, 0.27) or (0.9, 0.12) by v // 96; r2 = the horizontal
and vertical gradients (normalized): a major axis 2^(v // 16 % 6) times the
minor one, minor 2^((v % 16) / 2 - 1) texels of 64, the major axis along u,
along v or at 30 degrees by v // 96.
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gpugen  # noqa: E402
import xenos as x  # noqa: E402

SLOT = 1
VERTICES = 256
POINTS = ((0.3, 0.6), (0.53, 0.27), (0.9, 0.12))
BASE_DWORDS, MIP_DWORDS = 0x4000, 0x8000  # 64 and 128 KiB
MIP_TAG = 1 << 20
F32, F8888 = 36, 6
POINT, LINEAR, BASEMAP = 0, 1, 2
f2w = gpugen.f2w


def inputs():
    words = []
    for v in range(VERTICES):
        u, w = POINTS[v // 96]
        minor = 2.0 ** ((v % 16) / 2 - 1) / 64
        major = minor * 2.0 ** (v // 16 % 6)
        angle = (0.0, math.pi / 2, math.pi / 6)[v // 96]
        c, s = math.cos(angle), math.sin(angle)
        words += [f2w(u), f2w(w), f2w(-1 + (v % 96) / 8), 0,
                  f2w(major * c), f2w(major * s), f2w(-minor * s),
                  f2w(minor * c)]
    return words


def tags(dwords, tag):
    return [(d & 0xFF) | ((d >> 8) & 0xFF) << 8 | ((d >> 16) | tag) << 16 |
            0xFF << 24 for d in range(dwords)]


def build(suite):
    suite.set_inputs(inputs(), 8)
    suite.add_texture('addr base', [f2w(float(d)) for d in range(BASE_DWORDS)])
    suite.add_texture('addr mips', [f2w(float(MIP_TAG + d))
                                    for d in range(MIP_DWORDS)])
    suite.add_texture('tags base', tags(BASE_DWORDS, 0x40))
    suite.add_texture('tags mips', tags(MIP_DWORDS, 0xC0))
    vertex = gpugen.fetch_inputs(8)
    move = gpugen.export_copy(3)

    def test(name, texture, fmt, size, levels, filters=(POINT, POINT, POINT),
             gradients=False, fetch=None, **constant):
        width, height = size
        pitch = max(width, 64)
        words = gpugen.texture_constant(
            fmt, width, height, filters=filters, pitch=pitch,
            mips=levels, **constant)
        keywords = dict(use_reg_gradients=True) if gradients else dict(
            use_reg_lod=True, use_comp_lod=False)
        keywords.update(fetch or {})
        if gradients:
            sets = [x.set_texture_state(x.FETCH_SET_GRADIENTS_HORZ, 2,
                                        'xy').serialized(),
                    x.set_texture_state(x.FETCH_SET_GRADIENTS_VERT, 2, 'zw')]
        else:
            sets = [x.set_texture_state(x.FETCH_SET_LOD, 1, 'z').serialized()]
        fetches = vertex + sets + [x.tfetch(3, 1, 'xy', SLOT, **keywords)]
        suite.test(name, gpugen.memexport_shader(8, [], [[move]],
                                                 fetches=fetches),
                   fetch_constants={SLOT: (words, (f'{texture} base',
                                                   f'{texture} mips'))})

    # Which level and texel each LOD reads, point sampled.
    for size in ((64, 64), (128, 32), (32, 32), (16, 16), (64, 16)):
        levels = (0, int(math.log2(max(size))))
        name = f'{size[0]}x{size[1]}'
        test(f'addr {name}', 'addr', F32, size, levels)
        test(f'addr {name} packed', 'addr', F32, size, levels,
             packed_mips=True)
    square = (64, 64)
    for levels in ((1, 5), (2, 4), (0, 9), (3, 3)):
        test(f'addr levels {levels[0]}-{levels[1]} packed', 'addr', F32,
             square, levels, packed_mips=True)
    for bias in (-16, -32, 40, -512):
        test(f'addr const lod bias {bias}', 'addr', F32, square, (0, 6),
             packed_mips=True, lod_bias=bias)
    for bias in (-8, 16, -64):
        test(f'addr fetch lod bias {bias}', 'addr', F32, square, (0, 6),
             packed_mips=True, fetch=dict(lod_bias=bias))
    test('addr basemap', 'addr', F32, square, (0, 6),
         (POINT, POINT, BASEMAP), packed_mips=True)
    test('addr computed lod', 'addr', F32, square, (0, 6), packed_mips=True,
         fetch=dict(use_reg_lod=False, use_comp_lod=True))
    test('addr computed and register lod', 'addr', F32, square, (0, 6),
         packed_mips=True, fetch=dict(use_reg_lod=True, use_comp_lod=True))

    # Filtering between texels and levels.
    trilinear = (LINEAR, LINEAR, LINEAR)
    for clamp in range(4):
        test(f'trilinear tri_clamp {clamp}', 'tags', F8888, square, (0, 6),
             trilinear, packed_mips=True, tri_clamp=clamp)
    test('trilinear unpacked', 'tags', F8888, square, (0, 6), trilinear)
    test('bilinear mip point', 'tags', F8888, square, (0, 6),
         (LINEAR, LINEAR, POINT), packed_mips=True)
    test('mag linear min point', 'tags', F8888, square, (0, 6),
         (LINEAR, POINT, POINT), packed_mips=True)
    test('mag point min linear', 'tags', F8888, square, (0, 6),
         (POINT, LINEAR, LINEAR), packed_mips=True)
    test('trilinear fetch override', 'tags', F8888, square, (0, 6),
         (POINT, POINT, POINT), packed_mips=True,
         fetch=dict(mag_filter=LINEAR, min_filter=LINEAR,
                    mip_filter=LINEAR))

    # Gradients: the LOD they give, isotropic and anisotropic.
    test('addr gradients', 'addr', F32, square, (0, 6), gradients=True,
         packed_mips=True)
    for aniso in range(6):
        test(f'addr gradients aniso {aniso}', 'addr', F32, square, (0, 6),
             gradients=True, packed_mips=True, aniso=aniso)
        test(f'gradients trilinear aniso {aniso}', 'tags', F8888, square,
             (0, 6), trilinear, gradients=True, packed_mips=True,
             aniso=aniso)
    test('gradients trilinear aniso fetch 4', 'tags', F8888, square, (0, 6),
         trilinear, gradients=True, packed_mips=True,
         fetch=dict(aniso_filter=3))
    test('gradients trilinear aniso 4 bias 4', 'tags', F8888, square, (0, 6),
         trilinear, gradients=True, packed_mips=True, aniso=3, aniso_bias=4)
    test('addr gradients exp adjust 1 -1', 'addr', F32, square, (0, 6),
         gradients=True, packed_mips=True, grad_exp_adjust=(1, -1))
