"""Texture LOD from vertex shaders, finely: the mip blend weight a register
LOD and isotropic gradients give (so their precision), LOD biases, and the
fetch constant's exp_adjust on fetched values.

'lvl' is a 64x64 k_8_8_8_8 texture with levels 0-6, not packed (each level
from 1 on 2048 dwords apart under the mip address, as tmips found): every
texel of level k is (32 k, 255 - 32 k, 32 k, 255), so with mip filtering
linear (min and mag point) z is 32 (k + f) for the blend f between levels
k and k + 1. 'mid' is a 64x64 base level of (0x80, 0x40, 0x01, 0xFF).

Vertex v (1024): r1 = (0.3, 0.6, -1 + v / 128, 0): LODs -1 to 7 in steps
of 1/128. r2 = isotropic gradients (g, 0, 0, g), g = 2^(-1 + v / 128) / 64:
LOD log2 of g in base texels, the same sweep.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gpugen  # noqa: E402
import xenos as x  # noqa: E402

SLOT = 1
VERTICES = 1024
SIDE = 64
LEVELS = (0, 6)
MIP_SPACING = 2048
F8888 = 6
POINT, LINEAR = 0, 1
TRILINEAR_POINT = (POINT, POINT, LINEAR)
f2w = gpugen.f2w


def inputs():
    words = []
    for v in range(VERTICES):
        lod = -1 + v / 128
        g = 2.0 ** lod / SIDE
        words += [f2w(0.3), f2w(0.6), f2w(lod), 0, f2w(g), 0, 0, f2w(g)]
    return words


def tag(level):
    t = 32 * level
    return t | (255 - t) << 8 | t << 16 | 0xFF << 24


def build(suite):
    suite.set_inputs(inputs(), 8)
    suite.add_texture('lvl base', [tag(0)] * (SIDE * SIDE))
    suite.add_texture('lvl mips', [tag(1 + d // MIP_SPACING)
                                   for d in range(6 * MIP_SPACING)])
    suite.add_texture('mid', [0x80 | 0x40 << 8 | 0x01 << 16 | 0xFF << 24] *
                      (SIDE * SIDE))
    vertex = gpugen.fetch_inputs(8)
    move = gpugen.export_copy(3)

    def test(name, texture, filters=TRILINEAR_POINT, gradients=False,
             fetch=None, mips=LEVELS, **constant):
        words = gpugen.texture_constant(F8888, SIDE, SIDE, filters=filters,
                                        pitch=SIDE, mips=mips, **constant)
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
                   fetch_constants={SLOT: (words, texture)})

    lvl = ('lvl base', 'lvl mips')
    test('register lod', lvl)
    test('register lod tri_clamp 1', lvl, tri_clamp=1)
    test('gradients', lvl, gradients=True)
    test('gradients mag linear min linear', lvl, (LINEAR, LINEAR, LINEAR),
         gradients=True)
    test('register lod const bias 7', lvl, lod_bias=7)
    test('register lod const bias -45', lvl, lod_bias=-45)
    test('register lod fetch bias 3', lvl, fetch=dict(lod_bias=3))
    test('gradients const bias -16', lvl, gradients=True, lod_bias=-16)
    for adjust in (1, -1, 3, -4):
        test(f'exp adjust {adjust}', 'mid', (POINT, POINT, POINT),
             mips=(0, 0), exp_adjust=adjust)
    test('exp adjust -4 signed', 'mid', (POINT, POINT, POINT), mips=(0, 0),
         exp_adjust=-4, sign=1)
