"""Gradient LOD from vertex shaders, finely: which vertex of each group of
four gives the group's LOD, the norm of a rotated gradient, log2's
precision over one octave, and the anisotropic LOD for ratios that aren't
powers of two, with anisotropy biases.

'lvl' is texlod's: a 64x64 k_8_8_8_8 texture with levels 0-6, not packed
(levels from 1 on 2048 dwords apart under the mip address), every texel of
level k (32 k, 255 - 32 k, 32 k, 255), filtered trilinearly, so z is
32 (k + f) for the blend f between levels k and k + 1.

Vertex v (1024), q = v // 4 its group of four, l = v % 4 its place in it;
r1 = (0.3, 0.6, 0, 0); gradients in texels of 64, H = (hx, hy) and
V = (vx, vy) as r = (hx, hy, vx, vy):
- r2 (lanes): isotropic, H = (g, 0), V = (0, g), g = 2^L with L = 1 +
  PERMS[q % 24][l] / 4 + (q // 24 % 4) / 64; groups from 96 on put L - 1
  in one lane only (lane q % 4) and 1 + 1 / 2 in the others.
- r3 (angle): H = 2 (cos a, sin a), V = 2 (-sin a, cos a), a = q / 255 of
  90 degrees, the same in every lane.
- r4 (h only): H = 2 (cos a, sin a), V = (0, 0), a as for r3.
- r5 (octave): isotropic, g = 2^(1 + q / 256).
- r6 (ratio): H = (2 (1 + q / 32), 0), V = (0, 2): ratios 1 to 9 at
  LOD 1 along the minor axis.
"""

import itertools
import math
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
LINEAR = 1
TRILINEAR = (LINEAR, LINEAR, LINEAR)
PERMS = list(itertools.permutations(range(4)))
f2w = gpugen.f2w


def lanes_lod(q, l):
    if q < 96:
        return 1 + PERMS[q % 24][l] / 4 + (q // 24 % 4) / 64
    return 0.0 if l == q % 4 else 1.5


def inputs():
    words = []
    for v in range(VERTICES):
        q, l = divmod(v, 4)
        g = 2.0 ** lanes_lod(q, l)
        a = math.pi / 2 * q / 255
        c, s = math.cos(a), math.sin(a)
        o = 2.0 ** (1 + q / 256)
        r = 2.0 * (1 + q / 32)
        rows = [(0.3, 0.6, 0, 0), (g, 0, 0, g), (2 * c, 2 * s, -2 * s, 2 * c),
                (2 * c, 2 * s, 0, 0), (o, 0, 0, o), (r, 0, 0, 2.0)]
        for i, row in enumerate(rows):
            # Gradients are normalized: texels of 64.
            words += [f2w(e if i == 0 else e / SIDE) for e in row]
    return words


def tag(level):
    t = 32 * level
    return t | (255 - t) << 8 | t << 16 | 0xFF << 24


def build(suite):
    suite.set_inputs(inputs(), 24)
    suite.add_texture('lvl base', [tag(0)] * (SIDE * SIDE))
    suite.add_texture('lvl mips', [tag(1 + d // MIP_SPACING)
                                   for d in range(6 * MIP_SPACING)])
    vertex = gpugen.fetch_inputs(24)
    move = gpugen.export_copy(7)

    def test(name, grad, aniso=0, **constant):
        words = gpugen.texture_constant(F8888, SIDE, SIDE, filters=TRILINEAR,
                                        pitch=SIDE, mips=LEVELS, aniso=aniso,
                                        **constant)
        sets = [x.set_texture_state(x.FETCH_SET_GRADIENTS_HORZ, grad,
                                    'xy').serialized(),
                x.set_texture_state(x.FETCH_SET_GRADIENTS_VERT, grad, 'zw')]
        fetches = vertex + sets + [x.tfetch(7, 1, 'xy', SLOT,
                                            use_reg_gradients=True)]
        suite.test(name, gpugen.memexport_shader(24, [], [[move]],
                                                 fetches=fetches),
                   fetch_constants={SLOT: (words, ('lvl base', 'lvl mips'))})

    test('lanes', 2)
    test('angle', 3)
    test('h only', 4)
    test('octave', 5)
    for aniso in (2, 3):
        test(f'ratio aniso {aniso}', 6, aniso)
    for bias in (-8, -4, -1, 1, 4, 7):
        test(f'ratio aniso 3 bias {bias}', 6, 3, aniso_bias=bias)
    test('angle aniso 3', 3, 3)
    test('h only aniso 3', 4, 3)
