"""Anisotropic filtering's footprint from vertex shaders: an impulse swept
under explicit anisotropic gradients, so the result is the filter's
kernel along the major axis (its taps, their spacing and weights) and,
from the level tags, the LOD it picked.

'imp' is a 64x64 k_8_8_8_8 texture with levels 0-6, not packed (levels
from 1 on 2048 dwords apart under the mip address): at level 0 texel
(col, row) is (255 if col == 32, 255 if row == 38, 0, 255), at level k
(0, 0, 32 k, 255).

Vertex v (1024): ratio 2^(v // 256) between the major and minor
gradients, the minor one texel (1/64), the major axis along u for the 'u'
tests (H = (ratio / 64, 0), V = (0, 1 / 64)) and along v for the 'v'
tests (H = (1 / 64, 0), V = (0, ratio / 64)). The coordinates sweep the
impulse along that axis in 1/16 texel steps, (c + 0.5 + (v % 256 - 128) /
16) / 64 for its column (32) or row (38), the other coordinate on the
impulse's row or column centre.
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
LINEAR = 1
TRILINEAR = (LINEAR, LINEAR, LINEAR)
IMPULSE = (32, 38)
f2w = gpugen.f2w


def inputs():
    """r1 = (sweep along u, 38.5 / 64, 32.5 / 64, sweep along v),
    r2 = u-major gradients, r3 = v-major gradients."""
    words = []
    for v in range(VERTICES):
        ratio = 2.0 ** (v // 256)
        offset = (v % 256 - 128) / 16
        su = (IMPULSE[0] + 0.5 + offset) / SIDE
        sv = (IMPULSE[1] + 0.5 + offset) / SIDE
        words += [f2w(su), f2w((IMPULSE[1] + 0.5) / SIDE),
                  f2w((IMPULSE[0] + 0.5) / SIDE), f2w(sv),
                  f2w(ratio / SIDE), 0, 0, f2w(1 / SIDE),
                  f2w(1 / SIDE), 0, 0, f2w(ratio / SIDE)]
    return words


def base_texel(col, row):
    return ((255 if col == IMPULSE[0] else 0) |
            (255 if row == IMPULSE[1] else 0) << 8 | 0xFF << 24)


def build(suite):
    suite.set_inputs(inputs(), 12)
    suite.add_texture('imp base', [base_texel(d % SIDE, d // SIDE)
                                   for d in range(SIDE * SIDE)])
    suite.add_texture('imp mips', [(32 * (1 + d // MIP_SPACING)) << 16 |
                                   0xFF << 24
                                   for d in range(6 * MIP_SPACING)])
    vertex = gpugen.fetch_inputs(12)
    move = gpugen.export_copy(4)

    def test(name, axis, aniso, fetch=None, **constant):
        words = gpugen.texture_constant(F8888, SIDE, SIDE, filters=TRILINEAR,
                                        pitch=SIDE, mips=LEVELS, aniso=aniso,
                                        **constant)
        grad = 2 if axis == 'u' else 3
        sets = [x.set_texture_state(x.FETCH_SET_GRADIENTS_HORZ, grad,
                                    'xy').serialized(),
                x.set_texture_state(x.FETCH_SET_GRADIENTS_VERT, grad, 'zw')]
        keywords = dict(use_reg_gradients=True)
        keywords.update(fetch or {})
        source = 'xy' if axis == 'u' else 'zw'
        fetches = vertex + sets + [x.tfetch(4, 1, source, SLOT, **keywords)]
        suite.test(name, gpugen.memexport_shader(12, [], [[move]],
                                                 fetches=fetches),
                   fetch_constants={SLOT: (words, ('imp base', 'imp mips'))})

    for aniso in (0, 2, 3, 4, 5):
        test(f'u aniso {aniso}', 'u', aniso)
    test('v aniso 3', 'v', 3)
    test('u aniso 3 bias 4', 'u', 3, aniso_bias=4)
