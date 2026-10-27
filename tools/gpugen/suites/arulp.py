"""The alpha test at the reference's own last bits: alpharef's step()
moves a float64 by its last bit, which the float32 store drops, so its
'reference moved -3..3 float bits' alphas all equal the reference. Here
each alpha is a float32 reference moved d float32 bits (d -8 to 8), for
the references titles use (0.5, 128 / 255 = 0x3F008081, 0x3C808081,
0x3EE8E8EA) and -0.5, the smallest normal and denormal, and their
negatives, then alpharef's specials and ramp. The rb suite's draw into
k_8_8_8_8 over a first pass storing 'zero', as alpharef.

Tests: 'alpha test {function} ref {name}', function 1..6 (LESS ..
GEQUAL).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpugen  # noqa: E402
import msaarb  # noqa: E402
import rb  # noqa: E402
import alpharef  # noqa: E402
from cwulp import step32  # noqa: E402

REFS = [('0.5', 0x3F000000), ('128/255', 0x3F008081),
        ('0x3c808081', 0x3C808081), ('0x3ee8e8ea', 0x3EE8E8EA),
        ('-0.5', 0xBF000000), ('min normal', 0x00800000),
        ('-min normal', 0x80800000), ('denorm 2', 0x00000002),
        ('-denorm 2', 0x80000002)]


def alphas():
    out = list(alpharef.SPECIALS)
    for _, ref in REFS:
        out += [step32(ref, d) for d in range(-8, 9)]
    out += [alpharef.bits(-1 + k / 512) for k in range(1025)]
    out += [alpharef.bits((k + 0.5) / 255) for k in range(256)]
    out += [alpharef.bits(k / 255) for k in range(256)]
    while len(out) < rb.WIDTH * rb.HEIGHT:
        out.append(step32(out[len(out) - 512], 1))
    return out[:rb.WIDTH * rb.HEIGHT]


def texture():
    words = []
    for a in alphas():
        words += [alpharef.bits(1.0)] * 3 + [a]
    return words


def build(suite):
    suite.set_inputs(rb.RECT, 2)
    suite.add_texture('zero', rb.fill((0.0, 0.0, 0.0, 0.0)))
    suite.add_texture('alphas', texture())
    for ref_name, ref in REFS:
        for function in range(1, 7):
            state = rb.registers(samples=0, mode_cntl=gpugen.MODE_CNTL)
            state.append(('RB_ALPHA_REF', [ref]))
            rb.test(suite, f'alpha test {function} ref {ref_name}', 'alphas',
                    state, msaarb.raw(0),
                    passes=[([('RB_COLORCONTROL', [msaarb.COLORCONTROL])],
                             rb.fetch('zero'))],
                    after_passes=[('RB_COLORCONTROL', [
                        function | alpharef.ALPHA_TEST])])
