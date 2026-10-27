"""Metro: Last Light's sprite pass (console replay stage 10/845) in one draw:
a 1x k_8_8_8_8 target, a first draw storing 'dest' (dark colours k / 255,
k < 8, alpha k / 255 random), then the sprites' state: RB_BLENDCONTROL0
0x01060106 (SRC_ALPHA + ONE in colour and alpha), RB_COLOR_MASK 0x7,
RB_COLORCONTROL 0x87000007, storing 'source' (dark colours: random 24-bit
fractions times 0.125; alpha random 24-bit fractions). Read back raw.
Tests: 'sprite mask 7', and 'sprite mask f' (alpha written too), 'add mask
f' (ONE + ONE), 'sprite 2x' at 2x (sample 0 raw)."""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpugen  # noqa: E402
import msaarb  # noqa: E402
import rb  # noqa: E402

SPRITE = 0x01060106
ADD_ONE = 0x01010101
NO_BLEND = 0x00010001
COLORCONTROL = 0x87000007
f2w = gpugen.f2w


def textures():
    rng = random.Random(0x5B10)
    dest, source = [], []
    for _ in range(rb.WIDTH * rb.HEIGHT):
        dest += [f2w(rng.randrange(8) / 255) for _ in range(3)]
        dest.append(f2w(rng.randrange(256) / 255))
        source += [f2w(rng.getrandbits(24) / 2 ** 24 * 0.125)
                   for _ in range(3)]
        source.append(f2w(rng.getrandbits(24) / 2 ** 24))
    return dest, source


TESTS = [('sprite mask 7', SPRITE, 0x7, 0), ('sprite mask f', SPRITE, 0xF, 0),
         ('add mask f', ADD_ONE, 0xF, 0), ('sprite 2x', SPRITE, 0x7, 1)]


def build(suite):
    suite.set_inputs(rb.RECT, 2)
    dest, source = textures()
    suite.add_texture('dest', dest)
    suite.add_texture('source', source)
    for name, control, mask, samples in TESTS:
        state = rb.registers(0, samples, mode_cntl=gpugen.MODE_CNTL)
        rb.test(suite, name, 'source', state, msaarb.raw(0),
                passes=[([('RB_COLORCONTROL', [COLORCONTROL]),
                          ('RB_BLENDCONTROL0', [NO_BLEND]),
                          ('RB_COLOR_MASK', [0xF])], rb.fetch('dest'))],
                after_passes=[('RB_BLENDCONTROL0', [control]),
                              ('RB_COLOR_MASK', [mask])])
