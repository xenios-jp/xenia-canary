"""8_8_8_8_GAMMA at 2x and 4x, resolved with an exponent bias and into other
formats. The msaarb suite's setup: a first draw stores 'a' (random k / 255
colours, alpha 1) unblended in every sample; a second stores 'b' (random
k / 255 colours) with alpha to mask at offsets 0xAA, its alpha per 2x2 quad
q = x // 2 + 32 (y // 2) = (q % 5) / 4 + 1 / 8 (0 to 4 of 4 samples, at 2x 0
to 2), so samples differ. Then a test per readback:
- '{n}x sample {k}': raw copy of each sample;
- '{n}x {format}[ exp {bias}]': the averaging resolve to k_8_8_8_8 with
  exponent bias 0, 2, -1 and 4, to k_2_10_10_10, k_16_16_16_16 and
  k_16_16_16_16_FLOAT.
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpugen  # noqa: E402
import msaarb  # noqa: E402
import rb  # noqa: E402

GAMMA = 1
f2w = gpugen.f2w
# (name, dest format, bytes per texel, number format, exponent bias)
RESOLVES = [('8_8_8_8', 6, 4, 0, 0), ('8_8_8_8 exp 2', 6, 4, 0, 2),
            ('8_8_8_8 exp -1', 6, 4, 0, -1), ('8_8_8_8 exp 4', 6, 4, 0, 4),
            ('2_10_10_10', 7, 4, 0, 0), ('16_16_16_16', 26, 8, 0, 0),
            ('16_16_16_16_FLOAT', 32, 8, 7, 0)]
# samples -> (RB_SURFACE_INFO / PA_SC_AA_CONFIG code, averaging select)
SAMPLES = {2: (1, msaarb.SELECT_01), 4: (2, msaarb.SELECT_0123)}


def random_texture(rng, alpha=None):
    words = []
    for y in range(rb.HEIGHT):
        for x in range(rb.WIDTH):
            q = x // 2 + 32 * (y // 2)
            a = 1.0 if alpha is None else (q % 5) / 4 + 1 / 8
            words += [f2w(rng.randrange(256) / 255) for _ in range(3)]
            words.append(f2w(a))
    return words


def build(suite):
    rng = random.Random(0x6A3E)
    suite.set_inputs(rb.RECT, 2)
    suite.add_texture('a', random_texture(rng))
    suite.add_texture('b', random_texture(rng, alpha=True))
    for count, (code, select) in SAMPLES.items():
        readbacks = [(f'{count}x sample {k}', msaarb.raw(k))
                     for k in range(count)]
        readbacks += [(f'{count}x {name}', gpugen.resolve_registers(
            rb.WIDTH, rb.HEIGHT, dest, bpp, number, sample_select=select,
            exp_bias=bias)) for name, dest, bpp, number, bias in RESOLVES]
        state = rb.registers(GAMMA, code, mode_cntl=gpugen.MODE_CNTL)
        for name, resolve in readbacks:
            rb.test(suite, name, 'b', state, resolve,
                    passes=[([('RB_COLORCONTROL', [msaarb.COLORCONTROL])],
                             rb.fetch('a'))],
                    after_passes=[('RB_COLORCONTROL', [
                        msaarb.COLORCONTROL | msaarb.ALPHA_TO_MASK |
                        0xAA << 24])])
