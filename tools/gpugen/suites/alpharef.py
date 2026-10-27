"""The alpha test against other references, and alpha to mask with special
alphas, with the alpha suite's setup (the rb suite's draw into k_8_8_8_8,
a first pass storing 'zero').

- 'alpha test {function} ref {name}': RB_COLORCONTROL alpha_test_enable and
  function 1..6 (LESS .. GEQUAL) at 1x against RB_ALPHA_REF = each of REFS;
  the texture's alphas: SPECIALS, then each reference moved -3..3 float bits,
  then a ramp from -1 to 1 in steps of 1/512, then random floats of every
  exponent and sign.
- 'alpha to mask {count}x sample {k}': at 2x and 4x with offsets 0xE4
  (pixel (x & 1) + 2 (y & 1) gets offset 0, 1, 2, 3), the same texture,
  every sample resolved raw.
"""

import math
import os
import random
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpugen  # noqa: E402
import msaarb  # noqa: E402
import rb  # noqa: E402

ALPHA_TEST = 1 << 3
REFS = [('-0.5', 0xBF000000), ('-0', 0x80000000), ('0', 0x00000000),
        ('nan', 0x7FC00000), ('-inf', 0xFF800000), ('denorm', 0x00000001),
        ('-1', 0xBF800000)]
SPECIALS = [0x7FC00000, 0xFFC00000, 0x7F800001, 0xFFBFFFFF, 0x7F800000,
            0xFF800000, 0x00000000, 0x80000000, 0x00000001, 0x80000001,
            0x007FFFFF, 0x807FFFFF, 0x00800000, 0x80800000, 0x7F7FFFFF,
            0xFF7FFFFF, 0x3F800000, 0xBF800000, 0x3F000000, 0xBF000000]


def bits(v):
    return struct.unpack('>I', struct.pack('>f', v))[0]


def value(w):
    return struct.unpack('>f', struct.pack('>I', w))[0]


def step(w, d):
    v = value(w)
    for _ in range(abs(d)):
        v = math.nextafter(v, math.inf if d > 0 else -math.inf)
    return bits(v)


def alphas():
    rng = random.Random(0xA1F)
    out = list(SPECIALS)
    for _, ref in REFS:
        if ref in (0x7FC00000, 0xFF800000):
            continue
        out += [step(ref, d) for d in range(-3, 4)]
    out += [bits(-1 + k / 512) for k in range(1025)]
    while len(out) < rb.WIDTH * rb.HEIGHT:
        out.append(bits(rng.choice((-1, 1)) * rng.uniform(1, 2) *
                        2.0 ** rng.randint(-40, 40)))
    return out


def texture():
    words = []
    for a in alphas():
        words += [bits(1.0)] * 3 + [a]
    return words


def build(suite):
    suite.set_inputs(rb.RECT, 2)
    suite.add_texture('zero', rb.fill((0.0, 0.0, 0.0, 0.0)))
    suite.add_texture('alphas', texture())

    def test(name, samples, colorcontrol, ref=0x3F000000, sample=0):
        state = rb.registers(samples=samples, mode_cntl=gpugen.MODE_CNTL)
        state.append(('RB_ALPHA_REF', [ref]))
        rb.test(suite, name, 'alphas', state, msaarb.raw(sample),
                passes=[([('RB_COLORCONTROL', [msaarb.COLORCONTROL])],
                         rb.fetch('zero'))],
                after_passes=[('RB_COLORCONTROL', [colorcontrol])])

    for ref_name, ref in REFS:
        for function in range(1, 7):
            test(f'alpha test {function} ref {ref_name}', 0,
                 function | ALPHA_TEST, ref)
    for samples, count in ((1, 2), (2, 4)):
        for k in range(count):
            test(f'alpha to mask {count}x sample {k}', samples,
                 msaarb.COLORCONTROL | msaarb.ALPHA_TO_MASK | 0xE4 << 24,
                 sample=k)
