"""Pinned depth representation boundaries, ties and clamps at both 2x samples.

Each flat case occupies one pixel, in cases() order within each format/path.
Covers the 20e4 subnormal/normal boundary, both retained-mantissa parities at
all exponents in [0, 1], and oDepth clamp/special values. Raw readback keeps
stencil 0xA5, making accidental stencil modification observable. Expected
words come from console capture, not a host rounding model.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import depthcases as dc  # noqa: E402
import stencil  # noqa: E402


def values():
    words = {0, 0x80000000}
    for exponent in range(16):
        for mantissa in (0, 1, (1 << 20) - 2):
            first = stencil.float20e4(exponent, mantissa)
            last = stencil.float20e4(exponent, mantissa + 1)
            for value in (first, (first + last) * 0.5, last):
                bits = dc.bits(dc.f32(value))
                for adjacent in (bits - 1, bits, bits + 1):
                    if 0 <= adjacent <= 0x3F800000:
                        words.add(adjacent)
    return sorted(words)


def cases():
    out = []
    for fmt in (dc.D24S8, dc.D24FS8):
        for path in (0, 1):
            inputs = values()
            if path:
                inputs += [0xBF800000, 0x3F800001, 0x3FC00000, 0x40000000,
                           0x7F800000, 0xFF800000, 0x7FC00000]
            for bits in inputs:
                value = dc.from_bits(bits)
                out.append((len(out), f'bits-{bits:08X}', dc.case(
                    fmt, 0x555555A5, [dc.draw(
                        path, z=value if not path else 0.5, exported=value)])))
    return out


def build_samples(suite, samples):
    dc.build(suite, cases(), samples=samples)
    for test in suite.tests:
        test.name = test.name.replace('555555A5 0', 'rounding boundaries')


def build(suite):
    build_samples(suite, 2)
