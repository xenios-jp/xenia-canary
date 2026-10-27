"""Signed constant/slope bias, Z versus oDepth, raw depth and stencil at 2x.

Flat cases pin tiny and ordinary depths with both offset signs. Whole-target
planes separate slope scale from constant offset, including negative slopes.
All tests preserve a nonzero stencil clear. Console output is the oracle.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import depthcases as dc  # noqa: E402


def cases():
    out = []
    for fmt in (dc.D24S8, dc.D24FS8):
        for path in (0, 1):
            for z in (2.0 ** -34, 2.0 ** -14, 0.25, 0.5, dc.f32(0.3)):
                for bias in (0.0, -(2.0 ** -24), 2.0 ** -24, -1.0, 1.0):
                    out.append((len(out), f'flat-{dc.bits(z):08X}-{bias:g}',
                                dc.case(fmt, 0x555555A5, [dc.draw(
                                    path, z=z, exported=z, bias=bias)])))
        for direction in (-1, 1):
            corners = [0.5, 0.5 + direction * 2.0 ** -8,
                       0.5, 0.5 + direction * 2.0 ** -8]
            for slope in (-1.0, 0.0, 1.0):
                out.append((len(out), f'plane-x{direction:+d}-scale{slope:+g}',
                            dc.case(fmt, 0x555555A5, [dc.draw(
                                corners=corners, slope=slope)], full=1)))
    return out


def build_samples(suite, samples):
    dc.build(suite, cases(), samples=samples)
    for test in suite.tests:
        test.name = test.name.replace('555555A5 0', 'signed constant bias')


def build(suite):
    build_samples(suite, 2)
