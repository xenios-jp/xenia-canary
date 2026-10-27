"""Sub-LSB depth plus signed bias: distinguish addition from quantization order.

For each format, depth and bias are exact binary fractions of its smallest
positive depth step. Z and oDepth share the matrix; both 2x samples are read
raw, preserving stencil A5. Expected words are measured on the console.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import depthcases as dc  # noqa: E402


def cases():
    out = []
    for fmt, unit in ((dc.D24S8, 2.0 ** -24), (dc.D24FS8, 2.0 ** -34)):
        for path in (0, 1):
            for quarter_depth in (1, 2, 3, 5):
                z = quarter_depth * unit * 0.25
                for quarter_bias in (0, -3, -2, -1, 1, 2, 3):
                    bias = quarter_bias * unit * 0.25
                    out.append((len(out),
                                f'depth{quarter_depth}/4-bias{quarter_bias}/4',
                                dc.case(fmt, 0x555555A5, [dc.draw(
                                    path, z=z, exported=z, bias=bias)])))
    return out


def build(suite):
    dc.build(suite, cases(), samples=2)
    for test in suite.tests:
        test.name = test.name.replace('555555A5 0', 'bias before rounding')
