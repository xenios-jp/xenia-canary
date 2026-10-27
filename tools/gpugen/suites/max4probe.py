"""max4 over every arrangement of eight values in its four lanes: signed zeros
and denormals (which compare equal, so the result's bits say which lane wins
a tie), an ordinary value, and two NaNs with different payloads (which lane's
NaN comes out, and whether a NaN in each lane is kept or skipped). The extreme
suite shows a NaN in w comes out and NaNs elsewhere are ignored; this settles
ties and two NaNs. 8^4 = 4096 vertices of one float4."""

import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gpugen  # noqa: E402
import xenos as x  # noqa: E402

VALUES = [0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x3F800000,
          0xBF800000, 0x7FC00001, 0xFFC00002]


def inputs():
    words = []
    for combo in itertools.product(range(len(VALUES)), repeat=4):
        words += [VALUES[i] for i in combo]
    return words


def build(suite):
    suite.set_inputs(inputs(), 4)
    a = gpugen.INPUT_REGISTER
    suite.test('max4', gpugen.memexport_shader(
        4, [], [gpugen.export_vector(x.V_MAX4, x.r(a))]))
