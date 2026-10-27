"""Initialized scalar and dual-issue component writes to eM0.

All 16 scalar masks replace initialized float4 components with r2.w.
The paired cases also write r2 through the complementary vector mask in
that same instruction. Masks are disjoint, so no same-component priority
or undefined initial export-register value is assumed. Vertices export to
separate addresses, format float4 endian 8in32.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gpugen  # noqa: E402
import xenos as x  # noqa: E402


def build(suite):
    words = []
    for i in range(8):
        words += [gpugen.f2w(v) for v in
                  (i, i + .25, i + .5, i + .75,
                   -i - 1, -i - 2, -i - 3, -i - 4)]
    suite.set_inputs(words, 8)
    for dual in (False, True):
        for mask in range(16):
            writes = [gpugen.export_copy(gpugen.INPUT_REGISTER),
                      x.alu(vector_op=x.V_MAX,
                            vector_dest=x.EXPORT_DATA0,
                            vector_mask=(15 ^ mask) if dual else 0,
                            scalar_op=x.S_MAXS, scalar_mask=mask,
                            export=True,
                            src1=x.r(gpugen.INPUT_REGISTER + 1),
                            src2=x.r(gpugen.INPUT_REGISTER + 1),
                            src3=x.r(gpugen.INPUT_REGISTER + 1, 'w'))]
            suite.test(f'initialized scalar mask {mask:X} dual {int(dual)}',
                       gpugen.memexport_shader(8, [], [writes]))
