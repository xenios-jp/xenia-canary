"""Affine vertex-indexed memory exports at the stream-count boundary.

The MAD address form matches the host's vertex-index range analysis. One
invocation owns sixteen elements and writes eM0..4 with ALLOC size 4.
Count 2 rejects base index 2; count 3 accepts all five exported elements.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gpugen
import xenos as x


def build(suite):
    suite.set_generated(1)
    suite.export_slots = 16
    constants = {8: [gpugen.f2w(1.), gpugen.f2w(2.), 0, 0]}
    for i in range(5):
        constants[9 + i] = [gpugen.f2w(float(16 * i + j + 1)) for j in range(4)]

    def flow(s):
        s.exec_([x.alu(vector_op=x.V_MAD, vector_dest=1, vector_mask=15,
                       src1=x.r(0, 'x'), src2=x.c(8, 'xxxx'),
                       src3=x.c(8, 'yyyy'), serialize=True)])
        s.alloc(x.ALLOC_MEMORY, 4)
        s.exec_([x.alu(vector_op=x.V_MAD, vector_dest=x.EXPORT_ADDRESS,
                       vector_mask=15, export=True, src1=x.r(1, 'x'),
                       src2=x.c(gpugen.C_EA, 'xyxx'), src3=x.c(gpugen.C_STREAM))] + [
            x.alu(vector_op=x.V_MAX, vector_dest=x.EXPORT_DATA0 + i,
                  vector_mask=15, export=True, src1=x.c(9 + i),
                  src2=x.c(9 + i)) for i in range(5)])

    for count in (2, 3):
        suite.test(f'affine base 2 count {count}',
                   gpugen.memexport_shader(4, [], [], fetches=[], flow=flow),
                   constants=constants,
                   stream=lambda w, count=count: w[:3] + [0x4B000000 | count])
