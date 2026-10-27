"""Affine memexport count bounds followed by a dependent vertex fetch.

Every memory allocation writes an address even when its data clause is inactive.
One invocation writes eM0..4 at base index 2 with ALLOC size 4. A second draw
reads element 6 and exports it into element 0. Both draws stay within the same
sixteen-element owned output. Count 2 is the no-write control; count 3 accepts
all five elements on Xbox, including the element the second draw reads.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gpugen
import pm4
import xenos as x


def build(suite):
    suite.set_generated(1)
    suite.export_slots = 16
    # Float-constant writes cover four constants: keep the pass flag in
    # its own group so changing it cannot clear the read index in c14.
    constants = {8: [gpugen.f2w(1.), gpugen.f2w(2.), 0, 0],
                 14: [gpugen.f2w(6.)] * 4, 16: [0] * 4}
    for i in range(5):
        constants[9 + i] = [gpugen.f2w(float(16 * i + j + 1)) for j in range(4)]

    def flow(s):
        s.exec_([x.alu(vector_op=x.V_MAD, vector_dest=1, vector_mask=15,
                       src1=x.r(0, 'x'), src2=x.c(8, 'xxxx'),
                       src3=x.c(8, 'yyyy'), serialize=True),
                 x.alu(scalar_op=x.S_SETP_EQ, scalar_dest=5, scalar_mask=1,
                       src3=x.c(16))], predicate_clean=False)
        s.alloc(x.ALLOC_MEMORY, 4)
        s.exec_([x.alu(vector_op=x.V_MAD, vector_dest=x.EXPORT_ADDRESS,
                       vector_mask=15, export=True, src1=x.r(1, 'x'),
                       src2=x.c(gpugen.C_EA, 'xyxx'), src3=x.c(gpugen.C_STREAM))])
        s.exec_([
            x.alu(vector_op=x.V_MAX, vector_dest=x.EXPORT_DATA0 + i,
                  vector_mask=15, export=True, src1=x.c(9 + i),
                  src2=x.c(9 + i)) for i in range(5)], condition=('pred', True))
        gpugen.emit_exports(s, [], x.r(0, 'x'), False)
        s.exec_([x.alu(vector_op=x.V_MAX, vector_dest=3, vector_mask=15,
                       src1=x.c(14), src2=x.c(14))], condition=('pred', False))
        s.exec_([x.vfetch(2, 3, 'x', 0, 0, x.VF_32_32_32_32_FLOAT, 4,
                          prefetch=4)], condition=('pred', False))
        s.alloc(x.ALLOC_MEMORY, 1)
        s.exec_([x.alu(vector_op=x.V_MAX, vector_dest=x.EXPORT_ADDRESS,
                       vector_mask=15, export=True, src1=x.c(gpugen.C_STREAM),
                       src2=x.c(gpugen.C_STREAM), serialize=True)])
        s.exec_([gpugen.export_copy(2)], condition=('pred', False))

    for count in (2, 3):
        suite.test(f'affine base 2 count {count} dependent element 6',
                   gpugen.memexport_shader(4, [], [], fetches=[], flow=flow),
                   constants=constants,
                   stream=lambda w, count=count: w[:3] + [0x4B000000 | count],
                   passes=[([], {0: ([3, 0x10000102, 0, 0, 0, 0],
                                      ('$output_vertex', 0))},
                            (pm4.PRIM_POINT_LIST, 0, 1))],
                   after_passes=[('SHADER_CONSTANT_016_X', [gpugen.f2w(1.)] * 4)],
                   fetch_constants={0: ([3, 0x10000102, 0, 0, 0, 0],
                                        ('$output_vertex', 0))})
