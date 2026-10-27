"""Predicated memory exports from one invocation into its owned output.

The initialized cases distinguish skipped overwrites from loss of the previous
value. Uninitialized cases observe whether a skipped eM write marks it written.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gpugen
import xenos as x


def build(suite):
    suite.set_generated(1)
    suite.export_slots = 4
    constants = {8: [gpugen.f2w(v) for v in (1., 2., 3., 4.)],
                 9: [gpugen.f2w(v) for v in (11., 12., 13., 14.)], 10: [0] * 4}
    def copy(constant, predicate=None):
        return x.alu(vector_op=x.V_MAX, vector_dest=x.EXPORT_DATA0,
                     vector_mask=15, export=True, src1=x.c(constant),
                     src2=x.c(constant), predicate=predicate)
    for initialized in (False, True):
        for state in (False, True):
            for condition in (False, True):
                def flow(s, initialized=initialized, state=state, condition=condition):
                    s.alloc(x.ALLOC_MEMORY, 1)
                    s.exec_([x.alu(vector_op=x.V_MAX,
                                  vector_dest=x.EXPORT_ADDRESS, vector_mask=15,
                                  export=True, src1=x.c(gpugen.C_STREAM),
                                  src2=x.c(gpugen.C_STREAM), serialize=True)])
                    if initialized:
                        s.exec_([copy(8)])
                    s.exec_([x.alu(scalar_op=x.S_SETP_EQ if state else x.S_SETP_NE,
                                  scalar_dest=5, scalar_mask=1, src3=x.c(10))])
                    s.exec_([copy(9, condition)])
                suite.test(f'initialized {int(initialized)} predicate {int(state)} condition {int(condition)}',
                           gpugen.memexport_shader(4, [], [], fetches=[], flow=flow),
                           constants=constants)
