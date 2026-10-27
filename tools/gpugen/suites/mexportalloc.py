"""Memory export count boundaries with a five-register allocation.

ALLOC size 4 covers eM0 through eM4. Sizes 2 and 3 are explicit undersized
controls. One invocation owns sixteen elements; no inter-invocation ordering
or out-of-allocation addresses are involved.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gpugen
import xenos as x


def build(suite):
    suite.set_generated(1)
    suite.export_slots = 16
    constants = {8 + i: [gpugen.f2w(float(16 * i + j + 1)) for j in range(4)]
                 for i in range(6)}

    def shader(writes, allocation):
        def flow(s):
            s.alloc(x.ALLOC_MEMORY, allocation)
            s.exec_([x.alu(vector_op=x.V_MAX,
                          vector_dest=x.EXPORT_ADDRESS, vector_mask=15,
                          export=True, src1=x.c(gpugen.C_STREAM),
                          src2=x.c(gpugen.C_STREAM), serialize=True)] + [
                x.alu(vector_op=x.V_MAX, vector_dest=x.EXPORT_DATA0 + element,
                      vector_mask=15, export=True, src1=x.c(8 + value),
                      src2=x.c(8 + value)) for element, value in writes])
        return gpugen.memexport_shader(4, [], [], fetches=[], flow=flow)

    def add(name, index, writes, allocation=4, count=16, export=gpugen.EXPORT_FLOAT4):
        suite.test(name, shader(writes, allocation), constants=constants, export=export,
                   stream=lambda w: [w[0], 0x4B000000 | index, w[2], 0x4B000000 | count])

    writes = list(enumerate(range(5)))
    for count in range(9):
        add(f'allocation 4 index 2 count {count}', 2, writes, count=count)
    for allocation in (2, 3):
        for index in (2, 7):
            add(f'allocation {allocation} index {index}', index, writes,
                allocation=allocation)
    for count in (7, 8, 12):
        add(f'allocation 4 index 7 count {count}', 7, writes, count=count)
