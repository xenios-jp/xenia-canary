"""Memory export allocation-size and address-width discriminator.

Single-register writes isolate index mapping under ALLOC size 1. Sizes 0 and 4
control whether the five-register pattern is undersized. One invocation owns
sixteen elements; all observed and candidate addresses stay in that allocation.
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

    def add(name, index, writes, allocation=1, export=gpugen.EXPORT_FLOAT4):
        suite.test(name, shader(writes, allocation), constants=constants, export=export,
                   stream=lambda w: [w[0], 0x4B000000 | index, w[2], 0x4B000010])

    for index in (0, 1, 3, 4):
        for element in range(5):
            add(f'float4 index {index} only eM{element}', index, [(element, element)])
    for allocation in (0, 4):
        add(f'float4 index 2 allocation {allocation}', 2,
            list(enumerate(range(5))), allocation=allocation)
    for name, export in (('float1', (36, 7, 4)), ('float2', (37, 7, 8))):
        for element in range(5):
            add(f'{name} index 3 only eM{element}', 3, [(element, element)], export=export)
