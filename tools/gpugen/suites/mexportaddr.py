"""Memory export bounds and ordering with ALLOC size 1.

Five eM registers intentionally characterize an undersized allocation. This is
not an assertion that ordinary correctly allocated exports wrap. One invocation
owns sixteen elements, enclosing both linear and observed wrapped addresses.
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

    def shader(writes):
        def flow(s):
            s.alloc(x.ALLOC_MEMORY, 1)
            s.exec_([x.alu(vector_op=x.V_MAX,
                          vector_dest=x.EXPORT_ADDRESS, vector_mask=15,
                          export=True, src1=x.c(gpugen.C_STREAM),
                          src2=x.c(gpugen.C_STREAM), serialize=True)] + [
                x.alu(vector_op=x.V_MAX, vector_dest=x.EXPORT_DATA0 + element,
                      vector_mask=15, export=True, src1=x.c(8 + value),
                      src2=x.c(8 + value)) for element, value in writes])
        return gpugen.memexport_shader(4, [], [], fetches=[], flow=flow)

    def add(name, count, writes):
        suite.test(name, shader(writes), constants=constants,
                   stream=lambda w: [w[0], 0x4B000002, w[2], 0x4B000000 | count])

    for count in range(9):
        add(f'index 2 five elements count {count}', count, list(enumerate(range(5))))
    for element in range(5):
        add(f'index 2 only eM{element}', 16, [(element, element)])
    add('index 2 reverse element writes', 16, list(reversed(list(enumerate(range(5))))))
    add('index 2 repeated eM2 last write', 16, [(2, 1), (2, 5)])
