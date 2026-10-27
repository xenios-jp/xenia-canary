"""Distinguish stream-relative from physical four-element export wrapping.

The stream starts one element into an owned sixteen-element sentinel buffer.
Index 3 writes only eM1 under ALLOC1 and ALLOC2, for float1/2/4. Both the
relative-index wrapping candidate (guard slot1) and physical-window or
linear candidates (slots4/5) remain inside the owned buffer. No physical
boundary or cross-invocation address is probed.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gpugen  # noqa: E402
import xenos as x  # noqa: E402


def shader(allocation):
    def flow(s):
        s.alloc(x.ALLOC_MEMORY, allocation)
        s.exec_([
            x.alu(vector_op=x.V_MAX, vector_dest=x.EXPORT_ADDRESS,
                  vector_mask=15, export=True,
                  src1=x.c(gpugen.C_STREAM), src2=x.c(gpugen.C_STREAM),
                  serialize=True),
            x.alu(vector_op=x.V_MAX, vector_dest=x.EXPORT_DATA0 + 1,
                  vector_mask=15, export=True, src1=x.c(8), src2=x.c(8))])
    return gpugen.memexport_shader(4, [], [], fetches=[], flow=flow)


def build(suite):
    suite.set_generated(1)
    suite.export_slots = 16
    for name, fmt, size in (('float1', 36, 4), ('float2', 37, 8),
                            ('float4', 38, 16)):
        for allocation in (1, 2):
            suite.test(
                f'{name} base element 1 index 3 eM1 allocation {allocation}',
                shader(allocation),
                constants={8: [gpugen.f2w(v) for v in (17, 18, 19, 20)]},
                export=(fmt, 7, size),
                stream=lambda words, size=size:
                    [words[0] + size // 4, 0x4B000003,
                     words[2], 0x4B00000F])
