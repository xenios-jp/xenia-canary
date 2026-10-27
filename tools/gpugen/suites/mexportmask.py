"""Component write masks merge into initialized memory-export data.

Every vertex first writes all eM0 components from its first float4, then
writes its second float4 with one of all 16 vector masks. This avoids
assuming any value for unwritten export-register components. The same
merge is checked before float32, float16, and packed unorm conversion.
No two vertices share an export address. Endian is the normal 8in32.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gpugen  # noqa: E402
import xenos as x  # noqa: E402

FORMATS = (('float4', 38, 7, 16), ('rgba8', 6, 0, 4),
           ('rgba16', 26, 0, 8), ('half4', 32, 7, 8),
           ('rgb10a2', 7, 0, 4))


def build(suite):
    words = []
    for i in range(16):
        a = (i / 16, (i + 3) % 16 / 16, (i + 7) % 16 / 16,
             (i + 11) % 16 / 16)
        b = tuple(1 - v for v in a)
        words += [gpugen.f2w(v) for v in a + b]
    suite.set_inputs(words, 8)
    for name, fmt, num, size in FORMATS:
        for mask in range(16):
            writes = [gpugen.export_copy(gpugen.INPUT_REGISTER),
                      x.alu(vector_op=x.V_MAX,
                            vector_dest=x.EXPORT_DATA0, vector_mask=mask,
                            export=True, src1=x.r(gpugen.INPUT_REGISTER + 1),
                            src2=x.r(gpugen.INPUT_REGISTER + 1))]
            suite.test(f'{name} initialized mask {mask:X}',
                       gpugen.memexport_shader(8, [], [writes]),
                       export=(fmt, num, size))
