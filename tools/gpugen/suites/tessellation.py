"""Adaptive quad patch indices, domain coordinates and edge factors.

Two patches have different screen positions and export (domain U, domain V,
patch index, 1) to RGBA16F. Alpha distinguishes covered pixels from the clear,
including vertices whose coordinates are zero. The console reference records
both the domain range and interpolation; neither is inferred from a host API.

Each patch has four float32 edge factors in the index buffer. Uniform factors
1, 2, 4 and 8 isolate subdivision while preserving the same two patch indices.
This suite covers the explicit adaptive quad grouping below, not triangle,
control-point-indexed or nonadaptive tessellation.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpugen  # noqa: E402
import pm4  # noqa: E402
import rb  # noqa: E402
import xenos as x  # noqa: E402

# VGT_DRAW_INITIATOR: quad patch, explicit major mode. Its index buffer holds
# tessellation factors rather than vertex indices in adaptive mode.
QUAD_PATCH_EXPLICIT = 0x12 | (1 << 8)
FACTORS = (1, 2, 4, 8)
f2w = gpugen.f2w


def vertex_shader(canonical=False):
    """Screen position from r0.yz; patch index r0.x separates the two patches.

    Export the original coordinates and index, with a constant alpha coverage
    marker. The position's W is explicitly 1 rather than an unspecified input.
    """
    s = x.Shader()
    if canonical:
        # Clockwise corner selector: 0=(u,v), 1=(1-u,v),
        # 2=(1-u,1-v), 3=(u,1-v). Keep the raw input test separate.
        s.exec_([x.alu(vector_op=x.V_SEQ, vector_dest=4, vector_mask=0xF,
                       src1=x.r(1, 'xxxx'), src2=x.c(14))])
        s.exec_([x.alu(vector_op=x.V_ADD, vector_dest=5, vector_mask=0x3,
                       src1=x.r(4, 'yzzz'), src2=x.r(4, 'zwww'))])
        s.exec_([x.alu(vector_op=x.V_MAD, vector_dest=6, vector_mask=0x3,
                       src1=x.r(5), src2=x.c(16), src3=x.c(15))])
        s.exec_([x.alu(vector_op=x.V_MAD, vector_dest=6, vector_mask=0x3,
                       src1=x.r(0, 'yzyz'), src2=x.r(6), src3=x.r(5))])
        s.exec_([x.alu(vector_op=x.V_MAX, vector_dest=6, vector_mask=0x4,
                       src1=x.r(0, 'xxxx'), src2=x.r(0, 'xxxx'))])
    coord = x.r(6, 'xyxy') if canonical else x.r(0, 'yzyz')
    color = x.r(6) if canonical else x.r(0, 'yzxx')
    s.exec_([
        x.alu(vector_op=x.V_MUL, vector_dest=2, vector_mask=0xF,
              src1=coord, src2=x.c(8)),
        x.alu(vector_op=x.V_MAD, vector_dest=3, vector_mask=0xF,
              src1=x.r(0, 'xxxx'), src2=x.c(10), src3=x.c(11)),
    ])
    s.exec_([x.alu(vector_op=x.V_ADD, vector_dest=2, vector_mask=0xF,
                   src1=x.r(2), src2=x.r(3))])
    s.alloc(x.ALLOC_INTERPOLATORS, 0)
    s.exec_([
        x.alu(vector_op=x.V_MAX, vector_dest=0, vector_mask=0x7,
              export=True, src1=color, src2=color),
        x.alu(vector_op=x.V_MAX, vector_dest=0, vector_mask=0x8,
              export=True, src1=x.c(13), src2=x.c(13)),
    ])
    s.alloc(x.ALLOC_POSITION, 0)
    s.exec_([x.alu(vector_op=x.V_MAX, vector_dest=x.EXPORT_POSITION,
                   vector_mask=0xF, export=True,
                   src1=x.r(2), src2=x.r(2))], end=True)
    return s


def build(suite):
    suite.set_inputs(rb.RECT, 2)
    resolve = gpugen.resolve_registers(rb.WIDTH, rb.HEIGHT, 32, 8, number=7)
    clear = gpugen.resolve_registers(rb.WIDTH, rb.HEIGHT, 32, 8, number=7,
                                    clear=(0, 0), clear_enable=1)
    constants = {
        8: list(map(f2w, (12, 10, 0, 0))),
        10: list(map(f2w, (24, 6, 0, 0))),
        11: list(map(f2w, (3.5, 3.5, 0, 1))),
        13: list(map(f2w, (0.25, 0.25, 0.25, 1))),
    }
    state = rb.registers(rt_format=7, mode_cntl=gpugen.MODE_CNTL) + [
        ('SQ_CONTEXT_MISC', [0]),
        ('SQ_INTERPOLATOR_CNTL', [0xFFFF]),
        ('VGT_OUTPUT_PATH_CNTL', [1]),
        ('VGT_HOS_CNTL', [2]),
        ('VGT_HOS_MAX_TESS_LEVEL', [f2w(15), f2w(1)]),
        # Explicit adaptive quad grouping, including both input vectors.
        # Leaving these registers at the ordinary draw defaults stalls the
        # console; enabling tessellation alone is not a complete setup.
        ('VGT_HOS_REUSE_DEPTH', [14, 14, 4, 4, 0x801, 0x8080F,
                               0x61412107, 0x66462606]),
    ]
    # Four VS and four PS GPRs, resource flags and normal pixel export mode.
    program = 3 | (3 << 8) | (1 << 16) | (1 << 17) | (2 << 27)
    for factor in FACTORS:
        texture = f'edge factors {factor}'
        suite.add_texture(texture, [f2w(factor)] * 8)
        suite.test(f'adaptive quad factor {factor} two patches',
                   vertex_shader(),
                   constants=constants,
                   pixel_shader=gpugen.trivial_pixel_shader(),
                   program_cntl=program, registers=state,
                   draw=(QUAD_PATCH_EXPLICIT, 0, 8),
                   indices=(texture, pm4.INDEX_INT32, pm4.ENDIAN_8IN32),
                   resolve=resolve, clear=clear)
