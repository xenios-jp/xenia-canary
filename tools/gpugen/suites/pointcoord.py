"""Point sprite orientation and clipping, independent of triangle coverage.

Each covered pixel exports (texel_x, texel_y, 16*texel_x+texel_y, 1),
point-sampled from an asymmetric 8x8 float texture, then raw PsParamGen.
Uncovered pixels retain the sentinel.
The 16 non-overlapping sprites sweep sixteenth-pixel center phases. Sizes
include odd, even and fractional diameters. Viewport tests map the same
screen centers through both Y signs; scissor tests cut all four sprite edges.
Single-sample only: pixel memexports do not provide per-sample ordering.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpugen  # noqa: E402
import pixel  # noqa: E402
import pm4  # noqa: E402
import xenos as x  # noqa: E402


def build(suite):
    centers = [(8 + 14 * (i % 4) + i / 16,
                8 + 14 * (i // 4) + ((7 * i) % 16) / 16)
               for i in range(16)]
    blocks = [('screen', pm4.PRIM_POINT_LIST,
               [pixel.vertex(p, (i, 0, 0, 1))
                for i, p in enumerate(centers)])]
    for sign in (-1, 1):
        blocks.append((str(sign), pm4.PRIM_POINT_LIST,
                       [[gpugen.f2w(v) for v in
                         ((px - 32) / 32, (py - 32) / (32 * sign), .5, 1,
                          i, 0, 0, 1)]
                        for i, (px, py) in enumerate(centers)]))
    draws = pixel.set_blocks(suite, blocks)
    # A linear float4 texture has a 32-texel pitch, including row padding.
    suite.add_texture('orientation', [gpugen.f2w(v)
                      for py in range(8) for px in range(32)
                      for v in (px, py, 16 * px + py, 1)])
    body = [x.alu(vector_op=x.V_FLOOR, vector_dest=2, vector_mask=3,
                  src1=x.r(1, absolute=True), src2=x.r(1, absolute=True)),
            x.alu(vector_op=x.V_MAD, vector_dest=2, vector_mask=1,
                  src1=x.r(2, 'y'), src2=x.c(pixel.K, 'x'),
                  src3=x.r(2, 'x'))]
    ps = gpugen.memexport_pixel_shader(
        body, [[gpugen.export_copy(3)], [gpugen.export_copy(1)]], x.r(2, 'x'),
        [x.tfetch(3, 1, 'zw', 1).serialized()])
    probe = pixel.Probe()
    for diameter in (3, 4, 7.5):
        fixed = int(diameter * 8)
        size = [('PA_SU_POINT_SIZE', [fixed | (fixed << 16)])]
        for mode in ('screen', '-1', '1', 'scissor'):
            regs = list(pixel.REGISTERS) + size
            block = mode if mode in ('-1', '1') else 'screen'
            if mode in ('-1', '1'):
                regs += [('PA_CL_VTE_CNTL', [0x43F]),
                         ('PA_CL_CLIP_CNTL', [1 << 19]),
                         ('PA_CL_VPORT_XSCALE', [gpugen.f2w(v) for v in
                          (32, 32, 32 * int(mode), 32, 1, 0)])]
            if mode == 'scissor':
                regs += [('PA_SC_WINDOW_SCISSOR_TL', [(8 << 16) | 8]),
                         ('PA_SC_WINDOW_SCISSOR_BR', [(51 << 16) | 51])]
            suite.test(f'point diameter {diameter:g} {mode}', probe.shader,
                       constants=probe.constants, elements=2, pixel_shader=ps,
                       program_cntl=pixel.PROGRAM_CNTL, registers=regs,
                       draw=draws[block], fetch_constants={1: (
                           gpugen.texture_constant(38, 8, 8), 'orientation')})
