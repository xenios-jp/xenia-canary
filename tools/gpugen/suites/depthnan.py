"""Separate NaN fetch, VMAX, and depth-export behavior with exact raw bits.

CNDEQ with a known-zero condition selects the source without VMAX. Raw
memory exports compare both instructions before the same source/instruction
is written to D24S8/D24FS8. Console capture, not a clamp model, is the oracle.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import depth  # noqa: E402
import gpugen  # noqa: E402
import rb  # noqa: E402
import xenos as x  # noqa: E402

VALUES = (0x7FC00001, 0xFFC00001, 0x7F800001, 0xFF800001,
          0x7F800000, 0x3F000000)
K = rb.K


def copy(source, dest, maximum=False):
    return x.alu(vector_op=x.V_MAX if maximum else x.V_CNDEQ,
                 vector_dest=dest, vector_mask=0xF, export=True,
                 src1=source if maximum else x.c(K, 'zzzz'),
                 src2=source, src3=source)


def shader(fetched, raw=False, maximum=False):
    s = x.Shader()
    if fetched:
        s.exec_([x.alu(vector_op=x.V_ADD, vector_dest=1, vector_mask=3,
                       src1=x.r(0, absolute=True), src2=x.c(K, 'xxxx'))])
        s.exec_([x.tfetch(2, 1, 'xy', rb.SLOT,
                          denormalized=True).serialized()])
    source = x.r(2) if fetched else x.c(K + 1)
    if raw:
        s.exec_([x.alu(vector_op=x.V_FLOOR, vector_dest=3, vector_mask=3,
                       src1=x.r(0, absolute=True),
                       src2=x.r(0, absolute=True)).serialized(),
                 x.alu(vector_op=x.V_MAD, vector_dest=3, vector_mask=1,
                       src1=x.r(3, 'yyyy'), src2=x.c(K, 'yyyy'),
                       src3=x.r(3, 'xxxx'))])
        gpugen.emit_exports(s, [[copy(source, x.EXPORT_DATA0)],
                                [copy(source, x.EXPORT_DATA0, True)]],
                            x.r(3, 'xxxx'), False)
    s.alloc(x.ALLOC_COLORS, 0)
    color = copy(x.c(K, 'wwww'), 0)
    if raw:
        s.exec_([color], end=True)
    else:
        s.exec_([color, copy(source, x.EXPORT_DEPTH, maximum)], end=True)
    return s


def build(suite):
    suite.set_inputs(rb.RECT, 2)
    suite.export_slots = rb.WIDTH * rb.HEIGHT
    constants = {256 + K: [gpugen.f2w(0.5), gpugen.f2w(rb.WIDTH),
                          0, gpugen.f2w(1.0)]}
    vertex = gpugen.resolve_vertex_shader()
    for index, vector in enumerate((VALUES[:4],
                                   (VALUES[4], 0xFF800000, VALUES[5], 0))):
        texture = f'raw {index}'
        suite.add_texture(texture, list(vector) * suite.slots)
        for fetched in (False, True):
            suite.test(f'{"texture" if fetched else "constant"} raw {index}',
                       vertex, constants={**constants, 256 + K + 1: vector},
                       elements=2, pixel_shader=shader(fetched, raw=True),
                       program_cntl=rb.PROGRAM_CNTL,
                       registers=rb.registers(mode_cntl=gpugen.MODE_CNTL),
                       draw=rb.RECT_DRAW, fetch_constants=rb.fetch(texture))
    for bits in VALUES:
        texture = f'{bits:08X}'
        suite.add_texture(texture, [bits] * (suite.slots * 4))
        for fetched in (False, True):
            for maximum in (False, True):
                for fmt in (depth.D24S8, depth.D24FS8):
                    state = rb.registers(depth_format=fmt,
                                         mode_cntl=gpugen.MODE_CNTL)
                    state.append(('RB_DEPTHCONTROL', [2 | 4 | (7 << 4)]))
                    suite.test(
                        f'{"texture" if fetched else "constant"} '
                        f'{"MAX" if maximum else "select"} '
                        f'{bits:08X} {"D24FS8" if fmt else "D24S8"}',
                        vertex, constants={**constants,
                                           256 + K + 1: [bits] * 4},
                        pixel_shader=shader(fetched, maximum=maximum),
                        program_cntl=((rb.PROGRAM_CNTL & ~(15 << 27)) |
                                      (3 << 27)),
                        registers=state, draw=rb.RECT_DRAW,
                        fetch_constants=rb.fetch(texture),
                        resolve=depth.RESOLVE,
                        clear=gpugen.resolve_registers(
                            rb.WIDTH, rb.HEIGHT, 6, 4, clear=(0x555555A5, 0)))
