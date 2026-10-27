"""Quad-domain reconstruction and nonadaptive patch counting.

The raw-input tessellation suite deliberately leaves the console's folded
coordinates unchanged. This companion applies its clockwise corner selector
and records canonical U/V instead, distinguishing a representation difference
from an observable difference after reconstruction. Asymmetric edge factors
exercise all four edge positions on two different patches.

The continuous cases consume one auto-generated index per patch, rather than
four float edge factors, covering the nonadaptive input-patch-size path.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpugen  # noqa: E402
import pm4  # noqa: E402
import rb  # noqa: E402
import tessellation  # noqa: E402

f2w = gpugen.f2w
CASES = (
    ('adaptive uniform canonical', 2, [2] * 8, True),
    ('adaptive asymmetric canonical', 2, [1, 2, 4, 8, 8, 4, 2, 1], True),
    ('continuous two patches raw', 1, None, False),
    ('continuous two patches canonical', 1, None, True),
)


def build(suite):
    suite.set_inputs(rb.RECT, 2)
    resolve = gpugen.resolve_registers(rb.WIDTH, rb.HEIGHT, 32, 8, number=7)
    clear = gpugen.resolve_registers(rb.WIDTH, rb.HEIGHT, 32, 8, number=7,
                                    clear=(0, 0), clear_enable=1)
    for name, mode, factors, canonical in CASES:
        constants = {
            8: list(map(f2w, (12, 10, 0, 0))),
            10: list(map(f2w, (24, 6, 0, 0))),
            11: list(map(f2w, (3.5, 3.5, 0, 1))),
            13: list(map(f2w, (0.25, 0.25, 0.25, 1))),
        }
        if canonical:
            constants.update({14: list(map(f2w, (0, 1, 2, 3))),
                              15: [f2w(1)] * 4, 16: [f2w(-2)] * 4})
        adaptive = mode == 2
        # Both modes emit an auto primitive index. Adaptive additionally
        # consumes the four edge-factor floats in vector 1; continuous uses
        # register factors and decrements the draw count once per patch.
        grouping = ([14, 14, 4, 4, 0x801, 0x8080F, 0x61412107, 0x66462606]
                    if adaptive else [14, 14, 1, 1, 1, 0, 7, 0])
        state = rb.registers(rt_format=7, mode_cntl=gpugen.MODE_CNTL) + [
            ('SQ_CONTEXT_MISC', [0]),
            ('SQ_INTERPOLATOR_CNTL', [0xFFFF]),
            ('VGT_OUTPUT_PATH_CNTL', [1]),
            ('VGT_HOS_CNTL', [mode]),
            ('VGT_HOS_MAX_TESS_LEVEL', [f2w(15 if adaptive else 2),
                                      f2w(1 if adaptive else 0)]),
            ('VGT_HOS_REUSE_DEPTH', grouping),
        ]
        indices = None
        if adaptive:
            suite.add_texture(name, list(map(f2w, factors)))
            indices = (name, pm4.INDEX_INT32, pm4.ENDIAN_8IN32)
        program = ((6 if canonical else 3) | (3 << 8) | (1 << 16) |
                   (1 << 17) | (2 << 27))
        suite.test(name, tessellation.vertex_shader(canonical),
                   constants=constants,
                   pixel_shader=gpugen.trivial_pixel_shader(),
                   program_cntl=program, registers=state,
                   draw=(tessellation.QUAD_PATCH_EXPLICIT, 0,
                         8 if adaptive else 2), indices=indices,
                   resolve=resolve, clear=clear)
