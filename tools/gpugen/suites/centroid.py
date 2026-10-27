"""Centroid sampling: where smooth interpolators are evaluated in a pixel at
1x, 2x and 4x, by SQ_CONTEXT_MISC.sc_sample_cntl (0 centroids, 1 centers, 2
by SQ_INTERPOLATOR_CNTL.sampling_pattern: a bit set for center, clear for
centroid, per interpolator), with PsParamGen and without.

The msaa suite's probes over a 32x32 window: pixel (X, Y)'s triangle is
(x0, y0), (x0 + s, y0), (x0, y0 + s) from the corner x0 = X - 0.5 +
(cx - 1) / 16, y0 likewise, of sub-pixel cell (cx, cy) = ((X + X // 16) % 16,
(Y + Y // 16) % 16), so it covers samples of its pixel or the ones beside
it: s = 6/16, at most a sample of a pixel, or in the 'large' tests 14/16,
several. Every vertex has interpolator 0 = (X, Y, 9i + 4, 0) for i = X + 32Y,
flat (SQ_INTERPOLATOR_CNTL.param_shade bit 0 clear), and interpolator 1 =
its position (x, y, 0, 1), smooth: interpolator 1 is the point the pixel
was evaluated at. Each pixel a triangle reaches exports to slot 9i +
3 (dy + 1) + dx + 1, (dx, dy) = floor(interpolator 1 + 0.5) - (X, Y):
element 0 interpolator 1, element 1 PsParamGen (r2), or interpolator 0
without it. Slots of pixels not reached keep the sentinel.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gpugen  # noqa: E402
import pm4  # noqa: E402
import xenos as x  # noqa: E402

WINDOW = 32
NEIGHBOURHOOD = 9
K = gpugen.C_FIRST_FREE  # pixel c8: (0.5, 3, 0, 0)
PARAM_GEN = 2  # PsParamGen's register, after the two interpolators
SCREEN_XY = gpugen.SCREEN_XY
# 8 VS and 4 PS registers, vertex fetch, two interpolators, Direct3D's PS
# export mode; PsParamGen on or off.
PROGRAM_CNTL = 7 | (3 << 8) | (1 << 16) | (1 << 20) | (2 << 27)
PROGRAM_PARAM_GEN = 1 << 18
SMOOTH = 0xFFFE  # interpolator 0 flat, the rest smooth
INTERPOLATOR_1 = 1 << 1
f2w = gpugen.f2w
# (name, RB_SURFACE_INFO.msaa_samples, PA_SC_AA_CONFIG) as Direct3D's.
MSAA = [('1x', 0, 0), ('2x', 1, 0x8001), ('4x', 2, 0xC003)]


def triangle(px, py, size):
    cx, cy = (px + px // 16) % 16, (py + py // 16) % 16
    x0, y0 = px - 0.5 + (cx - 1) / 16, py - 0.5 + (cy - 1) / 16
    return [(x0, y0), (x0 + size / 16, y0), (x0, y0 + size / 16)]


def inputs(size):
    words = []
    for py in range(WINDOW):
        for px in range(WINDOW):
            index = px + WINDOW * py
            for p in triangle(px, py, size):
                words += [f2w(v) for v in (p[0], p[1], 0.0, 1.0,
                                           px, py, 9 * index + 4, 0.0,
                                           p[0], p[1], 0.0, 1.0)]
    return words


def pixel_shader(param_gen):
    point = x.r(1)
    body = [
        # r3.xy = the pixel the point is in, less (X, Y).
        x.alu(vector_op=x.V_ADD, vector_dest=3, vector_mask=0x3,
              src1=point, src2=x.c(K, 'x')),
        x.alu(vector_op=x.V_FLOOR, vector_dest=3, vector_mask=0x3,
              src1=x.r(3), src2=x.r(3)),
        x.alu(vector_op=x.V_ADD, vector_dest=3, vector_mask=0x3,
              src1=x.r(3), src2=x.r(0, negate=True)),
        # r3.w = 9i + 4 + dx + 3 dy.
        x.alu(vector_op=x.V_ADD, vector_dest=3, vector_mask=0x8,
              src1=x.r(3, 'x'), src2=x.r(0, 'z')),
        x.alu(vector_op=x.V_MAD, vector_dest=3, vector_mask=0x8,
              src1=x.r(3, 'y'), src2=x.c(K, 'y'), src3=x.r(3, 'w'))]
    return gpugen.memexport_pixel_shader(
        body, [[gpugen.export_copy(1)],
               [gpugen.export_copy(PARAM_GEN if param_gen else 0)]],
        x.r(3, 'w'))


def build(suite):
    suite.set_inputs(inputs(6) + inputs(14), 12)
    suite.export_slots = WINDOW * WINDOW * NEIGHBOURHOOD
    count = 3 * WINDOW * WINDOW
    draws = {6: (pm4.PRIM_TRIANGLE_LIST, 0, count),
             14: (pm4.PRIM_TRIANGLE_LIST, count, count)}
    shader = gpugen.interpolating_vertex_shader(12)
    shaders = {True: pixel_shader(True), False: pixel_shader(False)}
    constants = {256 + K: [f2w(0.5), f2w(3.0), 0, 0]}

    def test(name, samples, aa_config, sample_cntl, pattern, param_gen=True,
             size=6):
        misc = (sample_cntl << 2) | SCREEN_XY | (
            PARAM_GEN << 8 if param_gen else 0)
        suite.test(name, shader, constants=constants, elements=2,
                   pixel_shader=shaders[param_gen],
                   program_cntl=PROGRAM_CNTL | (
                       PROGRAM_PARAM_GEN if param_gen else 0),
                   registers=[
                       ('SQ_CONTEXT_MISC', [misc]),
                       ('SQ_INTERPOLATOR_CNTL', [SMOOTH | pattern << 16]),
                       ('PA_SC_WINDOW_SCISSOR_BR', [(WINDOW << 16) | WINDOW]),
                       ('RB_COLOR_MASK', [0xF]),
                       ('RB_SURFACE_INFO', [1280 | (samples << 16)]),
                       ('PA_SU_SC_MODE_CNTL',
                        [gpugen.MODE_CNTL |
                         (gpugen.MSAA_ENABLE if samples else 0)]),
                       ('PA_SC_AA_CONFIG', [aa_config])],
                   draw=draws[size])

    X1, X4 = MSAA[0][1:], MSAA[2][1:]
    patterns = (('centroid', 0), ('center', 0xFFFF))
    # Every sc_sample_cntl with every interpolator's sampling_pattern bit
    # clear and set.
    for pattern_name, pattern in patterns:
        test(f'1x pattern {pattern_name}', *X1, 2, pattern)
    for name, samples, aa_config in MSAA[1:]:
        for sample_cntl in range(3):
            for pattern_name, pattern in patterns:
                test(f'{name} pattern {pattern_name} sample_cntl '
                     f'{sample_cntl}', samples, aa_config, sample_cntl,
                     pattern)
    # Which interpolator a sampling_pattern bit is.
    test('4x pattern center interpolator 1 only', *X4, 2, INTERPOLATOR_1)
    test('4x pattern center all but interpolator 1', *X4, 2,
         0xFFFF & ~INTERPOLATOR_1)
    # Without PsParamGen, as most titles' MSAA draws.
    for sample_cntl in range(3):
        test(f'4x pattern center sample_cntl {sample_cntl} no param_gen',
             *X4, sample_cntl, 0xFFFF, param_gen=False)
    for sample_cntl in (0, 2):
        test(f'4x pattern centroid sample_cntl {sample_cntl} no param_gen',
             *X4, sample_cntl, 0, param_gen=False)
    # Triangles 14/16 of a pixel, covering several samples of one.
    for name, samples, aa_config in MSAA[1:]:
        for sample_cntl in (0, 2):
            for pattern_name, pattern in patterns:
                test(f'{name} pattern {pattern_name} sample_cntl '
                     f'{sample_cntl} large', samples, aa_config, sample_cntl,
                     pattern, size=14)
