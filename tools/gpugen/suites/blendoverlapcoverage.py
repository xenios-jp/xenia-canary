"""Sample-distinct destinations seeded by alpha-to-coverage.

Descending prefix coverage writes each sample's RGB independently: first all
samples, then all but the last, ending with sample zero. Symmetric coverage
thresholds are already measured by the merger suites. Alpha is coupled to
coverage and is checked against console readbacks, not assumed independent.
Every resolve reads one sample, and initialization-only tests separate seed
failures from overlapping-primitive blending failures. PA_SC_AA_MASK stays
fully enabled, so unsupported explicit sample masks cannot obscure this test.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blend  # noqa: E402
import gpugen  # noqa: E402
import rb  # noqa: E402
import rbdraw  # noqa: E402


REPEATS = 5
SAMPLES = ((2, 1), (4, 2))
MODES = blend.MODES[:2] + blend.MODES[3:4]


def build(suite):
    shapes = rbdraw.Shapes()
    count = rb.WIDTH * rb.HEIGHT
    suite.add_texture('clear', [0] * (count * 4))
    source = []
    for i in range(count):
        # Include negative RGB without saturating every repeated result.
        source.extend(map(gpugen.f2w, (
            (-0.125, 0.0625, 0.375, 1.125)[i % 4],
            (0.03125, -0.0625, 0.25, 0.75)[(i // 4) % 4],
            (0.125, 0.5, -0.25, 1.5)[(i // 16) % 4],
            (0.125, 0.25, 0.375, 0.5)[(i // 64) % 4])))
    suite.add_texture('source', source)
    for samples, _ in SAMPLES:
        for sample in range(samples):
            words = []
            for i in range(count):
                words.extend(map(gpugen.f2w, (
                    0.5 + sample * 1.25 + (i % 4) / 16,
                    6.0 - sample * 0.75 + ((i // 4) % 4) / 8,
                    1.0 + sample * 2.0 + ((i // 16) % 4) / 4,
                    (sample + 1) / samples)))
            suite.add_texture(f'destination{samples}-{sample}', words)

    for fmt in (3, 12):
        for samples, code in SAMPLES:
            state = rb.registers(fmt, code, mode_cntl=gpugen.MODE_CNTL)
            initialize = [([('RB_BLENDCONTROL0', [blend.NO_BLEND]),
                            ('PA_SC_AA_MASK', [0xFFFF]),
                            ('RB_COLORCONTROL', [0x87000007])],
                           rb.fetch('clear'), rb.RECT_DRAW)]
            initialize += [([('RB_BLENDCONTROL0', [blend.NO_BLEND]),
                             ('RB_COLORCONTROL', [0xAA000017])],
                            rb.fetch(f'destination{samples}-{sample}'),
                            rb.RECT_DRAW)
                           for sample in reversed(range(samples))]
            for sample in range(samples):
                resolve = gpugen.resolve_registers(
                    rb.WIDTH, rb.HEIGHT, 32, 8, 7, sample_select=sample)
                # The last draw writes nothing, preserving all seeded samples.
                rb.test(suite, f'format{fmt} {samples}x initialize sample{sample}',
                        'clear', state, resolve, passes=initialize,
                        after_passes=[('RB_COLOR_MASK', [0]),
                                      ('PA_SC_AA_MASK', [0xFFFF]),
                                      ('RB_COLORCONTROL', [0x87000007])])
                for mode, control in MODES:
                    registers = [('RB_BLENDCONTROL0', [control]),
                                 ('PA_SC_AA_MASK', [0xFFFF]),
                                 ('RB_COLORCONTROL', [0x87000007])]
                    for topology in ('one draw', 'separate draws'):
                        passes = list(initialize)
                        if topology == 'one draw':
                            draw = shapes.pixels(0, count, REPEATS)
                        else:
                            draw = rb.RECT_DRAW
                            passes += [(registers, rb.fetch('source'), draw)
                                       for _ in range(REPEATS - 1)]
                        rb.test(suite, f'format{fmt} {samples}x {mode} '
                                f'{REPEATS} overlaps {topology} sample{sample}',
                                'source', state, resolve, draw=draw,
                                passes=passes, after_passes=registers)
    suite.set_inputs(shapes.inputs, 2)
