"""Per-fragment float10 storage under repeated overlapping primitives.

Compare one rectangle-list draw containing repeated overlapping rectangles
with the same rectangles submitted as separate draws. Both guest float10
formats are covered at 1x/2x/4x MSAA. Signed texture values exercise stored
clamping and quantization between fragments, rather than a post-draw fix.
Full-target coverage keeps sample coverage independent of edge locations.
"""
import os
import random
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blend  # noqa: E402
import gpugen  # noqa: E402
import rb  # noqa: E402
import rbdraw  # noqa: E402


def build(suite):
    shapes = rbdraw.Shapes()
    rng = random.Random(0xB10F)
    edges = [-2., -.51, -.125, -.003, 0., .003, .125, .501,
             .999, 1.001, 3.999, 15.9, 31.8, 32.1]
    for name, offset in [('source', 0), ('destination', 5)]:
        words = []
        for i in range(rb.WIDTH * rb.HEIGHT):
            values = [edges[(i + c * 3 + offset) % len(edges)]
                      if i < 196 else rng.uniform(-2, 34) for c in range(4)]
            words.extend(map(gpugen.f2w, values))
        suite.add_texture(name, words)
    for fmt in (3, 12):
        for samples, code, select in ((1, 0, 0), (2, 1, 4), (4, 2, 6)):
            state = rb.registers(fmt, code, mode_cntl=gpugen.MODE_CNTL)
            resolve = gpugen.resolve_registers(
                rb.WIDTH, rb.HEIGHT, 32, 8, 7, sample_select=select)
            for mode, control in blend.MODES[:2] + blend.MODES[3:4]:
                for repeats in (3, 17):
                    for topology in ('one draw', 'separate draws'):
                        passes = [([('RB_BLENDCONTROL0', [blend.NO_BLEND])],
                                   rb.fetch('destination'), rb.RECT_DRAW)]
                        registers = [('RB_BLENDCONTROL0', [control])]
                        if topology == 'one draw':
                            draw = shapes.pixels(0, rb.WIDTH * rb.HEIGHT, repeats)
                        else:
                            draw = rb.RECT_DRAW
                            passes += [(registers, rb.fetch('source'), draw)
                                       for _ in range(repeats - 1)]
                        rb.test(suite, f'format{fmt} {samples}x {mode} '
                                f'{repeats} overlaps {topology}', 'source', state,
                                resolve, draw=draw, passes=passes,
                                after_passes=registers)
    suite.set_inputs(shapes.inputs, 2)
