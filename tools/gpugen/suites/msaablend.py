"""Blending at 1x and at 2x and 4x MSAA into k_8_8_8_8 and k_2_10_10_10
render targets, resolved averaging the samples (01 at 2x, 0123 at 4x) and
converted to the target's format. Each case is a pixel: a first draw stores
its destination unblended, then its draws (0 to 3, each a rectangle over the
pixel) write its source through its RB_BLENDCONTROL0, RB_COLOR_MASK and
RB_BLEND_RED..ALPHA. The pixel shader is the rb suite's: sources and
destinations come from textures, the destination as packed levels k / steps
drawn as (k + 1/4) / steps, which store k however the render backend rounds.

- 'original': source (0.73, 0.17, 0.53, a) for alphas a in sixteenths,
  destination (0.2, 0.4, 0.6, 0.8) packed as a clear rounds it,
  constant (0.3, 0.6, 0.2, 0.7), each of 8 blends (replace, alpha, add,
  multiply, MIN and MAX of source alpha and one, reverse subtract, constant)
  with one and three draws, and a destination-only case.
- 'random': RGB and alpha equations and all 15 factors chosen independently,
  write masks 0 to 15, 0 to 3 draws, sources and constants in [-0.25, 1.25),
  destinations random levels.
Pixels hold the cases grouped by state, in order of first appearance.
"""

import os
import random
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpugen  # noqa: E402
import rb  # noqa: E402
import rbdraw  # noqa: E402

WIDTH, HEIGHT = rb.WIDTH, rb.HEIGHT
NO_BLEND = 0x00010001
FACTORS = (0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
# (name, RB_COLOR_INFO format, resolve format, levels of R, G, B, A)
FORMATS = [('8_8_8_8', 0, 6, (255, 255, 255, 255)),
           ('2_10_10_10', 2, 7, (1023, 1023, 1023, 3))]
# The 1x, 2x and 4x PA_SC_AA_CONFIG codes and resolve sample selects.
SAMPLES = [(1, 0, 0), (2, 1, 4), (4, 2, 6)]
RANDOM_CASES = 512
f2w = gpugen.f2w


def f32(v):
    return struct.unpack('>f', struct.pack('>f', v))[0]


def unorm(value, steps):
    """A clear's packing: the clamped float times the steps, plus a half,
    truncated, in float32."""
    v = min(max(f32(value), 0.0), 1.0)
    return int(f32(f32(v * steps) + 0.5))


def blend(op, source, dest, alpha_op, alpha_source, alpha_dest):
    return (source | op << 5 | dest << 8 | alpha_source << 16 |
            alpha_op << 21 | alpha_dest << 24)


def case(control, mask, draws, source, levels, constant):
    return {'state': (control, mask, draws, tuple(constant)),
            'source': [f32(v) for v in source], 'levels': list(levels)}


def original(steps):
    blends = [(0, 1, 0), (0, 6, 7), (0, 1, 1), (0, 8, 0), (2, 6, 1),
              (3, 6, 1), (4, 1, 1), (0, 12, 13)]
    levels = [unorm(v, s) for v, s in zip((0.2, 0.4, 0.6, 0.8), steps)]
    constant = [f32(v) for v in (0.3, 0.6, 0.2, 0.7)]
    cases = [case(NO_BLEND, 15, 0, (0.73, 0.17, 0.53, 0.25), levels,
                  constant)]
    for a in range(17):
        for b in blends:
            for draws in (1, 3):
                cases.append(case(blend(*b, *b), 15, draws,
                                  (0.73, 0.17, 0.53, a / 16), levels,
                                  constant))
    return cases


def random_cases(rng, steps):
    cases = []
    for _ in range(RANDOM_CASES):
        control = blend(rng.randrange(5), rng.choice(FACTORS),
                        rng.choice(FACTORS), rng.randrange(5),
                        rng.choice(FACTORS), rng.choice(FACTORS))
        mask, draws = rng.randrange(16), rng.randrange(4)
        source = [rng.uniform(-0.25, 1.25) for _ in range(4)]
        levels = [rng.randrange(s + 1) for s in steps]
        constant = [f32(rng.uniform(-0.25, 1.25)) for _ in range(4)]
        cases.append(case(control, mask, draws, source, levels, constant))
    return cases


def build(suite):
    rng = random.Random(0xB1E2)
    shapes = rbdraw.Shapes()

    sets = {}
    for fname, rt_format, dest_format, steps in FORMATS:
        sets[fname, 'original'] = original(steps)
    for fname, rt_format, dest_format, steps in FORMATS:
        sets[fname, 'random'] = random_cases(rng, steps)
    draw_sets = {}
    for (fname, name), cases in sets.items():
        steps = dict((f[0], f[3]) for f in FORMATS)[fname]
        order = rbdraw.grouped([c['state'] for c in cases])
        source, dest = [0] * (4 * WIDTH * HEIGHT), [0] * (4 * WIDTH * HEIGHT)
        for pixel, i in enumerate(order):
            source[4 * pixel:4 * pixel + 4] = map(f2w, cases[i]['source'])
            dest[4 * pixel:4 * pixel + 4] = [
                f2w((k + 0.25) / s) for k, s in zip(cases[i]['levels'], steps)]
        suite.add_texture(f'{fname} {name} source', source)
        suite.add_texture(f'{fname} {name} dest', dest)
        draws = [([('RB_BLENDCONTROL0', [control]), ('RB_COLOR_MASK', [mask]),
                   ('RB_BLEND_RED', list(map(f2w, constant)))],
                  shapes.pixels(first, count, n))
                 for (control, mask, n, constant), first, count in rbdraw.runs(
                     [cases[i]['state'] for i in order]) if n]
        draw_sets[fname, name] = draws

    for samples, code, select in SAMPLES:
        for fname, rt_format, dest_format, steps in FORMATS:
            resolve = gpugen.resolve_registers(
                WIDTH, HEIGHT, dest_format, 4, sample_select=select)
            state = rb.registers(rt_format, code, mode_cntl=gpugen.MODE_CNTL)
            for name in ('original', 'random'):
                draws = draw_sets[fname, name]
                passes = [([('RB_BLENDCONTROL0', [NO_BLEND])],
                           rb.fetch(f'{fname} {name} dest'), rb.RECT_DRAW)]
                passes += [(registers, rb.fetch(f'{fname} {name} source')
                            if not k else {}, draw)
                           for k, (registers, draw) in enumerate(draws[:-1])]
                registers, draw = draws[-1]
                rb.test(suite, f'{samples}x {fname} {name}',
                        f'{fname} {name} source', state, resolve, draw=draw,
                        passes=passes, after_passes=registers)
    suite.set_inputs(shapes.inputs, 2)
