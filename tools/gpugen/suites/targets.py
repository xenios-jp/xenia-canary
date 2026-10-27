"""Render targets and resolves games use that no other suite draws: the color
target formats, sample counts and exponent biases (RB_COLOR_INFO bits 20-25)
seen in traces of titles, each read back raw and through the resolves titles
were seen making of it (RB_COPY_DEST_INFO format, number format and exponent
bias).

Each config draws 64x32 pixels into its target:
- rows 0-15, write cases: one unblended draw stores each pixel's value from a
  texture, rb.SPECIALS (rotated through the channels) then random values of
  the format's range: unorm [-0.25, 1.25], 16-bit fixed point [-40, 40],
  float of every exponent and sign;
- rows 16-31, blend cases: the same draw stores a destination, then 64
  random blend states (msaablend's generator: RB_BLENDCONTROL0, RB_COLOR_MASK,
  RB_BLEND_RED..ALPHA, 1 to 3 draws) each over 16 pixels, sources from a
  second texture.
Readbacks, a test each over the same draws: 'raw' copies sample 0's EDRAM bits
(copy command 0 to k_8_8_8_8, or k_16_16_16_16 for 64-bit pixels), then each
resolve, averaging the samples at 2x and 4x.
"""

import os
import random
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpugen  # noqa: E402
import depth  # noqa: E402
import msaablend  # noqa: E402
import rb  # noqa: E402
import rbdraw  # noqa: E402
import stencil  # noqa: E402

WIDTH, HEIGHT = rb.WIDTH, rb.HEIGHT
STATES, PIXELS_PER_STATE = 64, 16
WRITE_ROWS = 16
UNORM, FIXED, FLOAT = 'unorm', 'fixed', 'float'
# ColorRenderTargetFormat: (name, bytes per pixel, value range).
TARGETS = {0: ('8_8_8_8', 4, UNORM), 1: ('8_8_8_8_GAMMA', 4, UNORM),
           2: ('2_10_10_10', 4, UNORM), 3: ('2_10_10_10_FLOAT', 4, FLOAT),
           4: ('16_16', 4, FIXED),
           5: ('16_16_16_16', 8, FIXED), 7: ('16_16_16_16_FLOAT', 8, FLOAT),
           10: ('2_10_10_10_AS_10_10_10_10', 4, UNORM),
           12: ('2_10_10_10_FLOAT_AS_16_16_16_16', 4, FLOAT),
           14: ('32_FLOAT', 4, FLOAT)}
# Resolve texture formats: (name, bytes per texel).
DESTS = {2: ('8', 1), 6: ('8_8_8_8', 4), 7: ('2_10_10_10', 4),
         15: ('4_4_4_4', 2), 25: ('16_16', 4), 26: ('16_16_16_16', 8),
         32: ('16_16_16_16_FLOAT', 8), 36: ('32_FLOAT', 4)}
UINT, FLOAT_NUMBER = 2, 7
# (target format, samples, color exponent bias, [(dest format, number format,
# resolve exponent bias, sample or None to average)]): the color target
# formats, sample counts and exponent biases titles use that no other suite
# draws, and every resolve titles were seen making of them.
AVERAGE = None
CONFIGS = [
    (0, 1, 0, [(6, 0, 0, 0), (2, 0, 0, 0), (7, 0, 0, 0)]),
    (0, 2, 0, [(6, 0, 0, AVERAGE), (6, 0, 0, 0)]),
    (0, 4, 0, [(6, 0, 0, AVERAGE), (15, 0, 0, AVERAGE), (2, 0, 0, AVERAGE)]),
    (1, 1, 0, [(6, 0, 0, 0), (7, 0, 0, 0)]),
    (1, 4, 0, [(6, 0, 0, AVERAGE), (7, 0, 0, AVERAGE)]),
    (2, 1, 0, [(7, 0, 0, 0)]),
    (3, 1, 0, [(32, FLOAT_NUMBER, e, 0) for e in (0, -4, -3, 1)] +
     [(26, 0, -5, 0)]),
    (3, 2, 0, [(6, 0, -4, AVERAGE)] +
     [(32, FLOAT_NUMBER, e, AVERAGE) for e in (-4, -3, -2)]),
    (3, 4, 0, [(32, FLOAT_NUMBER, e, AVERAGE) for e in (0, 1)] +
     [(26, 0, -5, AVERAGE)]),
    (4, 1, 0, [(25, 0, 0, 0)]),
    (4, 4, 0, [(25, 0, -5, AVERAGE)]),
    (5, 1, 0, [(26, 0, -5, 0), (26, 0, 0, 0), (32, FLOAT_NUMBER, 0, 0)]),
    (5, 1, 2, [(26, 0, -5, 0)]),
    (5, 1, 5, [(26, 0, -5, 0)]),
    (5, 4, 0, [(26, 0, 0, AVERAGE)]),
    (7, 1, 0, [(32, FLOAT_NUMBER, 0, 0)]),
    (7, 4, 0, [(32, FLOAT_NUMBER, 0, AVERAGE)]),
    (10, 1, 0, [(7, 0, 0, 0)]),
    (10, 4, 0, [(7, 0, 0, AVERAGE)]),
    (12, 1, 0, [(7, 0, -5, 0), (6, 0, -4, 0), (6, 0, 0, 0)] +
     [(32, FLOAT_NUMBER, e, 0) for e in (-3, 1)]),
    (12, 1, 4, [(6, 0, 0, 0)]),
    (12, 1, 5, [(6, 0, 0, 0)]),
    (12, 1, -1, [(6, 0, 0, 0)]),
    (12, 2, 0, [(32, FLOAT_NUMBER, -4, AVERAGE)]),
    (12, 2, 4, [(32, FLOAT_NUMBER, -4, AVERAGE)]),
    (12, 4, 0, [(32, FLOAT_NUMBER, e, AVERAGE) for e in (0, -4)]),
    (12, 4, 4, [(32, FLOAT_NUMBER, -4, AVERAGE)]),
    (14, 1, 0, [(36, FLOAT_NUMBER, 0, 0)]),
    (14, 4, 0, [(36, FLOAT_NUMBER, 0, AVERAGE)]),
]
# (depth format, samples, [number format]): the depth resolves titles make,
# raw copies of sample 0 to k_8_8_8_8.
DEPTH_CONFIGS = [(0, 1, [0]), (1, 1, [2]), (0, 2, [0]), (1, 2, [0, 2]),
                 (0, 4, [0]), (1, 4, [2])]
# PA_SC_AA_CONFIG codes and the averaging sample selects.
SAMPLES = {1: (0, 0), 2: (1, 4), 4: (2, 6)}
PARTS = 4
f2w = gpugen.f2w


def f32(v):
    return struct.unpack('>f', struct.pack('>f', v))[0]


def value(rng, kind):
    if kind == UNORM:
        return rng.uniform(-0.25, 1.25)
    if kind == FIXED:
        return rng.uniform(-40.0, 40.0)
    return (rng.choice((-1, 1)) * rng.uniform(1, 2) *
            2.0 ** rng.randint(-20, 20))


def words(rng, kind, specials):
    """A texture's words: the write rows' values, specials first, then the
    blend rows'."""
    out = []
    for p in range(WIDTH * HEIGHT):
        if specials and p < len(rb.SPECIALS):
            out += [rb.SPECIALS[(p + k * 16) % len(rb.SPECIALS)]
                    for k in range(4)]
        else:
            out += [f2w(f32(value(rng, kind))) for _ in range(4)]
    return out


def name(fmt, samples, bias):
    return (f'{TARGETS[fmt][0]} {samples}x' +
            (f' exp {bias}' if bias else ''))


def color_tests(suite, shapes, index, fmt, samples, bias, resolves):
    rng = random.Random(0x7A4E + index)
    target, bpp, kind = TARGETS[fmt]
    config = name(fmt, samples, bias)
    suite.add_texture(f'{config} dest', words(rng, kind, True))
    suite.add_texture(f'{config} source', words(rng, kind, False))
    states = msaablend.random_cases(rng, (1, 1, 1, 1))[:STATES]
    color_info = [('RB_COLOR_INFO', [fmt << 16 | (bias & 0x3F) << 20])]
    draws = [(color_info + [('RB_BLENDCONTROL0', [msaablend.NO_BLEND]),
                            ('RB_COLOR_MASK', [0xF])],
              rb.fetch(f'{config} dest'), rb.RECT_DRAW)]
    fetched = False
    for i, c in enumerate(states):
        control, mask, n, constant = c['state']
        draws.append(([('RB_BLENDCONTROL0', [control]),
                       ('RB_COLOR_MASK', [mask]),
                       ('RB_BLEND_RED', list(map(f2w, constant)))],
                      {} if fetched else rb.fetch(f'{config} source'),
                      shapes.pixels(WIDTH * WRITE_ROWS + i * PIXELS_PER_STATE,
                                    PIXELS_PER_STATE, max(n, 1))))
        fetched = True
    code, select = SAMPLES[samples]
    state = rb.registers(fmt, code, mode_cntl=gpugen.MODE_CNTL)
    readbacks = [('raw', gpugen.resolve_registers(
        WIDTH, HEIGHT, 6 if bpp == 4 else 26, bpp, command=0))]
    for dest, number, exp, sample in resolves:
        readbacks.append((
            f'{DESTS[dest][0]}' + (f' num {number}' if number else '') +
            (f' exp {exp}' if exp else '') +
            (f' sample {sample}' if samples > 1 and sample is not None
             else ''),
            gpugen.resolve_registers(
                WIDTH, HEIGHT, dest, DESTS[dest][1], number,
                sample_select=select if sample is None else sample,
                exp_bias=exp)))
    registers, _, draw = draws[-1]
    for readback, resolve in readbacks:
        rb.test(suite, f'{config} {readback}', f'{config} source', state,
                resolve, draw=draw, passes=draws[:-1], after_passes=registers)


def depth_tests(suite, index, fmt, samples, numbers):
    """The depth suite's texture drawn into the depth buffer through oDepth
    (depth ALWAYS, written), then raw copies of sample 0."""
    texture = f'depth {index}'
    suite.add_texture(texture,
                      depth.depth_texture(random.Random(0xD3 + index)))
    code, _ = SAMPLES[samples]
    state = rb.registers(0, code, depth_format=fmt,
                         mode_cntl=gpugen.MODE_CNTL)
    for number in numbers:
        rb.test(suite, f'{("D24S8", "D24FS8")[fmt]} {samples}x' +
                (f' num {number}' if number else ''), texture, state,
                gpugen.resolve_registers(WIDTH, HEIGHT, 6, 4, number,
                                         command=0, source=4),
                pixel=rb.pixel_shader(depth=True),
                program_cntl=stencil.PROGRAM_CNTL)


def build(suite, part):
    """Part `part` of PARTS: the configs split to fit the ring, each built
    from its own seed."""
    shapes = rbdraw.Shapes()
    items = ([(color_tests, c) for c in CONFIGS] +
             [(depth_tests, c) for c in DEPTH_CONFIGS])
    for index, (tests, config) in enumerate(items):
        if index * PARTS // len(items) == part - 1:
            if tests is color_tests:
                tests(suite, shapes, index, *config)
            else:
                tests(suite, index, *config)
    suite.set_inputs(shapes.inputs, 2)
