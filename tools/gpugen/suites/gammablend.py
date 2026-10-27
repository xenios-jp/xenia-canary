"""k_8_8_8_8_GAMMA render targets at 1x, resolved into k_8_8_8_8 ('8') or
k_2_10_10_10 ('10', which keeps every linear value apart), as the gammamsaa
suite draws them: each case a
pixel, its destination bytes stored raw by a draw of (byte + 1/4) / 255 with
the target's format k_8_8_8_8, then with the format back to gamma its draws
of its source (linear, from a texture through the rb suite's pixel shader)
through its RB_BLENDCONTROL0 and RB_BLEND_RED..ALPHA, each draw a draw of its
own.

The cases:
- '8': clears of every byte; unblended encodes of linear sources in quarter
  steps of 1/1023; ONE + ONE additions onto every byte; SRC_ALPHA,
  INV_SRC_ALPHA at alpha 0.5 onto every byte; repeated additions.
- '10': clears and encodes again; the source and the destination term alone
  by every alpha byte; alphas between bytes; both terms; every blend factor
  on each side, with a constant; SUBTRACT, MIN, MAX and REVERSE_SUBTRACT;
  repeated alpha blends.
Each set's tests take up to 2048 cases in order, the pixels holding them
grouped by state.
"""

import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gammamsaa  # noqa: E402
import gpugen  # noqa: E402
import rb  # noqa: E402
import rbdraw  # noqa: E402

WIDTH, HEIGHT = rb.WIDTH, rb.HEIGHT
GAMMA = gammamsaa.GAMMA
ADD, SUBTRACT, MIN, MAX, REVERSE_SUBTRACT = 0, 1, 2, 3, 4
ZERO, ONE, SRC_ALPHA, ONE_MINUS_SRC_ALPHA = 0, 1, 6, 7
FACTORS = (0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
QUARTER = gammamsaa.QUARTER
SOURCES = (0, 1, 2, 3, 63, 64, 127, 128, 255, 256, 383, 511, 512, 640, 767,
           1022, 1023)
DESTINATIONS = (0, 1, 2, 32, 63, 64, 65, 96, 127, 128, 160, 191, 192, 200, 224,
                254, 255)
FEW_SOURCES = gammamsaa.FEW_SOURCES
FEW_DESTINATIONS = gammamsaa.FEW_DESTINATIONS
FEW_ALPHAS = gammamsaa.FEW_ALPHAS
# (name, resolve texture format)
RESOLVES = gammamsaa.RESOLVES
f2w = gpugen.f2w


def case(source, clear, color=(ADD, ONE, ZERO), alpha=(ADD, ONE, ZERO),
         draws=1, readback=0, constant=(0.0, 0.0, 0.0, 0.0)):
    """A case: its words, source, clear and constant."""
    return {'words': [GAMMA, *color, *alpha, 15, draws, 1, readback, 0],
            'source': list(source), 'clear': list(clear),
            'constant': list(constant)}


def words(c):
    """The case as 24 words."""
    return struct.unpack('>24I', struct.pack(
        '>12I12f', *c['words'], *c['source'], *c['clear'], *c['constant']))


def eight():
    out = []
    for g in range(256):
        out.append(case((0, 0, 0, 0), (g / 255.0,) * 4, draws=0))
    for j in range(1023 * 4 + 1):
        s = min(j * QUARTER, 1.0)
        out.append(case((s, s, s, 1.0), (0, 0, 0, 0)))
    add = ((ADD, ONE, ONE), (ADD, ONE, ONE))
    for g in range(256):
        for j in range(17):
            s = j * QUARTER
            out.append(case((s, s, s, 0.0), (g / 255.0,) * 4, *add))
    over = ((ADD, SRC_ALPHA, ONE_MINUS_SRC_ALPHA), (ADD, ONE, ZERO))
    for g in range(256):
        for s in (0.0, 0.25, 0.75):
            out.append(case((s, s, s, 0.5), (g / 255.0,) * 4, *over))
    for j in (1, 2, 3, 5, 6, 7):
        for draws in (2, 3, 4, 8):
            s = j * QUARTER
            out.append(case((s, s, s, 0.0), (0, 0, 0, 0), *add,
                            draws=draws))
    return out


def ten():
    out = []
    t = {'readback': 1}
    for g in range(256):
        out.append(case((0, 0, 0, 0), (g / 255.0,) * 4, draws=0, **t))
    for j in range(1023 * 4 + 1):
        s = min(j * QUARTER, 1.0)
        out.append(case((s, s, s, 1.0), (0, 0, 0, 0), **t))
    for a in range(256):
        for k in SOURCES:
            out.append(case((k / 1023.0,) * 3 + (a / 255.0,), (0, 0, 0, 0),
                            (ADD, SRC_ALPHA, ZERO), **t))
    for a in range(256):
        for g in DESTINATIONS:
            out.append(case((0, 0, 0, a / 255.0), (g / 255.0,) * 4,
                            (ADD, ZERO, ONE_MINUS_SRC_ALPHA), **t))
    for a in range(255):
        for f in (0.25, 0.5, 0.75):
            for k in (511, 1023):
                out.append(case((k / 1023.0,) * 3 + ((a + f) / 255.0,),
                                (0, 0, 0, 0), (ADD, SRC_ALPHA, ZERO), **t))
    for a in FEW_ALPHAS:
        for k in FEW_SOURCES:
            for g in FEW_DESTINATIONS:
                out.append(case((k / 1023.0,) * 3 + (a / 255.0,),
                                (g / 255.0,) * 4,
                                (ADD, SRC_ALPHA, ONE_MINUS_SRC_ALPHA), **t))
    constant = (0.375, 0.375, 0.375, 0.625)
    for f in FACTORS:
        for k in FEW_SOURCES:
            for g in FEW_DESTINATIONS:
                out.append(case((k / 1023.0,) * 3 + (0.75,),
                                (g / 255.0,) * 4, (ADD, f, ZERO),
                                constant=constant, **t))
                out.append(case((k / 1023.0,) * 3 + (0.75,),
                                (g / 255.0,) * 4, (ADD, ZERO, f),
                                constant=constant, **t))
    for op in (SUBTRACT, MIN, MAX, REVERSE_SUBTRACT):
        for k in SOURCES:
            for g in DESTINATIONS:
                out.append(case((k / 1023.0,) * 3 + (0.0,), (g / 255.0,) * 4,
                                (op, ONE, ONE), **t))
    for a in (64, 128, 192):
        for k in (64, 511, 1023):
            for g in (0, 128):
                for draws in (2, 4, 8):
                    out.append(case((k / 1023.0,) * 3 + (a / 255.0,),
                                    (g / 255.0,) * 4,
                                    (ADD, SRC_ALPHA, ONE_MINUS_SRC_ALPHA),
                                    draws=draws, **t))
    return out


SETS = {'8': eight, '10': ten}


def control(c):
    """The case's RB_BLENDCONTROL0."""
    color_op, color_src, color_dest, alpha_op, alpha_src, alpha_dest = (
        c['words'][1:7])
    return (color_src | color_op << 5 | color_dest << 8 | alpha_src << 16 |
            alpha_op << 21 | alpha_dest << 24)


def state(c):
    """(RB_BLENDCONTROL0, RB_BLEND_RED..ALPHA words, draws)."""
    return (control(c), tuple(f2w(v) for v in c['constant']), c['words'][8])


def stored(clear):
    """The bytes a clear to `clear` stores."""
    return [round(v * 255) for v in clear]


def build(suite):
    shapes = rbdraw.Shapes()
    whole = rb.RECT_DRAW
    for rname, dest_format in RESOLVES:
        resolve = gpugen.resolve_registers(WIDTH, HEIGHT, dest_format, 4)
        registers = rb.registers(GAMMA, mode_cntl=gpugen.MODE_CNTL)
        all_cases = SETS[rname]()
        for part in range(0, len(all_cases), WIDTH * HEIGHT):
            chunk = all_cases[part:part + WIDTH * HEIGHT]
            name = f'{rname} {part // (WIDTH * HEIGHT)}'
            states = [state(c) for c in chunk]
            order = rbdraw.grouped(states)
            source, dest = ([0] * (4 * WIDTH * HEIGHT),
                            [0] * (4 * WIDTH * HEIGHT))
            for pixel, i in enumerate(order):
                source[4 * pixel:4 * pixel + 4] = map(f2w, chunk[i]['source'])
                dest[4 * pixel:4 * pixel + 4] = [
                    f2w((g + 0.25) / 255) for g in stored(chunk[i]['clear'])]
            suite.add_texture(f'{name} source', source)
            suite.add_texture(f'{name} dest', dest)
            passes = [([('RB_COLOR_INFO', [0]),
                        ('RB_BLENDCONTROL0', [gammamsaa.NO_BLEND]),
                        ('RB_COLORCONTROL', [gammamsaa.COLORCONTROL])],
                       rb.fetch(f'{name} dest'), whole)]
            draws = []
            for (blend, constant, n), first, count in rbdraw.runs(
                    [states[i] for i in order]):
                for _ in range(n):
                    draws.append(([('RB_COLOR_INFO', [GAMMA << 16]),
                                   ('RB_BLENDCONTROL0', [blend]),
                                   ('RB_BLEND_RED', list(constant))],
                                  shapes.pixels(first, count)))
            passes += [(r, rb.fetch(f'{name} source') if not k else {}, draw)
                       for k, (r, draw) in enumerate(draws[:-1])]
            last, draw = draws[-1]
            rb.test(suite, name, f'{name} source', registers, resolve,
                    draw=draw, passes=passes, after_passes=last)
    suite.set_inputs(shapes.inputs, 2)
