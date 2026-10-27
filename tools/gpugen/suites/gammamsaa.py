"""k_8_8_8_8_GAMMA render targets at 2x and 4x MSAA, resolved averaging the
samples (01 at 2x, 0123 at 4x) into k_8_8_8_8 ('8') or k_2_10_10_10 ('10',
which keeps every linear value apart). Each case is a pixel: its destination
bytes stored raw by a draw of (byte + 1/4) / 255 with the target's format
k_8_8_8_8, then with the format back to gamma
its draws of its source (linear, from a texture through the rb suite's pixel
shader) through its blend and alpha to mask.

The cases, per sample count and resolve format: clears of every byte; unblended
encodes of linear sources in 1/1023 steps; replacing sources whose alpha,
through alpha to mask at Direct3D's symmetric offsets (0xAA), covers 1 of 2 or
1 to 3 of 4 samples, over destinations; and at '10' alpha blending (SRC_ALPHA,
INV_SRC_ALPHA) of sources over destinations. Each (samples, resolve) set's
tests take up to 2048 cases in order, the pixels holding them grouped by state.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpugen  # noqa: E402
import rb  # noqa: E402
import rbdraw  # noqa: E402

WIDTH, HEIGHT = rb.WIDTH, rb.HEIGHT
GAMMA = 1
ADD, ZERO, ONE, SRC_ALPHA, INV_SRC_ALPHA = 0, 0, 1, 6, 7
QUARTER = 1.0 / (1023.0 * 4.0)
COVERAGE_ALPHAS = {2: {1: 0.5}, 4: {1: 0.25, 2: 0.5, 3: 0.75}}
MSAA_SOURCES = tuple(range(0, 1024, 26)) + (1, 2, 63, 64, 127, 128, 511, 512,
                                            1022, 1023)
MSAA_DESTINATIONS = tuple(range(0, 256, 11)) + (1, 63, 64, 95, 96, 191, 192,
                                                254, 255)
FEW_SOURCES = (0, 1, 64, 256, 511, 512, 767, 1022, 1023)
FEW_DESTINATIONS = (0, 1, 63, 64, 96, 128, 192, 224, 255)
FEW_ALPHAS = (0, 1, 2, 32, 64, 85, 127, 128, 129, 170, 191, 192, 200, 253, 254,
              255)
NO_BLEND = 0x00010001
OVER = SRC_ALPHA | ADD << 5 | INV_SRC_ALPHA << 8 | ONE << 16 | ADD << 21
# RB_COLORCONTROL: Direct3D's, and with alpha to mask at symmetric offsets.
COLORCONTROL = 0x87000007
COVERAGE = 0xAA000017
# (samples, PA_SC_AA_CONFIG code, resolve sample select)
SAMPLES = [(2, 1, 4), (4, 2, 6)]
# (name, resolve texture format)
RESOLVES = [('8', 6), ('10', 7)]
f2w = gpugen.f2w


def case(source, clear, control=NO_BLEND, coverage=False, draws=1):
    """`clear` the destination bytes."""
    return {'state': (control, coverage, draws), 'source': list(source),
            'clear': list(clear)}


def cases():
    """{(samples, resolve name): [case]} in order."""
    out = {}
    for samples in (2, 4):
        for readback, rname in enumerate(('8', '10')):
            c = out[samples, rname] = []
            for g in range(256):
                c.append(case((0, 0, 0, 0), (g,) * 4, draws=0))
            for j in range(0, 1023 * 4 + 1, 4):
                s = min(j * QUARTER, 1.0)
                c.append(case((s, s, s, 1.0), (0,) * 4))
            for a in COVERAGE_ALPHAS[samples].values():
                for k in MSAA_SOURCES:
                    for g in MSAA_DESTINATIONS:
                        c.append(case((k / 1023.0,) * 3 + (a,), (g,) * 4,
                                      coverage=True))
        for a in FEW_ALPHAS:
            for k in FEW_SOURCES:
                for g in FEW_DESTINATIONS:
                    out[samples, '10'].append(case(
                        (k / 1023.0,) * 3 + (a / 255.0,), (g,) * 4, OVER))
    return out


def build(suite):
    shapes = rbdraw.Shapes()
    whole = rb.RECT_DRAW
    for samples, code, select in SAMPLES:
        for rname, dest_format in RESOLVES:
            resolve = gpugen.resolve_registers(
                WIDTH, HEIGHT, dest_format, 4, sample_select=select)
            state = rb.registers(GAMMA, code, mode_cntl=gpugen.MODE_CNTL)
            all_cases = cases()[samples, rname]
            for part in range(0, len(all_cases), WIDTH * HEIGHT):
                chunk = all_cases[part:part + WIDTH * HEIGHT]
                name = f'{samples}x {rname} {part // (WIDTH * HEIGHT)}'
                order = rbdraw.grouped([c['state'] for c in chunk])
                source, dest = ([0] * (4 * WIDTH * HEIGHT),
                                [0] * (4 * WIDTH * HEIGHT))
                for pixel, i in enumerate(order):
                    source[4 * pixel:4 * pixel + 4] = map(
                        f2w, chunk[i]['source'])
                    dest[4 * pixel:4 * pixel + 4] = [
                        f2w((g + 0.25) / 255) for g in chunk[i]['clear']]
                suite.add_texture(f'{name} source', source)
                suite.add_texture(f'{name} dest', dest)
                passes = [([('RB_COLOR_INFO', [0]),
                            ('RB_BLENDCONTROL0', [NO_BLEND]),
                            ('RB_COLORCONTROL', [COLORCONTROL])],
                           rb.fetch(f'{name} dest'), whole)]
                draws = [([('RB_COLOR_INFO', [GAMMA << 16]),
                           ('RB_BLENDCONTROL0', [control]),
                           ('RB_COLORCONTROL',
                            [COVERAGE if coverage else COLORCONTROL])],
                          shapes.pixels(first, count, n))
                         for (control, coverage, n), first, count in
                         rbdraw.runs([chunk[i]['state'] for i in order])
                         if n]
                passes += [(registers, rb.fetch(f'{name} source')
                            if not k else {}, draw)
                           for k, (registers, draw) in enumerate(draws[:-1])]
                registers, draw = draws[-1]
                rb.test(suite, name, f'{name} source', state, resolve,
                        draw=draw, passes=passes, after_passes=registers)
    suite.set_inputs(shapes.inputs, 2)
