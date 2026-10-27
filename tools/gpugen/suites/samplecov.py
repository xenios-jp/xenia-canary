"""Sample coverage at 1x, 2x and 4x into k_8_8_8_8 and k_2_10_10_10 render
targets, resolved per sample (select 0 to 3) and averaged (01 at 2x, 0123 at
4x) to the target's format. Over a destination of (0.2, 0.4, 0.6, 0.8),
stored as a clear packs it (its levels drawn as (k + 1/4) / steps),
each case replaces (0.75, 0.25, 0.5, a) on the samples its PA_SC_AA_MASK and
alpha to mask leave:
- masks: none, all, 0x0321, 0x8421 and 0x1111 times every sample subset, at
  alpha 1;
- alpha to mask at Direct3D's symmetric offsets (0xAA), masks all and 0x1111:
  alphas -0.25, sixteenths 0 to 1, 1.25, and each coverage threshold
  (2i + 1) / (2 samples) with its neighbouring floats;
- an edge: alpha 1, all samples, a rectangle whose left and top edges pass
  through the centers of the target's first column and row, as a full screen
  quad through a viewport of exactly the target does.
The mask's nibbles go to the pixels of each 2x2 quad, so the edge case takes
the left 32x32 pixels and every other case a 2x2 quad at even coordinates of
the right half. The source comes from a texture through the rb suite's pixel
shader, so the quads of equal state draw together.
"""

import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpugen  # noqa: E402
import rb  # noqa: E402
import rbdraw  # noqa: E402

WIDTH, HEIGHT = rb.WIDTH, rb.HEIGHT
SIDE = 32
QUADS = SIDE // 2
NO_BLEND = 0x00010001
ALL = 0xFFFF
# RB_COLORCONTROL: Direct3D's, and with alpha to mask at symmetric offsets.
COLORCONTROL = 0x87000007
COVERAGE = 0xAA000017
SOURCE = (0.75, 0.25, 0.5)
CLEAR = (0.2, 0.4, 0.6, 0.8)
# (name, RB_COLOR_INFO format, resolve format, levels of R, G, B, A)
FORMATS = [('8_8_8_8', 0, 6, (255, 255, 255, 255)),
           ('2_10_10_10', 2, 7, (1023, 1023, 1023, 3))]
# (samples, PA_SC_AA_CONFIG code, the averaging resolve's sample select)
SAMPLES = [(1, 0, None), (2, 1, 4), (4, 2, 6)]
f2w = gpugen.f2w


def f32(v):
    return struct.unpack('>f', struct.pack('>f', v))[0]


def adjacent(v, step):
    bits = struct.unpack('>I', struct.pack('>f', v))[0] + step
    return struct.unpack('>f', struct.pack('>I', bits))[0]


def unorm(value, steps):
    """A clear's packing: the clamped float times the steps, plus a half,
    truncated, in float32."""
    v = min(max(f32(value), 0.0), 1.0)
    return int(f32(f32(v * steps) + 0.5))


def cases(samples):
    """[(mask, alpha to mask, edge, alpha)] in order."""
    masks = sorted({0, 0x0321, 0x8421, ALL} |
                   {0x1111 * k for k in range(1, 1 << samples)})
    out = [(mask, False, False, 1.0) for mask in masks]
    out.append((ALL, False, True, 1.0))
    alphas = {-0.25, 1.25} | {i / 16 for i in range(17)}
    for i in range(samples):
        t = (2 * i + 1) / (2 * samples)
        alphas |= {t, adjacent(t, -1), adjacent(t, 1)}
    for a in sorted(alphas):
        for mask in (ALL, 0x1111):
            out.append((mask, True, False, a))
    return out


def layout(samples):
    """{case index: the pixel rectangle holding it}: the edge case the left
    half, the others 2x2 quads of the right half in order."""
    out = {}
    quad = 0
    for i, (mask, coverage, edge, a) in enumerate(cases(samples)):
        if edge:
            out[i] = (0, 0, SIDE, SIDE)
        else:
            x, y = SIDE + 2 * (quad % QUADS), 2 * (quad // QUADS)
            out[i] = (x, y, x + 2, y + 2)
            quad += 1
    return out


def build(suite):
    shapes = rbdraw.Shapes()
    for samples, code, average in SAMPLES:
        rects = layout(samples)
        source = [0] * (4 * WIDTH * HEIGHT)
        groups = {}
        for i, (mask, coverage, edge, a) in enumerate(cases(samples)):
            x0, y0, x1, y1 = rects[i]
            for y in range(y0, y1):
                for x in range(x0, x1):
                    p = 4 * (x + WIDTH * y)
                    source[p:p + 4] = map(f2w, SOURCE + (a,))
            if edge:
                # Edges through the first column's and row's centers.
                rect = (x0 + 0.5, y0 + 0.5, x1, y1)
            else:
                rect = rects[i]
            groups.setdefault((mask, coverage, edge), []).append(rect)
        suite.add_texture(f'{samples}x source', source)
        draws = [([('PA_SC_AA_MASK', [mask]),
                   ('RB_COLORCONTROL',
                    [COVERAGE if coverage else COLORCONTROL])],
                  shapes.rectangles(group))
                 for (mask, coverage, edge), group in groups.items()]
        for fname, rt_format, dest_format, steps in FORMATS:
            dest = f'{fname} clear'
            if dest not in suite.textures:
                suite.add_texture(dest, rb.fill(
                    [(unorm(v, s) + 0.25) / s for v, s in zip(CLEAR, steps)]))
            state = rb.registers(rt_format, code, mode_cntl=gpugen.MODE_CNTL)
            state.append(('RB_BLENDCONTROL0', [NO_BLEND]))
            selects = [(f'sample {k}', k) for k in range(samples)]
            if average is not None:
                selects.append(('average', average))
            for sname, select in selects:
                passes = [([('PA_SC_AA_MASK', [ALL]),
                            ('RB_COLORCONTROL', [COLORCONTROL])],
                           rb.fetch(dest), rb.RECT_DRAW)]
                passes += [(registers, rb.fetch(f'{samples}x source')
                            if not k else {}, draw)
                           for k, (registers, draw) in enumerate(draws[:-1])]
                registers, draw = draws[-1]
                rb.test(suite, f'{samples}x {fname} {sname}',
                        f'{samples}x source', state,
                        gpugen.resolve_registers(WIDTH, HEIGHT, dest_format, 4,
                                                 sample_select=select),
                        draw=draw, passes=passes, after_passes=registers)
    suite.set_inputs(shapes.inputs, 2)
