"""Resolves of k_8_8_8_8 and k_2_10_10_10 render targets at 2x and 4x MSAA,
each sample (select 0 to 3) and the average (01 at 2x, 0123 at 4x): 1024
cases, 256 per format and sample count, each writing its source over its
destination on the samples its PA_SC_AA_MASK leaves, without blending or
alpha to mask. unormresolve.zlib holds the cases (24 big-endian words a case:
format, blend, write mask, draws, samples, PA_SC_AA_MASK, flags, then the
float source, clear and blend constant).

Each case is a 2x2 quad, 256 per test in the left 32x32 pixels. The
destination is stored as a clear packs it (its levels drawn as (k + 1/4) /
steps), and the source comes from a texture through the rb suite's pixel
shader, so the quads of equal mask draw together.
"""

import hashlib
import os
import struct
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpugen  # noqa: E402
import rb  # noqa: E402
import rbdraw  # noqa: E402
import samplecov  # noqa: E402

WIDTH, HEIGHT = rb.WIDTH, rb.HEIGHT
QUADS = 16
NO_BLEND = 0x00010001
ALL = 0xFFFF
COLORCONTROL = samplecov.COLORCONTROL
INPUTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      'unormresolve.zlib')
INPUTS_SHA256 = \
    '1553b7a5d1c23e55a3bff3066f70c71672cf37d920d709519e1d3f9485075bea'
# (name, RB_COLOR_INFO format, resolve format, levels of R, G, B, A)
FORMATS = samplecov.FORMATS
# (samples, PA_SC_AA_CONFIG code, the averaging resolve's sample select)
SAMPLES = [(2, 1, 4), (4, 2, 6)]
f2w = gpugen.f2w


def w2f(word):
    return struct.unpack('>f', struct.pack('>I', word))[0]


def cases():
    """[(RB_COLOR_INFO format, samples, PA_SC_AA_MASK, source words, clear
    words)] in order."""
    data = zlib.decompress(open(INPUTS, 'rb').read())
    assert hashlib.sha256(data).hexdigest() == INPUTS_SHA256
    out = []
    for i in range(len(data) // 96):
        c = struct.unpack_from('>24I', data, 96 * i)
        # Every case writes its source, once, with all channels and no
        # alpha to mask, through the expanded viewport, blend constant 0.
        assert c[1:9] == (0, 1, 0, 0, 1, 0, 15, 1) and c[11] == 0, c
        assert c[20:24] == (0, 0, 0, 0), c
        out.append((c[0], c[9], c[10], c[12:16], c[16:20]))
    return out


def build(suite):
    shapes = rbdraw.Shapes()
    everything = cases()
    for fname, rt_format, dest_format, steps in FORMATS:
        for samples, code, average in SAMPLES:
            block = [c for c in everything
                     if c[0] == rt_format and c[1] == samples]
            source = [0] * (4 * WIDTH * HEIGHT)
            dest = [0] * (4 * WIDTH * HEIGHT)
            groups = {}
            for q, (_, _, mask, source_words, clear_words) in \
                    enumerate(block):
                x0, y0 = 2 * (q % QUADS), 2 * (q // QUADS)
                clear = [f2w((samplecov.unorm(w2f(w), s) + 0.25) / s)
                         for w, s in zip(clear_words, steps)]
                for y in range(y0, y0 + 2):
                    for x in range(x0, x0 + 2):
                        p = 4 * (x + WIDTH * y)
                        source[p:p + 4] = source_words
                        dest[p:p + 4] = clear
                groups.setdefault(mask, []).append((x0, y0, x0 + 2, y0 + 2))
            name = f'{samples}x {fname}'
            suite.add_texture(f'{name} source', source)
            suite.add_texture(f'{name} clear', dest)
            draws = [([('PA_SC_AA_MASK', [mask])], shapes.rectangles(rects))
                     for mask, rects in groups.items()]
            state = rb.registers(rt_format, code, mode_cntl=gpugen.MODE_CNTL)
            state += [('RB_BLENDCONTROL0', [NO_BLEND]),
                      ('RB_COLORCONTROL', [COLORCONTROL])]
            selects = [(f'sample {k}', k) for k in range(samples)]
            selects.append(('average', average))
            for sname, select in selects:
                passes = [([('PA_SC_AA_MASK', [ALL])],
                           rb.fetch(f'{name} clear'), rb.RECT_DRAW)]
                passes += [(registers, rb.fetch(f'{name} source')
                            if not k else {}, draw)
                           for k, (registers, draw) in enumerate(draws[:-1])]
                registers, draw = draws[-1]
                rb.test(suite, f'{name} {sname}', f'{name} source', state,
                        gpugen.resolve_registers(WIDTH, HEIGHT, dest_format, 4,
                                                 sample_select=select),
                        draw=draw, passes=passes, after_passes=registers)
    suite.set_inputs(shapes.inputs, 2)
