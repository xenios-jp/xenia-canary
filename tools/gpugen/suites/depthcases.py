"""Depth and stencil cases for the depth case suites: each a (group, case) in
order, a case a dict
of the depth format, samples, the raw depth and stencil word the target is
cleared to, whether the whole 32x32 target is read back (else one uniform
word) and up to two draws. A draw is a dict of the path (0 the vertices' Z,
1 the pixel shader's depth), back face, depth test, write and function,
stencil (front and back), the Z at the corners of the case's quad (top
left, top right, bottom left, bottom right), the exported depth, polygon
offset and slope, and the geometry: bits 0-1 the layout, 2-3 how far each
triangle's vertices rotate, 8-15 and 16-23 the viewport's shift in signed
pixels. words() packs a case as words.

- main(): clear, convert, special, slope, compare, stencil-op,
  stencil-compare, two-sided, bias, bias-slope, msaa.
- second(): offsets from one depth step to past the range.
- third(): planes whose slopes are exact binary fractions, offsets wrapping
  past +-8 and finer than a depth unit.
- fourth(): steep planes drawn as other triangles.

build() draws them, each test first clearing depth and stencil to its cases'
word with a clearing Resolve, then resolving depth raw like the depth suite:
- A case read back as one word gets a pixel of a test of its format, clear
  word and path, each of its draws two triangles over that pixel in the
  case's vertex order at its Z, in window coordinates; the draws of the
  pixels go in order of the cases' first, then second draws, a draw for each
  state. With path 1 the pixel shader exports a texture's X, the draw's
  exported depth at its pixel.
- A case read back whole gets a test of its own, drawn through a 34x34
  viewport: its vertices in clip space, PA_CL_VTE_CNTL scaling and
  offsetting them to (-1 + shift, -1 + shift) .. (33 + shift, 33 + shift),
  PA_CL_CLIP_CNTL clipping in Direct3D's clip space. Its 32x32 pixels are the
  top left of the 64x32 target, which the scissor covers.
A draw's state: RB_DEPTHCONTROL with the stencil fields whether or not the
stencil test is on, the back face's masks the front's unless two-sided, and
with a polygon offset or slope both faces' offsets enabled, the slope's
scale in 1/16 pixels.
"""

import math
import os
import random
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import depth  # noqa: E402
import gpugen  # noqa: E402
import pm4  # noqa: E402
import rb  # noqa: E402
import rbdraw  # noqa: E402
import zdepth  # noqa: E402
from stencil import (bits, f32, float24_values, in_unit,  # noqa: E402
                     step, unorm_values)

NEVER, ALWAYS = 0, 7
KEEP, ZERO, REPLACE, INCR_CLAMP, DECR_CLAMP, INVERT, INCR_WRAP, DECR_WRAP = (
    range(8))
D24S8, D24FS8 = 0, 1
UNORM24 = (1 << 24) - 1
NO_STENCIL = (False, ALWAYS, KEEP, KEEP, KEEP, 0xFFFF00)
NO_BACK = (ALWAYS, KEEP, KEEP, KEEP, 0xFFFF00)


def from_bits(b):
    return struct.unpack('>f', struct.pack('>I', b & 0xFFFFFFFF))[0]


def draw(path=0, z=0.5, exported=0.5, depth_test=True, depth_write=True,
         compare=ALWAYS, stencil=None, back_face=0, two_sided=0, back=None,
         bias=0.0, slope=0.0, corners=None, geometry=0):
    """`stencil`: (enable, function, fail, depth fail, pass, reference | read
    mask << 8 | write mask << 16); `back` the same less the enable."""
    return {'path': path, 'back_face': back_face, 'depth_test': depth_test,
            'depth_write': depth_write, 'compare': compare,
            'stencil': stencil or NO_STENCIL, 'two_sided': two_sided,
            'back': back or NO_BACK, 'geometry': geometry,
            'z': [f32(v) for v in (corners if corners is not None
                                   else [z] * 4)],
            'exported': f32(exported), 'bias': f32(bias),
            'slope': f32(slope)}


def case(fmt, clear, draws, samples=1, full=0):
    return {'format': fmt, 'samples': samples, 'clear': clear,
            'full': full, 'draws': draws}


def words(c):
    """The case as 58 words."""
    out = [c['format'], c['samples'], c['clear'], len(c['draws']), c['full'],
           0]
    for d in c['draws']:
        s, b = d['stencil'], d['back']
        out += [d['path'], d['back_face'], int(d['depth_test']),
                int(d['depth_write']), d['compare'], int(s[0]), s[1], s[2],
                s[3], s[4], s[5], d['two_sided'], b[0], b[1], b[2], b[3],
                b[4], d['geometry']]
        out += [bits(v) for v in d['z'] + [d['exported'], d['bias'],
                                           d['slope']]] + [0]
    return out + [0] * (58 - len(out))


def corners(plane):
    a, bx, by = plane
    return [a - bx + by, a + bx + by, a - bx - by, a + bx - by]


PLANES = [(0.5, 0.25, 0.0), (0.5, 0.0, 0.25), (0.5, 0.4, 0.4),
          (0.25, 0.1, -0.1), (0.75, -0.2, 0.05), (0.01, 0.005, 0.004),
          (0.999, 0.0004, -0.0003), (0.5, 0.49, 0.0),
          (2.0 ** -12, 2.0 ** -13, 0), (0.3, 0.123456, 0.0654321),
          (0.6, -0.31, 0.27), (0.05, 0.03, 0.01)]


def main():
    rng = random.Random(0xD24F58)
    out = []
    for fmt in (D24S8, D24FS8):
        clears = [0, 0xFFFFFFFF, 0x12345678, 0x80000000, 0x7FFFFF00,
                  0x000001FF, 0x3F800000, 0xFFFFFF00]
        clears += [rng.getrandbits(32) for _ in range(18)]
        for c in clears:
            out.append((f'clear-{fmt}', case(fmt, c, [])))

    values = in_unit(unorm_values(rng, 60) + float24_values(rng, 4))
    values += in_unit([rng.random() for _ in range(150)] +
                      [2.0 ** -rng.uniform(0, 24) for _ in range(150)])
    for fmt in (D24S8, D24FS8):
        for path in (0, 1):
            for v in values:
                out.append((f'convert-{fmt}-{path}', case(
                    fmt, 0x55555500,
                    [draw(path, z=v if path == 0 else 0.5, exported=v)])))

    specials = [-0.0, -1e-30, -0.5, -1.0, 1.0, step(1.0, 1), 1.25, 1.5,
                1.9999999, 2.0, 3.9, 1e10, math.inf, -math.inf, 1e-40, 1e-45,
                from_bits(0x7FC00000), from_bits(0xFFFFFFFF),
                from_bits(0x7F800001)]
    for fmt in (D24S8, D24FS8):
        for clear in (0x55555500, 0xAAAAAA00):
            for v in specials:
                out.append((f'special-{fmt}',
                            case(fmt, clear, [draw(1, exported=v)])))

    for fmt in (D24S8, D24FS8):
        for p in PLANES:
            out.append((f'slope-{fmt}', case(
                fmt, 0x55555500, [draw(0, corners=corners(p))], full=1)))

    probe = (True, ALWAYS, KEEP, INCR_CLAMP, REPLACE, 0xFFFF5A)
    for fmt in (D24S8, D24FS8):
        if fmt == D24S8:
            def ulp(x):
                return 1.0 / UNORM24
        else:
            def ulp(x):
                return max(2.0 ** (math.floor(math.log2(max(x, 2.0 ** -15)))
                                   - 20), 2.0 ** -34)
        xs = [rng.choice(values) for _ in range(24)] + [0.0, 1.0, 0.5, 0.25]
        for path in (0, 1):
            for x in xs:
                u = ulp(x)
                ys = [x, step(x, 1), step(x, -1), f32(x + u / 2),
                      f32(x - u / 2), f32(x + u), f32(x - u)]
                for func in range(8):
                    for y in ys:
                        if not 0.0 <= y <= 1.0:
                            continue
                        first = draw(path, z=x if path == 0 else 0.5,
                                     exported=x)
                        second = draw(path, z=y if path == 0 else 0.5,
                                      exported=y, compare=func,
                                      stencil=probe)
                        out.append((f'compare-{fmt}-{path}-{func}',
                                    case(fmt, 0x00000010, [first, second])))

    slots = {'fail': lambda op: (True, NEVER, op, KEEP, KEEP),
             'zfail': lambda op: (True, ALWAYS, KEEP, op, KEEP),
             'pass': lambda op: (True, ALWAYS, KEEP, KEEP, op)}
    for fmt in (D24S8, D24FS8):
        for slot, make in slots.items():
            for op in range(8):
                for init in (0, 1, 0x7F, 0x80, 0xFE, 0xFF, 0x5A):
                    for ref in (0x00, 0x5A, 0xFF):
                        for write in ((0xFF, 0x0F, 0xF0) if fmt == D24S8
                                      else (0xFF,)):
                            s = make(op) + (ref | 0xFF << 8 | write << 16,)
                            d = draw(0, z=0.5, depth_write=False,
                                     compare=NEVER if slot == 'zfail'
                                     else ALWAYS, stencil=s)
                            out.append((f'stencil-op-{fmt}-{slot}',
                                        case(fmt, 0x80000000 | init, [d])))

    for func in range(8):
        for ref in (0x00, 0x5A, 0xFF, 0x0F):
            for value in (0x00, 0x5A, 0xFF, 0x0F, 0x5B, 0x59):
                for read in (0xFF, 0x0F, 0xF0, 0x00):
                    s = (True, func, KEEP, KEEP, INVERT,
                         ref | read << 8 | 0xFF << 16)
                    out.append((f'stencil-compare-{func}', case(
                        D24S8, 0x80000000 | value, [draw(0, stencil=s)])))

    front = (True, ALWAYS, KEEP, KEEP, REPLACE, 0xFFFF11)
    back = (ALWAYS, KEEP, KEEP, REPLACE, 0xFFFF22)
    back_never = (NEVER, INCR_WRAP, KEEP, REPLACE, 0xFFFF22)
    for fmt in (D24S8, D24FS8):
        for back_face in (0, 1):
            for two_sided in (0, 1):
                for b in (back, back_never):
                    out.append((f'two-sided-{fmt}', case(
                        fmt, 0x80000040,
                        [draw(0, stencil=front, back_face=back_face,
                              two_sided=two_sided, back=b)])))

    biases = [1.0, -1.0, 2.0, 16.0, 256.0, 0.5, 1.0 / UNORM24,
              -1.0 / UNORM24, 1e-3, -1e-3, 1e-6, 2.0 ** -20, 2.0 ** -24]
    for fmt in (D24S8, D24FS8):
        for z in (0.1, 0.25, 0.5, 0.75, 0.9, 0.001, 2.0 ** -10):
            for bias in biases:
                out.append((f'bias-{fmt}', case(
                    fmt, 0x55555500, [draw(0, z=z, bias=bias)])))
        for p in PLANES[:2]:
            for slope in (1.0, -1.0, 0.5, 2.0):
                out.append((f'bias-slope-{fmt}', case(
                    fmt, 0x55555500,
                    [draw(0, corners=corners(p), slope=slope)], full=1)))

    for samples in (2, 4):
        for fmt in (D24S8, D24FS8):
            for v in rng.sample(values, 40):
                for path in (0, 1):
                    out.append((f'msaa-{samples}-{fmt}-{path}', case(
                        fmt, 0x55555500,
                        [draw(path, z=v if path == 0 else 0.5, exported=v)],
                        samples=samples)))
    return out


def second():
    out = []
    steps = [k * 2.0 ** -24 for k in (1, 2, 3, 4, 5, 7, 8, 16, 255, 256)]
    steps += [2.0 ** -e for e in range(1, 24)] + [3.0, 4.0, 7.99, 8.0, 15.99,
                                                  16.0, 31.0, 1e5]
    biases = steps + [-x for x in steps]
    for fmt in (D24S8, D24FS8):
        for z in (0.5, 0.25, 0.125, 0.75, f32(0.3), f32(0.6), 2.0 ** -12):
            for bias in biases:
                out.append((f'bias2-{fmt}', case(
                    fmt, 0x55555500, [draw(0, z=z, bias=bias)])))
        for bias in biases:
            out.append((f'bias2-exported-{fmt}', case(
                fmt, 0x55555500, [draw(1, exported=0.5, bias=bias)])))
    return out


def third():
    out = []

    def ramp(fmt, e, nx, ny, j=0):
        # Corners within [2^-e, 2^(1-e)), spaced 2^-(e+23): a pixel's slope
        # is (corner difference) / 34 = n of those spacings.
        unit = 2.0 ** -(e + 23)
        a = 1.5 * 2.0 ** -e + j * unit
        bx, by = 17 * nx * unit, 17 * ny * unit
        zs = [a - bx + by, a + bx + by, a - bx - by, a + bx - by]
        assert all(f32(z) == z and 2.0 ** -e <= z < 2.0 ** (1 - e)
                   for z in zs)
        return case(fmt, 0x55555500, [draw(0, corners=zs)], full=1)

    for fmt in (D24S8, D24FS8):
        for e in ((1, 2, 3, 5) if fmt == D24S8 else (1, 3)):
            for p in (6, 9, 12, 15, 17):
                for r in (1, 3, 7):
                    out.append((f'ramp-x-{fmt}-{e}',
                                ramp(fmt, e, 2 ** p + r, 0)))
        if fmt == D24S8:
            for p in (6, 9, 12, 15, 17):
                for r in (1, 3, 7):
                    out.append((f'ramp-negx-{fmt}-1',
                                ramp(fmt, 1, -(2 ** p + r), 0)))
            for e in (1, 3):
                for p in (9, 12, 15, 17):
                    for r in (1, 3):
                        out.append((f'ramp-y-{fmt}-{e}',
                                    ramp(fmt, e, 0, 2 ** p + r)))
            for e in (1, 2):
                for p in (6, 9, 12):
                    out.append((f'ramp-xy-{fmt}-{e}',
                                ramp(fmt, e, 2 ** 15 + 3, 2 ** p + 3)))
                    out.append((f'ramp-yx-{fmt}-{e}',
                                ramp(fmt, e, 2 ** p + 3, -(2 ** 15 + 3))))
            for e in (1, 2):
                for j in (0, 1, 2, 3):
                    out.append((f'ramp-anchor-{fmt}-{e}',
                                ramp(fmt, e, 2 ** 12, 0, j)))

    zs = [f32(0.1), f32(0.3), f32(0.7), f32(1 / 3), f32(0.9),
          0.5 + 2.0 ** -24, 0.25 + 2.0 ** -25,
          0.125 + 2.0 ** -26 + 2.0 ** -27, 0.5, f32(0.6)]
    offsets = [-16.0, -32.0, -64.0, -1024.0, -1e5, 16.0, 32.0, 1024.0,
               -16.0 + 2.0 ** -20, -16.0 + 2.0 ** -24, -16.0 - 2.0 ** -24]
    for fmt in (D24S8, D24FS8):
        for z in zs:
            for off in offsets:
                out.append((f'wrap-{fmt}', case(
                    fmt, 0x55555500, [draw(0, z=z, bias=off)])))
    for z in (f32(0.1), f32(0.3), f32(0.7), f32(1 / 3)):
        for n in range(25, 33):
            for k in (1, 3):
                for sign in (1, -1):
                    out.append(('tiny-0', case(D24S8, 0x55555500, [
                        draw(0, z=z, bias=f32(sign * k * 2.0 ** -n))])))
        for off in (1.0 / UNORM24, -1.0 / UNORM24):
            out.append(('tiny-0', case(D24S8, 0x55555500,
                                       [draw(0, z=z, bias=f32(off))])))
    return out


def fourth():
    out = []
    unit = 2.0 ** -24

    def plane(a, nx, ny, x, y):
        z = a + (nx * (x - 16) + ny * (y - 16)) * unit
        assert f32(z) == z and 0 <= z <= 1, (a, nx, ny, x, y, z)
        return z

    def add(group, a, nx, ny, layout=0, turn=0, back=0, dx=0, dy=0):
        if layout == 2:
            xy = [(-1, -1), (67, -1), (-1, 67)]
        else:
            xy = [(-1, -1), (33, -1), (-1, 33), (33, 33)]
        zs = [plane(a, nx, ny, x + dx, y + dy) for x, y in xy]
        zs += [0.0] * (4 - len(xy))
        geometry = layout | turn << 2 | (dx & 0xFF) << 8 | (dy & 0xFF) << 16
        out.append((f'{group}:{nx}:{ny}:{layout}:{turn}:{back}:{dx}:{dy}:'
                    f'{a!r}',
                    case(D24S8, 0x55555500,
                         [draw(0, corners=zs, back_face=back,
                               geometry=geometry)], full=1)))

    for n in (2 ** 17, 2 ** 17 + 1, 2 ** 17 + 2, 2 ** 17 + 3, 2 ** 16 + 1,
              2 ** 15 + 1):
        a = 0.75
        for layout, turn, back in ((0, 0, 0), (0, 0, 1), (0, 1, 0),
                                   (0, 2, 0), (1, 0, 0), (1, 0, 1),
                                   (1, 1, 0)):
            add('split', a, n, 0, layout, turn, back)
        for dx, dy in ((1, 0), (2, 0), (4, 0), (0, 1), (0, 4)):
            add('shift', a, n, 0, dx=dx, dy=dy)
        for j in (1, 2, 4):
            add('slide', a + j * n * unit, n, 0)
        a = 0.5
        add('single', a, n, 0)
        for turn, back in ((0, 0), (1, 0), (2, 0), (0, 1)):
            add('single', a, n, 0, 2, turn, back)
    n = 2 ** 17 + 1
    for a in (0.75, 0.5):
        for layout in (0, 1, 2):
            if a == 0.5 or layout != 2:
                add('y', a, 0, n, layout)
    for layout in (0, 1, 2):
        add('negx', 0.5, -n, 0, layout)
    return out


FORMATS = ('D24S8', 'D24FS8')
PATHS = {None: 'none', 0: 'Z', 1: 'oDepth'}
PIXELS = rb.WIDTH * rb.HEIGHT
# Two triangles split either way, or one twice the size; the corners' clip
# space XY.
LAYOUT_CORNERS = ((0, 1, 2, 2, 1, 3), (0, 1, 3, 0, 3, 2), (0, 1, 2))
CORNER_XY = (((-1, 1), (1, 1), (-1, -1), (1, -1)),
             ((-1, 1), (3, 1), (-1, -3), (0, 0)))
POLY_OFFSET_ENABLE = 0x1800
CLIP_DX, VTE_VIEWPORT = 1 << 19, 0x43F
f2w = gpugen.f2w


def tests(cases):
    """The tests `cases` [(index, group, case)] go to, in order: dicts of
    the name, format, clear word, path (None without draws) and [(index,
    case)], one case if read back whole."""
    out, open_tests, counts = [], {}, {}
    for i, group, c in cases:
        if c['samples'] != 1:
            raise ValueError(f'case {i}: multisampled')
        paths = {d['path'] for d in c['draws']}
        if len(paths) > 1:
            raise ValueError(f'case {i}: draws of both paths')
        path = paths.pop() if paths else None
        if c['full']:
            if path != 0:
                raise ValueError(f'case {i}: read back whole, not path 0')
            out.append({'name': f'{FORMATS[c["format"]]} {group} {i}',
                        'format': c['format'], 'clear': c['clear'],
                        'path': 0, 'cases': [(i, c)]})
            continue
        if any(d['geometry'] or len(set(d['z'])) > 1 for d in c['draws']):
            raise ValueError(f'case {i}: read back as one word, not flat')
        key = (c['format'], c['clear'], path)
        test = open_tests.get(key)
        if test is None or len(test['cases']) == PIXELS:
            n = counts[key] = counts.get(key, -1) + 1
            test = open_tests[key] = {
                'name': (f'{FORMATS[key[0]]} {PATHS[path]} {key[1]:08X}'
                         f' {n}'),
                'format': key[0], 'clear': key[1], 'path': path,
                'cases': []}
            out.append(test)
        test['cases'].append((i, c))
    return out


def control(d):
    enable, func, fail, zfail, zpass, _ = d['stencil']
    value = ((1 if enable else 0) | (2 if d['depth_test'] else 0) |
             (4 if d['depth_write'] else 0) | d['compare'] << 4 |
             func << 8 | fail << 11 | zpass << 14 | zfail << 17)
    if d['two_sided']:
        func, fail, zfail, zpass, _ = d['back']
        value |= 1 << 7 | func << 20 | fail << 23 | zpass << 26 | zfail << 29
    return value


def state(d):
    """A draw's registers, as a key."""
    front = d['stencil'][5]
    scale, offset = bits(f32(d['slope'] * 16)), bits(d['bias'])
    biased = d['bias'] != 0 or d['slope'] != 0
    return (('RB_DEPTHCONTROL', (control(d),)),
            ('RB_STENCILREFMASK_BF', (d['back'][4] if d['two_sided']
                                      else front, front)),
            ('PA_SU_SC_MODE_CNTL', (gpugen.MODE_CNTL |
                                    (POLY_OFFSET_ENABLE if biased else 0),)),
            ('PA_SU_POLY_OFFSET_FRONT_SCALE', (scale, offset, scale,
                                               offset)))


def vertex_order(d):
    """The draw's corner indices in vertex order, and its corners' clip XY."""
    shape, turn = d['geometry'] & 3, d['geometry'] >> 2 & 3
    order = []
    for v in range(len(LAYOUT_CORNERS[shape])):
        slot = (v % 3 + turn) % 3
        if d['back_face'] and slot:
            slot = 3 - slot
        order.append(LAYOUT_CORNERS[shape][v - v % 3 + slot])
    return order, CORNER_XY[shape == 2]


def vertex(x, y, z):
    return [f2w(v) for v in (x, y, z, 1.0, 0, 0, 0, 1.0)]


def signed8(v):
    return (v & 0xFF) - ((v & 0x80) << 1)


def build(suite, cases, samples=1):
    """With `samples` 2 or 4, read every sample's raw depth/stencil words.

    Keep the original sample-0 tests first and reuse their geometry for the
    additional samples, including depth-fail and stencil cases.
    """
    extra_samples = []
    words = rb.RECT + [0, 0]

    def triangles(vertices):
        first = len(words) // 8
        for v in vertices:
            words.extend(v)
        return (pm4.PRIM_TRIANGLE_LIST, first, len(vertices))

    suite.add_texture('zero', rb.fill((0.0, 0.0, 0.0, 0.0)))
    vs = gpugen.interpolating_vertex_shader(8)
    odepth_cntl = (zdepth.PROGRAM_CNTL & ~(0xF << 27)) | (3 << 27)
    for test in tests(cases):
        fmt, path = test['format'], test['path']
        registers = rb.registers(samples={1: 0, 2: 1, 4: 2}[samples],
                                 depth_format=fmt,
                                 mode_cntl=gpugen.MODE_CNTL)
        draws = []
        if test['cases'][0][1]['full']:
            _, c = test['cases'][0]
            registers += [('PA_CL_CLIP_CNTL', [CLIP_DX]),
                          ('PA_CL_VTE_CNTL', [VTE_VIEWPORT])]
            for d in c['draws']:
                order, xy = vertex_order(d)
                x0 = -1 + signed8(d['geometry'] >> 8)
                y0 = -1 + signed8(d['geometry'] >> 16)
                viewport = [f2w(v) for v in (17.0, x0 + 17.0, -17.0,
                                             y0 + 17.0, 1.0, 0.0)]
                draws.append((state(d) + (('PA_CL_VPORT_XSCALE', viewport),),
                              'zero', triangles(
                                  [vertex(*xy[k], d['z'][k]) for k in order])))
        elif path is None:
            # Nothing to draw: one draw over the target, depth and stencil
            # off.
            draws.append(((('RB_DEPTHCONTROL', (0,)),), 'zero', triangles(
                [vertex(x - 0.5, y - 0.5, 0.5) for x, y in (
                    (0, 0), (rb.WIDTH, 0), (0, rb.HEIGHT), (0, rb.HEIGHT),
                    (rb.WIDTH, 0), (rb.WIDTH, rb.HEIGHT))])))
        else:
            for slot in range(max(len(c['draws']) for _, c in test['cases'])):
                texture = 'zero'
                if path == 1:
                    texture = f'{test["name"]} {slot}'
                    exported = [0.0] * PIXELS
                    for p, (_, c) in enumerate(test['cases']):
                        if slot < len(c['draws']):
                            exported[p] = c['draws'][slot]['exported']
                    suite.add_texture(texture, [w for v in exported
                                                for w in (f2w(v), 0, 0, 0)])
                groups = {}
                for p, (_, c) in enumerate(test['cases']):
                    if slot < len(c['draws']):
                        d = c['draws'][slot]
                        groups.setdefault(state(d), []).append((p, d))
                for key, pixels in groups.items():
                    vertices = []
                    for p, d in pixels:
                        x, y = p % rb.WIDTH, p // rb.WIDTH
                        order, xy = vertex_order(d)
                        vertices += [vertex(x + xy[k][0] / 2,
                                            y - xy[k][1] / 2, d['z'][k])
                                     for k in order]
                    draws.append((key, texture, triangles(vertices)))
        passes, fetched = [], None
        for key, texture, draw in draws[:-1]:
            passes.append((list(key), rb.fetch(texture)
                           if texture != fetched else {}, draw))
            fetched = texture
        key, texture, draw = draws[-1]
        name = test['name'] + (f' {samples}x' if samples > 1 else '')
        options = dict(draw=draw, passes=passes, after_passes=list(key),
                       shader=vs, pixel=rb.pixel_shader(depth=path == 1),
                       program_cntl=odepth_cntl if path == 1 else
                       zdepth.PROGRAM_CNTL,
                       clear=gpugen.resolve_registers(
                           rb.WIDTH, rb.HEIGHT, 6, 4,
                           clear=(test['clear'], 0)))
        rbdraw.test(suite, name, texture, registers, depth.RESOLVE, **options)
        for sample in range(1, samples):
            resolve = dict(depth.RESOLVE,
                           control=depth.RESOLVE['control'] | (sample << 4))
            extra_samples.append((f'{name} sample {sample}', texture,
                                  registers, resolve, options))
    for name, texture, registers, resolve, options in extra_samples:
        rbdraw.test(suite, name, texture, registers, resolve, **options)
    suite.set_inputs(words, 8)
