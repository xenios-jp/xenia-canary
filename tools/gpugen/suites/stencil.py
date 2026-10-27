"""Depth compare and stencil: draws into the depth suite's D24S8 / D24FS8
buffer, each pixel shader's oDepth the X of a 64x32 texture at its pixel
(SQ_PROGRAM_CNTL.ps_export_mode 3, as a pixel shader writing depth needs),
then the depth suite's raw depth Resolve. A test first sets every pixel to
depth 0.5 and stencil 0, then draws its cases into bands of the target, each
band a rectangle list with its own RB_DEPTHCONTROL and RB_STENCILREFMASK(_BF).

- 'compare F func': per pixel, a draw of depth x (ALWAYS, written, stencil
  REPLACE 0x10), then one of depth y with the compare function, writes, and
  stencil zfail INCR_CLAMP, zpass REPLACE 0x5A. The x and y come from a
  seeded generator, kept as it was: random x of every D24 and 20e4 range and 0,
  0.25, 0.5, 1, each with y = x, the floats next to it, and x +- a half and a
  whole depth unit; then more of the same.
- 'stencil op F slot': each of the 8 operations in the stencil fail slot
  (function NEVER), the depth fail slot (depth function NEVER) or the pass
  slot, over stencil 0, 1, 0x7F, 0x80, 0xFE, 0xFF, 0x5A with reference 0,
  0x5A, 0xFF and write masks 0xFF, 0x0F, 0xF0; depth not written. Row op * 3
  + reference, column write mask * 7 + stencil.
- 'stencil compare': D24S8, each function with references 0, 0x5A, 0xFF, 0x0F
  and read masks 0xFF, 0x0F, 0xF0, 0 over stencil 0, 0x5A, 0xFF, 0x0F, 0x5B,
  0x59, pass INVERT. Row function * 4 + reference, column read mask * 6 +
  stencil.
- 'two-sided F': over stencil 0x40, front stencil ALWAYS REPLACE 0x11 and back
  ALWAYS REPLACE or NEVER (fail INCR_WRAP) 0x22, two-sided or not, drawn as
  front (counterclockwise) or back (clockwise) faces; band 4 rows each.
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

NEVER, ALWAYS = 0, 7
KEEP, ZERO, REPLACE, INCR_CLAMP, DECR_CLAMP, INVERT, INCR_WRAP, DECR_WRAP = (
    range(8))
STENCIL_ENABLE, Z_ENABLE, Z_WRITE_ENABLE, BACKFACE_ENABLE = 1, 2, 4, 1 << 7
UNORM24 = (1 << 24) - 1
PROGRAM_CNTL = (rb.PROGRAM_CNTL & ~(0xF << 27)) | (3 << 27)
WIDTH, HEIGHT = rb.WIDTH, rb.HEIGHT
f2w = gpugen.f2w


def f32(v):
    return struct.unpack('>f', struct.pack('>f', v))[0]


def bits(v):
    return struct.unpack('>I', struct.pack('>f', v))[0]


def step(v, n):
    """The float n representable steps from non-negative v."""
    b = bits(v)
    if b & 0x80000000:
        b = 0
    return struct.unpack('>f', struct.pack('>I', max(0, b + n)))[0]


def float20e4(e, m):
    if e == 0:
        return m / float(1 << 20) * 2.0 ** -14
    return (1 + m / float(1 << 20)) * 2.0 ** (e - 15)


def unorm_values(rng, n):
    codes = [0, 1, 2, 3, 0x7FFFFF, 0x800000, 0x800001, UNORM24 - 1, UNORM24]
    codes += [rng.randrange(UNORM24 + 1) for _ in range(n)]
    values = []
    for code in codes:
        v = code / UNORM24
        values += [f32(v), step(f32(v), 1), step(f32(v), -1)]
        if code < UNORM24:
            t = (code + 0.5) / UNORM24
            values += [f32(t), step(f32(t), 1), step(f32(t), -1)]
    return values


def float24_values(rng, n):
    values = []
    for e in range(16):
        for m in [0, 1, (1 << 20) - 1] + [rng.randrange(1 << 20)
                                          for _ in range(n)]:
            v = float20e4(e, m)
            values += [f32(v), step(f32(v), 1), step(f32(v), -1)]
            if m < (1 << 20) - 1 or e < 15:
                after = (float20e4(e, m + 1) if m < (1 << 20) - 1 else
                         float20e4(e + 1, 0))
                t = (v + after) / 2
                values += [f32(t), step(f32(t), 1), step(f32(t), -1)]
    return values


def in_unit(values):
    seen, out = set(), []
    for v in values:
        if 0.0 <= v <= 1.0 and bits(v) not in seen:
            seen.add(bits(v))
            out.append(v)
    return out


def chosen():
    """The seeded choices: the depth values, each format's compare
    values x and each (samples, format)'s MSAA values."""
    rng = random.Random(0xD24F58)
    for _ in range(2):
        for _ in range(18):
            rng.getrandbits(32)  # clears, which no test uses
    values = in_unit(unorm_values(rng, 60) + float24_values(rng, 4))
    values += in_unit([rng.random() for _ in range(150)] +
                      [2.0 ** -rng.uniform(0, 24) for _ in range(150)])
    xs = {fmt: [rng.choice(values) for _ in range(24)] + [0.0, 1.0, 0.5, 0.25]
          for fmt in (depth.D24S8, depth.D24FS8)}
    msaa = {(samples, fmt): rng.sample(values, 40) for samples in (2, 4)
            for fmt in (depth.D24S8, depth.D24FS8)}
    return values, xs, msaa


def compare_pairs():
    """{(format, function): [(x, y)]}, the chosen pairs first."""
    values, xs, _ = chosen()
    more = random.Random(0x5C0)
    pairs = {}
    for fmt in (depth.D24S8, depth.D24FS8):
        if fmt == depth.D24S8:
            def ulp(x):
                return 1.0 / UNORM24
        else:
            def ulp(x):
                return max(2.0 ** (math.floor(math.log2(max(x, 2.0 ** -15)))
                                   - 20), 2.0 ** -34)

        def neighbours(x):
            u = ulp(x)
            return [y for y in (x, step(x, 1), step(x, -1), f32(x + u / 2),
                                f32(x - u / 2), f32(x + u), f32(x - u))
                    if 0.0 <= y <= 1.0]

        for func in range(8):
            pairs[fmt, func] = [(x, y) for x in xs[fmt]
                                for y in neighbours(x)]
        for func in range(8):
            while len(pairs[fmt, func]) < WIDTH * HEIGHT:
                x = more.choice(values)
                pairs[fmt, func] += [(x, y) for y in neighbours(x)]
            del pairs[fmt, func][WIDTH * HEIGHT:]
    return pairs


def control(zfunc=ALWAYS, z_write=True, stencil=None, back=None):
    """RB_DEPTHCONTROL: the depth test with `zfunc`, and `stencil` / `back` as
    (function, fail, depth fail, pass)."""
    value = Z_ENABLE | (Z_WRITE_ENABLE if z_write else 0) | zfunc << 4
    if stencil:
        func, fail, zfail, zpass = stencil
        value |= (STENCIL_ENABLE | func << 8 | fail << 11 | zpass << 14 |
                  zfail << 17)
    if back:
        func, fail, zfail, zpass = back
        value |= (BACKFACE_ENABLE | func << 20 | fail << 23 | zpass << 26 |
                  zfail << 29)
    return value


def masks(ref, read=0xFF, write=0xFF):
    return ref | read << 8 | write << 16


def state(depthcontrol, front, back=None):
    """The registers of a draw; without two-sided stencil the back face's
    masks are the front's, as Direct3D sets them."""
    return [('RB_DEPTHCONTROL', [depthcontrol]),
            ('RB_STENCILREFMASK_BF', [front if back is None else back,
                                      front])]


def build(suite):
    inputs = list(rb.RECT)
    shapes = {}

    def shape(key, points, primitive):
        """A draw of `points` (pixel corners), added to the inputs once."""
        if key not in shapes:
            first = len(inputs) // 2
            for px, py in points:
                inputs.extend((f2w(px - 0.5), f2w(py - 0.5)))
            shapes[key] = (primitive, first, len(points))
        return shapes[key]

    def band(x0, y0, x1, y1):
        return shape((x0, y0, x1, y1), [(x0, y0), (x1, y0), (x1, y1)],
                     pm4.PRIM_RECTANGLE_LIST)

    def quad(y0, y1, clockwise):
        x0, x1 = 0, WIDTH
        corners = [(x0, y0), (x1, y0), (x0, y1), (x1, y0), (x1, y1),
                   (x0, y1)]
        if not clockwise:
            corners = corners[2::-1] + corners[:2:-1]
        return shape((y0, y1, clockwise), corners, pm4.PRIM_TRIANGLE_LIST)

    whole = band(0, 0, WIDTH, HEIGHT)
    suite.add_texture('half', rb.fill((0.5, 0.0, 0.0, 0.0)))
    # Depth 0.5 and stencil 0 everywhere.
    reset = (state(control(stencil=(ALWAYS, KEEP, KEEP, REPLACE)),
                   masks(0)), 'half', whole)

    def test(name, fmt, draws, mode_cntl=None):
        """`draws`: [(registers, texture, draw)] after the reset, the last
        the test's own."""
        passes, fetched = [], None
        for registers, texture, draw in [reset] + draws[:-1]:
            passes.append((registers, rb.fetch(texture)
                           if texture != fetched else {}, draw))
            fetched = texture
        registers, texture, draw = draws[-1]
        rb.test(suite, name, texture,
                rb.registers(depth_format=fmt, mode_cntl=mode_cntl),
                depth.RESOLVE, draw=draw, passes=passes,
                after_passes=registers, pixel=rb.pixel_shader(depth=True),
                program_cntl=PROGRAM_CNTL)

    formats = (('D24S8', depth.D24S8), ('D24FS8', depth.D24FS8))
    pairs = compare_pairs()
    for fname, fmt in formats:
        for func in range(8):
            first, second = f'{fname} {func} x', f'{fname} {func} y'
            suite.add_texture(first, [w for x, _ in pairs[fmt, func]
                                      for w in (f2w(x), 0, 0, 0)])
            suite.add_texture(second, [w for _, y in pairs[fmt, func]
                                       for w in (f2w(y), 0, 0, 0)])
            test(f'compare {fname} {func}', fmt, [
                (state(control(stencil=(ALWAYS, KEEP, KEEP, REPLACE)),
                       masks(0x10)), first, whole),
                (state(control(func, stencil=(ALWAYS, KEEP, INCR_CLAMP,
                                              REPLACE)),
                       masks(0x5A)), second, whole)])

    inits = (0, 1, 0x7F, 0x80, 0xFE, 0xFF, 0x5A)
    refs = (0x00, 0x5A, 0xFF)
    writes = (0xFF, 0x0F, 0xF0)
    slots = {'fail': lambda op: (NEVER, op, KEEP, KEEP),
             'zfail': lambda op: (ALWAYS, KEEP, op, KEEP),
             'pass': lambda op: (ALWAYS, KEEP, KEEP, op)}
    for fname, fmt in formats:
        for slot, make in slots.items():
            draws = []
            for w in range(len(writes)):
                for i, init in enumerate(inits):
                    x = w * len(inits) + i
                    draws.append((state(control(
                        stencil=(ALWAYS, KEEP, KEEP, REPLACE)), masks(init)),
                        'half', band(x, 0, x + 1, 8 * len(refs))))
            for op in range(8):
                for r, ref in enumerate(refs):
                    y = op * len(refs) + r
                    for w, write in enumerate(writes):
                        zfunc = NEVER if slot == 'zfail' else ALWAYS
                        draws.append((state(
                            control(zfunc, False, make(op)),
                            masks(ref, 0xFF, write)), 'half',
                            band(w * len(inits), y, (w + 1) * len(inits),
                                 y + 1)))
            test(f'stencil op {fname} {slot}', fmt, draws)

    refs = (0x00, 0x5A, 0xFF, 0x0F)
    values = (0x00, 0x5A, 0xFF, 0x0F, 0x5B, 0x59)
    reads = (0xFF, 0x0F, 0xF0, 0x00)
    draws = []
    for r in range(len(reads)):
        for v, value in enumerate(values):
            x = r * len(values) + v
            draws.append((state(control(stencil=(ALWAYS, KEEP, KEEP,
                                                 REPLACE)), masks(value)),
                          'half', band(x, 0, x + 1, 8 * len(refs))))
    for func in range(8):
        for f, ref in enumerate(refs):
            y = func * len(refs) + f
            for r, read in enumerate(reads):
                draws.append((state(control(stencil=(func, KEEP, KEEP,
                                                     INVERT)),
                                    masks(ref, read, 0xFF)), 'half',
                              band(r * len(values), y,
                                   (r + 1) * len(values), y + 1)))
    test('stencil compare', depth.D24S8, draws)

    front = (ALWAYS, KEEP, KEEP, REPLACE)
    backs = ((ALWAYS, KEEP, KEEP, REPLACE), (NEVER, INCR_WRAP, KEEP, REPLACE))
    for fname, fmt in formats:
        draws = [(state(control(stencil=front), masks(0x40)), 'half', whole)]
        y = 0
        for back_face in (False, True):
            for two_sided in (False, True):
                for back in backs:
                    draws.append((state(
                        control(stencil=front,
                                back=back if two_sided else None),
                        masks(0x11), masks(0x22) if two_sided else None),
                        'half', quad(y, y + 4, back_face)))
                    y += 4
        test(f'two-sided {fname}', fmt, draws, mode_cntl=gpugen.MODE_CNTL)

    suite.set_inputs(inputs, 2)
