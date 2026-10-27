"""Hierarchical depth and stencil: which pixels RB_HIZCONTROL's tests reject,
for every value titles were seen drawing with, set up as titles' Direct3D sets
it up, in parts of the size the ring takes: 1 and 2 the values, 3 those with
bits above 17 (moving the buffer), 4 and 5 probes of the buffer's blocks, its
hiz_pitch and bit 3, 6 and 7 hierarchical stencil (bits 2, 3, 5 and 8-15;
his()).

Every test draws the rb suite's 64x32 target with a depth buffer,
RB_SURFACE_INFO's hiz_pitch the target's width as titles set it. A clearing
Resolve clears depth and stencil 0x5A, writing the hierarchical buffer from
RB_HIZ_CLEAR (RB_HIZCONTROL 1); then a priming pass after the depth clear: the
top 8 rows in depth-only mode, depth NEVER, the buffer tested from the side
that rejects them whole. Both have the test value's bits above 4 (its base)
too. The draws are rectangles at a depth from their vertices, under
RB_HIZCONTROL the test's value unless a draw says 0. The last tests depth (and
writes it, but for 'no write') with stencil ALWAYS, zfail INCR_CLAMP: a pixel
it fails per pixel holds 0x5B, one the hierarchical test rejects keeps 0x5A and
its depth. The output is the raw depth and stencil words.

In parts 1-5 a value with bit 4 (the greater-than test) is drawn
reversed, as titles draw it: D24FS8 cleared to 0 and GREATER, the depths
below mirrored (1 - z); one without, D24S8 cleared to 1 and LESS. Depths,
as less-than:
- 'hidden': all at 0.5, then all at 0.75;
- 'shown': all at 0.5, then all at 0.25;
- 'edge': x < 20 at 0.5, then all at 0.75 (the buffer's blocks at the
  edge);
- 'always': all at 0.5, then all at 0.75 with the compare ALWAYS;
- 'stale far': all at 0.5 with RB_HIZCONTROL 0, then all at 0.75;
- 'stale near': all at 0.25, then all at 0.75 compare ALWAYS with
  RB_HIZCONTROL 0, then all at 0.5;
- 'no write': 'hidden' with the last draw not writing depth.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import depth  # noqa: E402
import gpugen  # noqa: E402
import pm4  # noqa: E402
import rb  # noqa: E402
import zdepth  # noqa: E402

WIDTH, HEIGHT = rb.WIDTH, rb.HEIGHT
D24S8, D24FS8 = 0, 1
NEVER, LESS, GREATER, ALWAYS = 0, 1, 4, 7
KEEP, INCR_CLAMP = 0, 3
STENCIL, Z, Z_WRITE = 1, 2, 4
HIZ_TEST, HIZ_GREATER = 2, 0x10
DEPTH_ONLY, COLOR_DEPTH = 5, 4
HIZ_PASS_ROWS = 8
BASE = ~0x1F & 0xFFFFFFFF  # bits 5 and up
# Every value of RB_HIZCONTROL titles' draws testing depth were seen with,
# those with bits 18 and up apart: they move the buffer.
VALUES = (0x33, 0x13, 0x3, 0x113, 0x10, 0x11, 0x4035, 0x403D, 0x4038, 0x9,
          0x613, 0x15, 0x5, 0x803C, 0x8035, 0x803D, 0x17, 0x35, 0x1, 0x39,
          0x111, 0x38, 0xFF14, 0x14, 0x725, 0x925, 0xB25)
MOVED = (0x14000012, 0x18200003, 0x18200001, 0xCA80003, 0x9600013,
         0xE100001, 0x18200000, 0xCA80014)
SCENARIOS = ('hidden', 'shown', 'edge', 'always', 'stale far', 'stale near',
             'no write')
f2w = gpugen.f2w


def probes(test):
    """Which pixels the buffer keeps, by the shape written and hiz_pitch, and
    what bit 3 does."""
    shapes = {'x < 4': (0, 0, 4, HEIGHT), 'x < 8': (0, 0, 8, HEIGHT),
              'x < 12': (0, 0, 12, HEIGHT), 'x < 16': (0, 0, 16, HEIGHT),
              'x < 32': (0, 0, 32, HEIGHT), 'x < 48': (0, 0, 48, HEIGHT),
              'x >= 20': (20, 0, WIDTH, HEIGHT), 'y < 8': (0, 0, WIDTH, 8),
              'y < 16': (0, 0, WIDTH, 16), 'y < 24': (0, 0, WIDTH, 24),
              'y >= 16': (0, 16, WIDTH, HEIGHT),
              '8x8 at 8, 8': (8, 8, 16, 16),
              '16x16 at 16, 16': (16, 16, 32, 32)}
    for shape, rect in shapes.items():
        test(f'0x3 edge {shape}', 0x3, 'edge', rect=rect)
    test('0x3 edge no pass', 0x3, 'edge', pass_rows=0)
    test('0x3 edge pass all rows', 0x3, 'edge', pass_rows=HEIGHT)
    for pitch in (0, 16, 32, 48, 80, 128, 256):
        test(f'0x3 hidden hiz_pitch {pitch}', 0x3, 'hidden', hiz_pitch=pitch)
        test(f'0x3 edge hiz_pitch {pitch}', 0x3, 'edge', hiz_pitch=pitch)
        test(f'0x3 edge y < 24 hiz_pitch {pitch}', 0x3, 'edge',
             rect=(0, 0, WIDTH, 24), hiz_pitch=pitch)
    # Bit 3: over the clear, and over all at 0.5 written with bit 0 alone.
    for value in (0x8, 0x9, 0xA, 0xB, 0x18, 0x19, 0x1A, 0x1B):
        for written in (False, True):
            for z in (0.25, 0.75):
                over = 'over 0.5' if written else 'over the clear'
                test(f'{value:#x} probe {over} at {z}', value, 'probe',
                     draws=[(0.5, 1 | (value & HIZ_GREATER))] if written
                     else [], last_z=z)


def probes2(test):
    """The buffer's granularity, and whether the rows a draw reaches decide
    what it keeps."""
    shapes = {'x < 1': (0, 0, 1, HEIGHT), 'x < 2': (0, 0, 2, HEIGHT),
              'x < 3': (0, 0, 3, HEIGHT), 'x < 5': (0, 0, 5, HEIGHT),
              'x < 6': (0, 0, 6, HEIGHT), 'x < 7': (0, 0, 7, HEIGHT),
              'x < 36': (0, 0, 36, HEIGHT), 'x < 40': (0, 0, 40, HEIGHT),
              'x < 44': (0, 0, 44, HEIGHT),
              '4 <= x < 8': (4, 0, 8, HEIGHT),
              '16 <= x < 20': (16, 0, 20, HEIGHT),
              '20 <= x < 24': (20, 0, 24, HEIGHT),
              'y < 17': (0, 0, WIDTH, 17), 'y < 18': (0, 0, WIDTH, 18),
              'y < 20': (0, 0, WIDTH, 20), 'y < 28': (0, 0, WIDTH, 28),
              'y < 31': (0, 0, WIDTH, 31), 'y >= 8': (0, 8, WIDTH, HEIGHT),
              'y >= 24': (0, 24, WIDTH, HEIGHT),
              '8 <= y < 24': (0, 8, WIDTH, 24)}
    for x in (0, 16, 32, 48):
        for y in (0, 16):
            shapes[f'16x16 at {x}, {y}'] = (x, y, x + 16, y + 16)
    for x in (0, 32):
        for y in (0, 16):
            shapes[f'32x16 at {x}, {y}'] = (x, y, x + 32, y + 16)
        shapes[f'32x32 at {x}, 0'] = (x, 0, x + 32, HEIGHT)
    shapes['64x16 at 0, 8'] = (0, 8, WIDTH, 24)
    shapes['48x32 at 16, 0'] = (16, 0, WIDTH, HEIGHT)
    for shape, rect in shapes.items():
        test(f'0x3 edge {shape}', 0x3, 'edge', rect=rect)
    # x < 20 in two draws, each half of the rows, in either order.
    top, bottom = (0, 0, 20, 16), (0, 16, 20, HEIGHT)
    test('0x3 x < 20 top then bottom', 0x3, 'probe',
         draws=[(top, 0.5, 0x3), (bottom, 0.5, 0x3)])
    test('0x3 x < 20 bottom then top', 0x3, 'probe',
         draws=[(bottom, 0.5, 0x3), (top, 0.5, 0x3)])
    test('0x3 x < 20 twice', 0x3, 'probe',
         draws=[((0, 0, 20, HEIGHT), 0.5, 0x3)] * 2)
    # The tested draw over part of the target after x < 20.
    for name, rect in (('y >= 16', (0, 16, WIDTH, HEIGHT)),
                       ('y < 16', (0, 0, WIDTH, 16)),
                       ('x < 20', (0, 0, 20, HEIGHT)),
                       ('x < 20 y >= 16', bottom)):
        test(f'0x3 edge tested {name}', 0x3, 'edge', last_rect=rect)
    # And after all at 0.5.
    for name, rect in (('y >= 16', (0, 16, WIDTH, HEIGHT)),
                       ('16x16 at 16, 16', (16, 16, 32, 32)),
                       ('4x4 at 4, 20', (4, 20, 8, 24))):
        test(f'0x3 hidden tested {name}', 0x3, 'hidden', last_rect=rect)
    # Multisampled.
    for samples in (1, 2):
        for shape in ('x < 4', 'x < 20', 'y < 8', 'y < 16', '8x8 at 8, 8',
                      '16x16 at 16, 16'):
            rect = {'x < 4': (0, 0, 4, HEIGHT), 'x < 20': (0, 0, 20, HEIGHT),
                    'y < 8': (0, 0, WIDTH, 8), 'y < 16': (0, 0, WIDTH, 16),
                    '8x8 at 8, 8': (8, 8, 16, 16),
                    '16x16 at 16, 16': (16, 16, 32, 32)}[shape]
            test(f'0x3 {1 << samples}x edge {shape}', 0x3, 'edge', rect=rect,
                 samples=samples)


HIS_WRITE, HIS_TEST = 4, 8
REPLACE = 2
EQUAL, GEQUAL = 2, 6


def his_control(func, ref):
    return (func << 5) | (ref << 8)


def his(suite, rects, vs, fetch, part):
    """Hierarchical stencil, as titles use RB_HIZCONTROL bits 2 and 3 (bits 5-7
    and 8-15 as their values put them, apparently a compare and a reference):
    written by a depth and stencil clear or a draw, tested by a last draw over
    everything that runs with depth and stencil ALWAYS, writing depth 0.5 and
    incrementing stencil: a pixel it rejects keeps its depth and stencil."""
    def test(name, stencil, clear_hiz, last_hiz, draws=(), fmt=D24S8,
             samples=0, greater=False, last=None):
        clear_depth = 0 if greater else 0xFFFFFF00
        clear = gpugen.resolve_registers(
            WIDTH, HEIGHT, 6, 4, clear=(clear_depth | stencil, 0),
            clear_enable=2, hiz=clear_hiz)
        mirror = (lambda z: 1.0 - z) if greater else (lambda z: z)

        def state(depthcontrol, hiz, ref):
            return [('RB_MODECONTROL', [COLOR_DEPTH]),
                    ('RB_DEPTHCONTROL', [depthcontrol]),
                    ('RB_HIZCONTROL', [hiz]),
                    ('RB_STENCILREFMASK', [ref])]
        passes = [([], fetch, rects.draw(0, 0, 0, 0, 0.0))]
        for rect, z, depthcontrol, hiz, ref in draws:
            passes.append((state(depthcontrol, hiz, ref), {},
                           rects.draw(*rect, mirror(z))))
        if last is None:
            last = (Z | Z_WRITE | (ALWAYS << 4) | STENCIL | (ALWAYS << 8) |
                    (INCR_CLAMP << 14), 0x00FFFF00 | stencil)
        registers = rb.registers(samples=samples, depth_format=fmt,
                                 mode_cntl=gpugen.MODE_CNTL) + [
            ('RB_SURFACE_INFO', [80 | (samples << 16) | (WIDTH << 18)])]
        suite.test(name, vs, constants=rb.CONSTANTS,
                   pixel_shader=rb.pixel_shader(),
                   program_cntl=zdepth.PROGRAM_CNTL, registers=registers,
                   passes=passes,
                   after_passes=state(last[0], last_hiz, last[1]),
                   fetch_constants={},
                   draw=rects.draw(0, 0, WIDTH, HEIGHT, mirror(0.5)),
                   resolve=depth.RESOLVE, clear=clear)

    mark = Z | (ALWAYS << 4) | STENCIL | (ALWAYS << 8) | (REPLACE << 14)
    whole = (0, 0, WIDTH, HEIGHT)
    if part == 7:
        his_draws(test, mark, whole)
        return
    # Written by the clear: its stencil against reference 0x40 by each
    # compare, tested with the same.
    for stencil in (0x40, 0x5A):
        for func in range(8):
            control = his_control(func, 0x40)
            test(f'his clear {stencil:#x} func {func} ref 0x40', stencil,
                 1 | HIS_WRITE | control, HIS_TEST | control)
    # Whether the test's own compare and reference matter.
    for func, ref in ((0, 0x5A), (1, 0x5A), (1, 0x40)):
        test(f'his clear 0x40 func 0 ref 0x40 tested func {func} ref '
             f'{ref:#x}', 0x40, 1 | HIS_WRITE | his_control(0, 0x40),
             HIS_TEST | his_control(func, ref))
    # The clear writing the depth buffer alone, or neither: the stencil
    # buffer is what the test before left.
    test('his clear 0x40 no his write func 0 ref 0x40', 0x40, 1,
         HIS_TEST | his_control(0, 0x40))
    test('his clear 0x40 no hiz write func 0 ref 0x40', 0x40,
         HIS_WRITE | his_control(0, 0x40), HIS_TEST | his_control(0, 0x40))
    # Written by a draw replacing stencil with 0x40 over a shape (depth
    # ALWAYS without writes, stencil ALWAYS), over a clear to 0 writing it.
    shapes = {'whole': (0, 0, WIDTH, HEIGHT), 'x < 32': (0, 0, 32, HEIGHT),
              'y < 16': (0, 0, WIDTH, 16), 'y >= 16': (0, 16, WIDTH, HEIGHT),
              '16x16 at 16, 16': (16, 16, 32, 32),
              'x < 6': (0, 0, 6, HEIGHT)}
    for func in (0, 1):
        control = his_control(func, 0x40)
        for shape, rect in shapes.items():
            test(f'his draw {shape} func {func}', 0,
                 1 | HIS_WRITE | control, HIS_TEST | control,
                 draws=[(rect, 0.5, mark, HIS_WRITE | control,
                         0x00FFFF00 | 0x40)])
        # The draw not writing it.
        test(f'his draw x < 32 no his write func {func}', 0,
             1 | HIS_WRITE | control, HIS_TEST | control,
             draws=[((0, 0, 32, HEIGHT), 0.5, mark, control,
                     0x00FFFF00 | 0x40)])
    # Halo 3's (4D5307E6): D24FS8 at 4x, depth GEQUAL. A depth draw near
    # over x < 32, then its marking pass at 0.75 (behind it there): stencil
    # ALWAYS, zpass ZERO, zfail REPLACE 0x40, write mask 0x7F, under 0x4035;
    # then a draw under 0x4038, no depth test, stencil EQUAL 0x40, zpass and
    # fail INCR_CLAMP (a pixel run passing 0x41, failing 1, rejected 0 or
    # 0x40), or 0x403d with depth GEQUAL and zfail INCR_CLAMP.
    ZERO = 1
    near = Z | Z_WRITE | (ALWAYS << 4)
    halo_mark = (Z | (GEQUAL << 4) | STENCIL | (ALWAYS << 8) | (ZERO << 14) |
                 (REPLACE << 17))
    ref = 0x007FFF00 | 0x40
    for samples in (0, 2):
        for clear_hiz in (0x11, 0x15 | his_control(1, 0x40)):
            for name, value, last in (
                    ('0x4038', 0x4038, STENCIL | (EQUAL << 8) |
                     (INCR_CLAMP << 11) | (INCR_CLAMP << 14)),
                    ('0x403d', 0x403D, Z | (GEQUAL << 4) | STENCIL |
                     (EQUAL << 8) | (INCR_CLAMP << 11) | (INCR_CLAMP << 14) |
                     (INCR_CLAMP << 17))):
                test(f'his halo {1 << samples}x clear {clear_hiz:#x} {name}',
                     0, clear_hiz, value, fmt=D24FS8, samples=samples,
                     greater=True,
                     draws=[((0, 0, 32, HEIGHT), 0.5, near, 0x11, ref),
                            ((0, 0, WIDTH, HEIGHT), 0.75, halo_mark, 0x4035,
                             ref)],
                     last=(last, ref))


def his_draws(test, mark, whole):
    """Hierarchical stencil written by a draw over everything replacing
    stencil, as his()."""
    def written(name, value, write, tested, depthcontrol=mark, rect=whole,
                **kwargs):
        test(name, 0, 1 | HIS_WRITE | his_control(0, 0), HIS_TEST | tested,
             draws=[(rect, 0.5, depthcontrol, HIS_WRITE | write,
                     0x00FFFF00 | value)], **kwargs)
    # Each compare, the draw's stencil below, at and above the reference.
    for func in range(8):
        for value in (0x3F, 0x40, 0x41):
            control = his_control(func, 0x40)
            written(f'his func {func} stencil {value:#x} ref 0x40', value,
                    control, control)
    # Written with one compare or reference, tested with another.
    for (wf, wr), (tf, tr) in (((0, 0x40), (1, 0x40)), ((1, 0x40), (0, 0x40)),
                               ((0, 0x40), (0, 0x41)), ((0, 0x41), (0, 0x40)),
                               ((1, 0x41), (1, 0x40)), ((1, 0x40), (1, 0x41))):
        written(f'his stencil 0x40 written func {wf} ref {wr:#x} tested '
                f'func {tf} ref {tr:#x}', 0x40, his_control(wf, wr),
                his_control(tf, tr))
    # The stencil a draw leaves rather than its reference: masked off, its
    # stencil test failing, or replacing on zfail.
    control = his_control(0, 0x40)
    test('his stencil 0 kept by write mask 0, ref 0x40 func 0', 0,
         1 | HIS_WRITE, HIS_TEST | control,
         draws=[(whole, 0.5, mark, HIS_WRITE | control, 0x0000FF00 | 0x40)])
    control = his_control(0, 0)
    test('his stencil 0 kept by stencil NEVER, ref 0 func 0', 0,
         1 | HIS_WRITE, HIS_TEST | control,
         draws=[(whole, 0.5, Z | (ALWAYS << 4) | STENCIL | (NEVER << 8),
                 HIS_WRITE | control, 0x00FFFF00)])
    test('his stencil 0 untouched by stencil off, ref 0 func 0', 0,
         1 | HIS_WRITE, HIS_TEST | control,
         draws=[(whole, 0.5, Z | (ALWAYS << 4), HIS_WRITE | control,
                 0x00FFFF00)])
    control = his_control(0, 0x40)
    written('his stencil 0x40 by zfail, ref 0x40 func 0', 0x40, control,
            control, depthcontrol=Z | (NEVER << 4) | STENCIL | (ALWAYS << 8) |
            (REPLACE << 17))
    # At 4x, over x < 6.
    written('his 4x stencil 0x40 x < 6 func 0', 0x40, control, control,
            rect=(0, 0, 6, HEIGHT), samples=2)
    # A clear not writing it after a test that left everything culled: the
    # test before culls everything, then this one's draw doesn't write it.
    written('his func 0 stencil 0x40 again', 0x40, control, control)
    test('his after a clear without bit 2, func 0', 0, 1, HIS_TEST | control)
    written('his func 0 stencil 0x40 once more', 0x40, control, control)
    test('his after a clear writing hi-Z and hi-S, func 0', 0,
         1 | HIS_WRITE, HIS_TEST | control)


class Rects:
    """Inputs of 8 dwords a vertex after the Resolve's rectangle, rectangle
    lists at a depth."""

    def __init__(self):
        self.inputs = rb.RECT + [0, 0]
        self.draws = {}

    def draw(self, x0, y0, x1, y1, z):
        key = (x0, y0, x1, y1, z)
        if key not in self.draws:
            first = len(self.inputs) // 8
            for x, y in ((x0, y0), (x1, y0), (x1, y1)):
                self.inputs += [f2w(v) for v in (x - 0.5, y - 0.5, z, 1.0,
                                                 0, 0, 0, 1.0)]
            self.draws[key] = (pm4.PRIM_RECTANGLE_LIST, first, 3)
        return self.draws[key]


def build(suite, part, values=None):
    """Part 1 and 2 the values without base bits, 2 also D24S8 and 4x, 3
    those with, 4 and 5 probes, 6 and 7 hierarchical stencil. ``values``
    overrides the control values for parts 1-3 without changing scenarios.
    """
    rects = Rects()
    suite.add_texture('zero', rb.fill((0.0, 0.0, 0.0, 0.0)))
    vs = gpugen.interpolating_vertex_shader(8)
    fetch = rb.fetch('zero')

    def test(name, value, scenario, fmt=None, samples=0, hiz_pitch=WIDTH,
             rect=(0, 0, 20, HEIGHT), draws=(), last_z=0.75,
             pass_rows=HIZ_PASS_ROWS, last_rect=(0, 0, WIDTH, HEIGHT)):
        greater = bool(value & HIZ_GREATER)
        base = value & BASE
        if fmt is None:
            fmt = D24FS8 if greater else D24S8
        func = GREATER if greater else LESS
        mirror = (lambda z: 1.0 - z) if greater else (lambda z: z)
        whole = lambda z: rects.draw(0, 0, WIDTH, HEIGHT, mirror(z))
        clear = gpugen.resolve_registers(
            WIDTH, HEIGHT, 6, 4, clear=(0x5A if greater else 0xFFFFFF5A, 0),
            clear_enable=2, hiz=base | 1)

        def state(depthcontrol, hiz=value):
            return [('RB_MODECONTROL', [COLOR_DEPTH]),
                    ('RB_DEPTHCONTROL', [depthcontrol]),
                    ('RB_HIZCONTROL', [hiz])]
        write = Z | Z_WRITE | (ALWAYS << 4)
        # The priming pass: at z 0 testing greater-than over a clear to 1, at
        # 1 testing less-than over a clear to 0.
        passes = [([('RB_MODECONTROL', [DEPTH_ONLY]),
                    ('RB_DEPTHCONTROL', [Z | (NEVER << 4)]),
                    ('RB_HIZCONTROL',
                     [base | HIZ_TEST | (0 if greater else HIZ_GREATER)])],
                   fetch, rects.draw(0, 0, WIDTH, pass_rows,
                                     1.0 if greater else 0.0))]
        if not pass_rows:
            passes = [([], fetch, rects.draw(0, 0, 0, 0, 0.0))]
        last_func, last_write = func, True
        if scenario == 'hidden':
            passes.append((state(write), {}, whole(0.5)))
        elif scenario == 'shown':
            passes.append((state(write), {}, whole(0.5)))
            last_z = 0.25
        elif scenario == 'edge':
            passes.append((state(write), {},
                           rects.draw(*rect, mirror(0.5))))
        elif scenario == 'always':
            passes.append((state(write), {}, whole(0.5)))
            last_func = ALWAYS
        elif scenario == 'stale far':
            passes.append((state(write, 0), {}, whole(0.5)))
        elif scenario == 'stale near':
            passes += [(state(write), {}, whole(0.25)),
                       (state(write, 0), {}, whole(0.75))]
            last_z = 0.5
        elif scenario == 'no write':
            passes.append((state(write), {}, whole(0.5)))
            last_write = False
        elif scenario == 'probe':
            for draw in draws:
                z, hiz = draw[-2:]
                area = draw[0] if len(draw) == 3 else (0, 0, WIDTH, HEIGHT)
                passes.append((state(write, hiz), {},
                               rects.draw(*area, mirror(z))))
        last = (Z | (Z_WRITE if last_write else 0) | (last_func << 4) |
                STENCIL | (ALWAYS << 8) | (KEEP << 11) | (KEEP << 14) |
                (INCR_CLAMP << 17))
        registers = rb.registers(samples=samples, depth_format=fmt,
                                 mode_cntl=gpugen.MODE_CNTL) + [
            ('RB_SURFACE_INFO', [80 | (samples << 16) | (hiz_pitch << 18)])]
        suite.test(name, vs, constants=rb.CONSTANTS,
                   pixel_shader=rb.pixel_shader(),
                   program_cntl=zdepth.PROGRAM_CNTL, registers=registers,
                   passes=passes, after_passes=state(last),
                   fetch_constants={},
                   draw=rects.draw(*last_rect, mirror(last_z)),
                   resolve=depth.RESOLVE, clear=clear)

    if part >= 6:
        his(suite, rects, vs, fetch, part)
        suite.set_inputs(rects.inputs, 8)
        return
    if part >= 4:
        (probes if part == 4 else probes2)(test)
        suite.set_inputs(rects.inputs, 8)
        return
    if values is None:
        values = {1: VALUES[:14], 2: VALUES[14:], 3: MOVED}[part]
    for value in values:
        for scenario in SCENARIOS:
            test(f'{value:#x} {scenario}', value, scenario)
    if part == 2:
        # The greater-than values titles also draw into D24S8, and at 4x.
        for value in (0x13, 0x33):
            for scenario in ('hidden', 'edge', 'stale near'):
                test(f'{value:#x} D24S8 {scenario}', value, scenario,
                     fmt=D24S8)
        for value in (0x3, 0x33):
            for scenario in ('hidden', 'edge', 'stale near'):
                test(f'{value:#x} 4x {scenario}', value, scenario, samples=2)
    suite.set_inputs(rects.inputs, 8)
