"""Hierarchical depth beyond hiz1-hiz5: which of the buffer's blocks a later
draw sees written, what a written block holds, how blocks map to samples
and hiz_pitch, and what a clearing Resolve does, in parts of the size the
ring takes (build()).

Every test draws a W x H target (W, H by part) with a depth buffer,
RB_SURFACE_INFO's hiz_pitch W as titles set it unless a test says
otherwise, after a clearing Resolve that clears depth to 1 and stencil 0x5A
and writes the hierarchical buffer from RB_HIZ_CLEAR (RB_HIZCONTROL 1). The
draws are rectangles at a depth from their vertices; one writing (depth
ALWAYS, writes, stencil off) or only testing (depth NEVER, no writes) the
depth buffer, under the RB_HIZCONTROL it gives. The last tests depth LESS
and writes it with stencil ALWAYS, zfail INCR_CLAMP, under RB_HIZCONTROL 3
unless a test says otherwise: a pixel the hierarchical test rejects keeps
its depth and stencil 0x5A, one failing the depth test holds 0x5B, one
passing the last draw's depth. With `greater` the depths are mirrored
(1 - z) into D24FS8 cleared to 0, tested GREATER with RB_HIZCONTROL bit 4.
The output is the raw depth and stencil words, sample 0 of each pixel.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpugen  # noqa: E402
import pm4  # noqa: E402
import rb  # noqa: E402
import zdepth  # noqa: E402

D24S8, D24FS8 = 0, 1
NEVER, LESS, GREATER, ALWAYS = 0, 1, 4, 7
KEEP, INCR_CLAMP = 0, 3
STENCIL, Z, Z_WRITE = 1, 2, 4
HIZ_GREATER = 0x10
COLOR_DEPTH = 4
WRITE = Z | Z_WRITE | (ALWAYS << 4)
READ = Z | (NEVER << 4)
DEPTH_BASE = 0x100
f2w = gpugen.f2w


class Rects:
    """Inputs of 8 dwords a vertex after the Resolve's W x H rectangle:
    rectangle lists at a depth, or with a depth at each of the three
    vertices given."""

    def __init__(self, width, height):
        self.inputs = [f2w(v) for v in (-0.5, -0.5, width - 0.5, -0.5,
                                        width - 0.5, height - 0.5)] + [0, 0]
        self.draws = {}

    def draw(self, x0, y0, x1, y1, z):
        zs = z if isinstance(z, tuple) else (z, z, z)
        key = (x0, y0, x1, y1, zs)
        if key not in self.draws:
            first = len(self.inputs) // 8
            for (x, y), vz in zip(((x0, y0), (x1, y0), (x1, y1)), zs):
                self.inputs += [f2w(v) for v in (x - 0.5, y - 0.5, vz, 1.0,
                                                 0, 0, 0, 1.0)]
            self.draws[key] = (pm4.PRIM_RECTANGLE_LIST, first, 3)
        return self.draws[key]


def hiz_clear_word(z, greater=False):
    """RB_HIZ_CLEAR as resolve_registers sets it from a depth clear: the top
    11 bits of the depth and stencil word (unorm24 or, `greater`, 20e4)."""
    return depth_word(z, greater) >> 21


def depth_word(z, greater=False):
    """The depth and stencil word of depth z, stencil 0."""
    if not greater:
        return min(int(z * 0xFFFFFF + 0.5), 0xFFFFFF) << 8
    if z <= 0:
        return 0
    import math
    m, e = math.frexp(z)  # z = m 2^e, 0.5 <= m < 1
    exponent = e - 1 + 15
    if exponent <= 0:
        return int(z * 2.0 ** 14 * (1 << 20) + 0.5) << 8
    mantissa = int((m * 2 - 1) * (1 << 20) + 0.5)
    if mantissa >> 20:
        exponent, mantissa = exponent + 1, 0
    return ((exponent << 20) | mantissa) << 8


class Part:
    """One part's suite, target and tests."""

    def __init__(self, suite, width, height, pitch=None):
        self.suite = suite
        self.width, self.height = width, height
        self.pitch = pitch or (width + 79) // 80 * 80
        self.rects = Rects(width, height)
        suite.add_texture('zero', rb.fill((0.0, 0.0, 0.0, 0.0)))
        self.vs = gpugen.interpolating_vertex_shader(8)
        self.fetch = rb.fetch('zero')
        self.resolve = gpugen.resolve_registers(width, height, 6, 4,
                                                command=0, source=4)

    def whole(self):
        return (0, 0, self.width, self.height)

    def test(self, name, draws=(), last_rect=None, last_z=0.75, last_hiz=0x3,
             samples=0, greater=False, fmt=None, hiz_pitch=None,
             clear_hiz=1, hiz_clear=None, clear_z=None, last_func=None,
             last_write=True, mode_regs=(), viz_queries=None, viz=None,
             last_regs=()):
        """`draws`: (rect, z, RB_HIZCONTROL, WRITE or READ or another
        RB_DEPTHCONTROL, extra registers before it); an entry ('regs',
        registers) adds registers before the next draw. `hiz_clear`
        replaces RB_HIZ_CLEAR, `clear_hiz` the clear's RB_HIZCONTROL."""
        width, height = self.width, self.height
        if fmt is None:
            fmt = D24FS8 if greater else D24S8
        if hiz_pitch is None:
            hiz_pitch = width
        mirror = (lambda z: tuple(1.0 - v for v in z) if isinstance(z, tuple)
                  else 1.0 - z) if greater else (lambda z: z)
        hiz_bit = HIZ_GREATER if greater else 0
        if clear_z is None:
            clear_z = 1.0
        clear_word = depth_word(mirror(clear_z), fmt == D24FS8) | 0x5A
        clear = gpugen.resolve_registers(
            width, height, 6, 4, clear=(clear_word, 0), clear_enable=2,
            hiz=clear_hiz | (hiz_bit if clear_hiz else 0), command=0,
            source=4)
        if hiz_clear is not None:
            clear['clear'][0] = hiz_clear

        def state(depthcontrol, hiz):
            return [('RB_MODECONTROL', [COLOR_DEPTH]),
                    ('RB_DEPTHCONTROL', [depthcontrol]),
                    ('RB_HIZCONTROL', [hiz])]
        passes = [([], self.fetch, self.rects.draw(0, 0, 0, 0, 0.0))]
        pending = []
        for entry in draws:
            if entry[0] == 'regs':
                pending += list(entry[1])
                continue
            rect, z, hiz, control, *extra = entry
            if hiz & 3 or hiz & 0xC:
                hiz |= hiz_bit
            if greater and control == READ:
                control = Z | (NEVER << 4)
            passes.append((pending + state(control, hiz) +
                           (extra[0] if extra else []), {},
                           self.rects.draw(*rect, mirror(z))))
            pending = []
        func = last_func if last_func is not None else (
            GREATER if greater else LESS)
        last = (Z | (Z_WRITE if last_write else 0) | (func << 4) |
                STENCIL | (ALWAYS << 8) | (KEEP << 11) | (KEEP << 14) |
                (INCR_CLAMP << 17))
        registers = [
            ('SQ_CONTEXT_MISC', [gpugen.SCREEN_XY]),
            ('RB_SURFACE_INFO', [self.pitch | (samples << 16) |
                                 (hiz_pitch << 18)]),
            ('RB_COLOR_INFO', [0]),
            ('RB_DEPTH_INFO', [DEPTH_BASE | (fmt << 16)]),
            ('RB_COLOR_MASK', [0xF]),
            ('RB_DEPTHCONTROL', [rb.DEPTHCONTROL]),
            ('RB_STENCILREFMASK', [rb.STENCILREFMASK]),
            ('PA_SC_WINDOW_SCISSOR_BR', [(height << 16) | width]),
            ('PA_SC_AA_CONFIG', [gpugen.AA_CONFIG[samples]]),
            ('PA_SU_SC_MODE_CNTL', [gpugen.MODE_CNTL |
                                    (gpugen.MSAA_ENABLE if samples else 0)])
        ] + list(mode_regs)
        last_hiz = last_hiz | (hiz_bit if last_hiz & 3 else 0)
        self.suite.test(name, self.vs, constants=rb.CONSTANTS,
                        pixel_shader=rb.pixel_shader(),
                        program_cntl=zdepth.PROGRAM_CNTL,
                        registers=registers, passes=passes,
                        after_passes=pending + state(last, last_hiz) +
                        list(last_regs),
                        fetch_constants={},
                        draw=self.rects.draw(*(last_rect or self.whole()),
                                             mirror(last_z)),
                        resolve=self.resolve_back(registers)
                        if last_regs else self.resolve, clear=clear,
                        viz_queries=viz_queries or {}, viz=viz)

    def view(self, samples, pitch, hiz_pitch, width, height):
        """Registers drawing the EDRAM through another view: `samples`
        (0, 1, 2), surface `pitch` and `hiz_pitch`, a `width` x `height`
        scissor."""
        return [('RB_SURFACE_INFO', [pitch | (samples << 16) |
                                     (hiz_pitch << 18)]),
                ('PA_SC_AA_CONFIG', [gpugen.AA_CONFIG[samples]]),
                ('PA_SU_SC_MODE_CNTL', [gpugen.MODE_CNTL |
                                        (gpugen.MSAA_ENABLE if samples
                                         else 0)]),
                ('PA_SC_WINDOW_SCISSOR_BR', [(height << 16) | width])]

    def resolve_back(self, registers):
        """The Resolve, through the test's own view again."""
        names = ('RB_SURFACE_INFO', 'PA_SC_AA_CONFIG', 'PA_SU_SC_MODE_CNTL',
                 'PA_SC_WINDOW_SCISSOR_BR')
        return gpugen.resolve_registers(
            self.width, self.height, 6, 4, command=0, source=4,
            registers=[r for r in registers if r[0] in names])

    def finish(self):
        self.suite.set_inputs(self.rects.inputs, 8)


def tiles(x0, y0, x1, y1, size=16):
    return (x0 * size, y0 * size, x1 * size, y1 * size)


def part_a(p):
    """Which written blocks the last draw sees, at 1x on 128x64 (8 x 4
    16-pixel tiles): by the tiles a draw writes, by whether the draws test
    the buffer (bit 1) and write it (bit 0), by draws and waits between, and
    by the last draw's own extent."""
    t = p.test
    w = lambda rect, hiz=0x3, z=0.5: (rect, z, hiz, WRITE)
    # A row of k tiles, a column of k, single tiles, 2x2 and 4x2 squares,
    # everything.
    for k in range(1, 9):
        t(f'row 0 x < {16 * k}', [w(tiles(0, 0, k, 1))])
    for k in range(1, 5):
        t(f'column 0 y < {16 * k}', [w(tiles(0, 0, 1, k))])
    for tx in (0, 3, 7):
        for ty in (1, 3):
            t(f'tile {tx}, {ty}', [w(tiles(tx, ty, tx + 1, ty + 1))])
    for tx, ty in ((0, 0), (2, 0), (6, 0), (0, 2), (6, 2), (3, 1)):
        t(f'2x2 tiles at {tx}, {ty}', [w(tiles(tx, ty, tx + 2, ty + 2))])
    t('4x2 tiles at 4, 0', [w(tiles(4, 0, 8, 2))])
    t('everything', [w(p.whole())])
    # The writing draw without the test, the last draw without the write.
    two = tiles(0, 0, 2, 1)
    t('row 0 x < 32 written by 0x1', [w(two, 0x1)])
    t('row 0 x < 32 tested by 0x2', [w(two)], last_hiz=0x2)
    t('row 0 x < 32 written by 0x1 tested by 0x2', [w(two, 0x1)],
      last_hiz=0x2)
    t('2x2 tiles at 6, 0 written by 0x1', [w(tiles(6, 0, 8, 2), 0x1)])
    t('everything written by 0x1', [w(p.whole(), 0x1)])
    # Read first by a draw testing only (depth NEVER): everything, row 0,
    # row 3; then written by 0x1.
    for name, rect in (('everything', p.whole()), ('row 0', tiles(0, 0, 8, 1)),
                       ('row 3', tiles(0, 3, 8, 4))):
        t(f'read {name} then row 0 x < 32 written by 0x1',
          [(rect, 0.0, 0x2, READ), w(two, 0x1)])
        t(f'read {name} then row 0 x < 32 written by 0x3',
          [(rect, 0.0, 0x2, READ), w(two)])
    # Between the write and the last draw: draws reading k tiles of row 3,
    # writing k tiles of row 3 (0x1, at 0.9), or the buffer untouched (0).
    for k in (1, 2, 4, 8):
        t(f'row 0 x < 32 then read {k} of row 3',
          [w(two), (tiles(0, 3, k, 4), 0.0, 0x2, READ)])
        t(f'row 0 x < 32 then 0x1 writes {k} of row 3',
          [w(two), (tiles(0, 3, k, 4), 0.9, 0x1, WRITE)])
    t('row 0 x < 32 then row 3 drawn with 0',
      [w(two), (tiles(0, 3, 8, 4), 0.9, 0x0, WRITE)])
    # Two draws: top then bottom, bottom then top, the same twice.
    top, bottom = tiles(0, 0, 2, 1), tiles(0, 1, 2, 2)
    t('top then bottom', [w(top), w(bottom)])
    t('bottom then top', [w(bottom), w(top)])
    t('2x2 tiles twice', [w(tiles(0, 0, 2, 2))] * 2)
    # The last draw over part of the target.
    for name, rect in (('row 0', tiles(0, 0, 8, 1)),
                       ('tile 0, 0', tiles(0, 0, 1, 1)),
                       ('tile 1, 0', tiles(1, 0, 2, 1)),
                       ('rows 1-3', tiles(0, 1, 8, 4)),
                       ('x >= 16', tiles(1, 0, 8, 4))):
        t(f'row 0 x < 32 tested over {name}', [w(two)], last_rect=rect)


def slope(p, centre, step):
    """A last draw's depths rising `step` a pixel along x, `centre` at the
    target's middle."""
    half = p.width / 2
    return (centre - (half + 0.5) * step, centre + (half - 0.5) * step,
            centre + (half - 0.5) * step)


def part_b(p):
    """What the buffer holds and how it compares, at 1x on 128x64: written
    by the clear from RB_HIZ_CLEAR values, by flat and sloped draws, by
    draws covering a block in part, and what clears with and without bit 0
    leave. Most last draws slope 2^-16 a pixel around a value (a block 2^-14,
    the target 2^-9)."""
    t = p.test
    fine = 2.0 ** -16
    # RB_HIZ_CLEAR against a depth clear to 1: the last draw (testing only)
    # sloping around h / 2048, then flat at it and a unorm24 step either
    # side.
    for h in (0x400, 0x401, 0x200, 0x7FE, 0x7FF, 0x001, 0x555):
        t(f'hiz clear {h:#x} sloped', hiz_clear=h,
          last_z=slope(p, h / 2048, fine), last_hiz=0x2)
    for d in (-1, 0, 1):
        t(f'hiz clear 0x400 flat 0.5 {d:+} unorm24', hiz_clear=0x400,
          last_z=0.5 + d / 0xFFFFFF, last_hiz=0x2)
    # Bits beyond 11: 0x800, 0xFFF, 0x1000, 0xFFFFFFFF against 0.25, 0.9.
    for h in (0x800, 0xC00, 0x1400, 0xFFFFFFFF):
        for z in (0.25, 0.9):
            t(f'hiz clear {h:#x} at {z}', hiz_clear=h, last_z=z,
              last_hiz=0x2)
    # Greater-than, D24FS8 cleared to 0: RB_HIZ_CLEAR from 20e4 words.
    for z in (0.5, 0.25, 0.75 + 2.0 ** -9, 2.0 ** -10):
        h = hiz_clear_word(z, True)
        t(f'greater hiz clear {h:#x} sloped', greater=True, hiz_clear=h,
          last_z=slope(p, 1.0 - z, fine), last_hiz=0x2)
    # Written by a draw over everything (0x3, depth ALWAYS) at z, the last
    # draw sloping around z.
    w = lambda rect, z, hiz=0x3: (rect, z, hiz, WRITE)
    for z in (0.5, 0.5 + 3 * 2.0 ** -13, 0.5 + 2.0 ** -24, 0.3, 0.9):
        t(f'written {z!r} sloped', [w(p.whole(), z)],
          last_z=slope(p, z, fine))
    t('written 0.5 by 0x1 sloped', [w(p.whole(), 0.5, 0x1)],
      last_z=slope(p, 0.5, fine))
    for z in (0.5, 0.3):
        t(f'greater written {z} sloped', [w(p.whole(), z)], greater=True,
          last_z=slope(p, z, fine))
    # Written by a draw sloping 1/256 a pixel from 0.25, tested flat.
    ramp = (0.25 - 0.5 / 256, 0.25 + (p.width - 0.5) / 256,
            0.25 + (p.width - 0.5) / 256)
    for m in range(166, 181, 2):
        t(f'written ramp tested at 0.25 + {m}/1024', [w(p.whole(), ramp)],
          last_z=0.25 + m / 1024)
    # A block written whole, then partly farther with depth ALWAYS: 0.25
    # everywhere, then 0.75 over x < 2 (part of block 0), x < 4 (all of
    # it), x < 6; the last draw at 0.5.
    for x in (2, 4, 6):
        t(f'0.25 then 0.75 over x < {x}', [w(p.whole(), 0.25),
                                           w((0, 0, x, p.height), 0.75)],
          last_z=0.5)
    t('0.25 then 0.75 over x < 2 by 0x2', [w(p.whole(), 0.25),
                                           w((0, 0, 2, p.height), 0.75, 0x2)],
      last_z=0.5)
    # Clears: a test leaving 0.25 everywhere, then clears with bit 0 or
    # not, RB_HIZ_CLEAR the depth's or another, the last draw at 0.5.
    seed = [w(p.whole(), 0.25)]
    t('seed 0.25 a', seed)
    t('clear without bit 0 after 0.25', clear_hiz=0, last_z=0.5,
      last_hiz=0x2)
    t('seed 0.25 b', seed)
    t('clear with bit 0 after 0.25', last_z=0.5, last_hiz=0x2)
    t('clear with bit 0 hiz clear 0x200', hiz_clear=0x200, last_z=0.5,
      last_hiz=0x2)
    t('clear without bit 0 after hiz clear 0x200', clear_hiz=0,
      last_z=0.5, last_hiz=0x2)
    t('clear with bit 0 hiz clear 0x7ff', hiz_clear=0x7FF, last_z=0.5,
      last_hiz=0x2)


LEQUAL_TEST = Z | (3 << 4)
DEPTH_ONLY = 5
VIZ_ENABLE, KILL_PIX_POST_HI_Z = 1, 0x80


def part_d(p):
    """Metro 2033's shadow atlas (4B4D07F5, play-metro command 436 on): its
    cascades after the first draw predicated on VIZ queries 6-11, which its
    commands 426-431 ran over the scene's depth (depth-only, LEQUAL without
    writes, RB_HIZCONTROL 3, PA_SC_VIZ_QUERY (id << 1) | 0x81). Here the
    scene is 0.25 everywhere (RB_HIZCONTROL 3), a query draw over
    everything at a depth, then draws not touching depth, then the last
    draw predicated on the query (VIZ token 0x100 | id), depth ALWAYS at
    0.75: a pixel holding 0.25 shows the draw was skipped."""
    t = p.test
    w = lambda rect, z, hiz=0x3: (rect, z, hiz, WRITE)
    filler = (tiles(7, 3, 8, 4), 0.5, 0x0, 0)

    def query(name, zq, hiz=0x3, kill=True, fillers=0, query=6,
              control=LEQUAL_TEST):
        viz = VIZ_ENABLE | (query << 1) | (KILL_PIX_POST_HI_Z if kill else 0)
        draws = [w(p.whole(), 0.25),
                 (p.whole(), zq, hiz, control,
                  [('RB_MODECONTROL', [DEPTH_ONLY]),
                   ('PA_SC_VIZ_QUERY', [viz])]),
                 ('regs', [('PA_SC_VIZ_QUERY', [0])])]
        draws += [filler] * fillers
        t(name, draws, last_func=ALWAYS, last_hiz=0,
          viz_queries={2: query}, viz=0x100 | query)

    for fillers in (0, 4, 64):
        query(f'hidden by hi-Z, {fillers} draws after', 0.5, fillers=fillers)
        query(f'visible, {fillers} draws after', 0.1, fillers=fillers)
        query(f'hidden per pixel only, {fillers} draws after', 0.5, hiz=0,
              fillers=fillers)
        query(f'hidden by hi-Z without kill, {fillers} draws after', 0.5,
              kill=False, fillers=fillers)
        query(f'hidden by hi-Z tested by 0x2, {fillers} draws after', 0.5,
              hiz=0x2, fillers=fillers)
    # As Metro: six queries, ids 6-11, each over its own tile column, the
    # third (id 8) visible; 16 draws after; the last on id 8 or id 6.
    for last in (8, 6):
        draws = [w(p.whole(), 0.25)]
        viz_queries = {}
        for k, query_id in enumerate(range(6, 12)):
            viz = VIZ_ENABLE | (query_id << 1) | KILL_PIX_POST_HI_Z
            draws += [(tiles(k, 0, k + 1, 4), 0.1 if query_id == 8 else 0.5,
                       0x3, LEQUAL_TEST,
                       [('RB_MODECONTROL', [DEPTH_ONLY]),
                        ('PA_SC_VIZ_QUERY', [viz])])]
            viz_queries[len(draws)] = query_id
        draws += [('regs', [('PA_SC_VIZ_QUERY', [0])])] + [filler] * 16
        t(f'six queries, last on id {last}', draws, last_func=ALWAYS,
          last_hiz=0, viz_queries=viz_queries, viz=0x100 | last)


def part_c(p):
    """Where a block lives: the clear writes 0.25 to every block (RB_HIZ_CLEAR
    0x200 over depth 1), a draw writing depth 1 without testing (0x1, depth
    ALWAYS) through some view (sample count, hiz_pitch) over a region, then
    the last draw testing only (0x2) at 0.5 through the 128x64 1x view
    (hiz_pitch 128) unless a test says otherwise: a pixel it writes is in a
    block the draw wrote, one it doesn't (depth 1 kept) in one it didn't.
    The 2x and 4x views cover the same EDRAM: 128x32 pixels at pitch 160 at
    2x, 64x32 at pitch 80 at 4x."""
    t = p.test
    one = p.view(0, p.pitch, p.width, p.width, p.height)
    two = p.view(1, p.pitch, p.width, p.width, p.height // 2)
    four = p.view(2, p.pitch // 2, p.width // 2, p.width // 2,
                  p.height // 2)

    def write(name, rect, view, control=0x1, last_regs=one, **kwargs):
        t(name, [(rect, 1.0, control, WRITE, view)], hiz_clear=0x200,
          last_z=0.5, last_hiz=0x2, last_regs=last_regs, **kwargs)
    write('1x 16x16 at 16, 16', (16, 16, 32, 32), one)
    write('1x 4x4 at 36, 20', (36, 20, 40, 24), one)
    write('1x 8x8 at 2, 2', (2, 2, 10, 10), one)
    write('1x everything by 0x5', p.whole(), one, control=0x5)
    # 4x through a view half as wide (Metro's command 436: 4x, pitch 640,
    # hiz_pitch 640, clearing its 1280-wide 1x atlas), and with hiz_pitch
    # 128.
    for name, rect in (('everything', (0, 0, 64, 32)),
                       ('8x8 at 8, 8', (8, 8, 16, 16)),
                       ('8x8 at 16, 0', (16, 0, 24, 8)),
                       ('2x2 at 6, 10', (6, 10, 8, 12)),
                       ('32x16 at 32, 16', (32, 16, 64, 32))):
        write(f'4x {name}', rect, four)
    write('4x everything by 0x5', (0, 0, 64, 32), four, control=0x5)
    write('4x everything hiz_pitch 128', (0, 0, 64, 32),
          p.view(2, p.pitch // 2, p.width, p.width // 2, p.height // 2))
    for name, rect in (('everything', (0, 0, 128, 32)),
                       ('8x8 at 8, 8', (8, 8, 16, 16)),
                       ('4x2 at 36, 10', (36, 10, 40, 12)),
                       ('2x4 at 36, 10', (36, 10, 38, 14))):
        write(f'2x {name}', rect, two)
    # 1x through other hiz_pitch values, over a tile and over a block.
    for pitch in (0, 4, 16, 64, 96, 100, 127, 129, 136, 160, 256):
        view = p.view(0, p.pitch, pitch, p.width, p.height)
        write(f'1x hiz_pitch {pitch} 16x16 at 32, 16', (32, 16, 48, 32), view)
    for pitch in (64, 100, 127):
        view = p.view(0, p.pitch, pitch, p.width, p.height)
        write(f'1x hiz_pitch {pitch} 4x4 at 36, 20', (36, 20, 40, 24), view)
    # Written through hiz_pitch 128, tested through 64 or 100.
    for pitch in (64, 100):
        write(f'1x 16x16 at 32, 16 tested through hiz_pitch {pitch}',
              (32, 16, 48, 32), one,
              last_regs=p.view(0, p.pitch, pitch, p.width, p.height))
    # Metro's stage 12: 1x depth written over the top half, tested through
    # the 4x view (LEQUAL at 0.5 there, as its command 11).
    t('1x top half then tested through 4x', [((0, 0, 128, 32), 1.0, 0x1,
                                             WRITE, one)],
      hiz_clear=0x200, last_z=0.5, last_hiz=0x2, last_rect=(0, 0, 64, 32),
      last_regs=four)
    t('4x top half then tested through 1x', [((0, 0, 64, 16), 1.0, 0x1,
                                             WRITE, four)],
      hiz_clear=0x200, last_z=0.5, last_hiz=0x2, last_regs=one)


def build(suite, part):
    p = Part(suite, 128, 64)
    {'a': part_a, 'b': part_b, 'c': part_c, 'd': part_d}[part](p)
    p.finish()
