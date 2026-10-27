"""hiz8 made independent of what the hierarchical buffer held before each
test, in the parts hiz8 has (build()).

hiz8 assumed its clearing Resolve (a raw copy of depth, RB_COPY_CONTROL
source 4 command 0, with RB_HIZCONTROL 1) wrote the hierarchical buffer
from RB_HIZ_CLEAR. On the console it doesn't: the buffer keeps what the test
before left, or the suite before in the chain, and hiz8a and hiz8b changed
with the chain. Here every test of parts a, c and d starts, after the clear,
with draws that write every block without testing it (RB_HIZCONTROL 0x1,
depth ALWAYS over the whole target): at depth 1 for a and d, which leaves
what hiz8 assumed the clear left; for c at 0.25 and then depth 1 again with
RB_HIZCONTROL 0, which leaves every block at 0.25 over depth 1, as c
assumed RB_HIZ_CLEAR 0x200 did.

Part b asks which clears write the buffer and what they write. Each clear
test follows a seed test that writes every block at depth 0 (0x1; greater,
depth 1 with 0x11), so a block a clear doesn't write rejects every draw.
The clears: hiz8's (source 4, command 0) and hiz1-hiz7's (source 0, the
colour target, command 1), and the other two pairings, with RB_HIZ_CLEAR
values, without RB_HIZCONTROL bit 0, and greater-than into D24FS8. The last
draw tests only (0x2), a ramp of depth 0 to 1 along x, or the fine slopes of
hiz8b around RB_HIZ_CLEAR / 2048. Then hiz8b's draw-written tests, each after
the depth 1 initialisation of part a.

Part e is part b's clear question on hiz1-hiz7's 64x32 target (pitch 80,
hiz_pitch 64), with hiz8's depth base 0x100 and theirs 0x10.

Everything else is hiz8's: see its docstring.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpugen  # noqa: E402
import hiz8  # noqa: E402
import rb  # noqa: E402
import zdepth  # noqa: E402
from hiz8 import (ALWAYS, D24FS8, D24S8, GREATER, HIZ_GREATER, INCR_CLAMP,  # noqa: E402
                  KEEP, LESS, STENCIL, WRITE, Z, Z_WRITE, COLOR_DEPTH,
                  depth_word, hiz_clear_word)

# hiz8's clear and hiz1-hiz7's: (RB_COPY_CONTROL source, command).
DEPTH_RAW, COLOR_CONVERT = (4, 0), (0, 1)


class Part(hiz8.Part):
    """hiz8's Part with `init` draws first in every test, the clear's copy
    source and command, and the depth buffer's base."""

    def __init__(self, suite, width, height, init=(), depth_base=0x100):
        super().__init__(suite, width, height)
        self.init = list(init)
        self.depth_base = depth_base

    def test(self, name, draws=(), last_rect=None, last_z=0.75, last_hiz=0x3,
             samples=0, greater=False, fmt=None, hiz_pitch=None,
             clear_hiz=1, hiz_clear=None, clear_z=None, last_func=None,
             last_write=True, mode_regs=(), viz_queries=None, viz=None,
             last_regs=(), init=None, clear_copy=DEPTH_RAW):
        width, height = self.width, self.height
        init = list(self.init if init is None else init)
        draws = init + list(draws)
        # hiz8's VIZ query indices count passes from its first draw.
        shift = sum(1 for entry in init if entry[0] != 'regs')
        if viz_queries:
            viz_queries = {k + shift: v for k, v in viz_queries.items()}
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
        source, command = clear_copy
        clear = gpugen.resolve_registers(
            width, height, 6, 4, clear=(clear_word, 0), clear_enable=2,
            hiz=clear_hiz | (hiz_bit if clear_hiz else 0), command=command,
            source=source)
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
            ('RB_DEPTH_INFO', [self.depth_base | (fmt << 16)]),
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


def far(p):
    """Every block written at depth 1 without a test."""
    return [(p.whole(), 1.0, 0x1, WRITE)]


def clears(p, configs):
    """Seed then clear pairs: `configs` (name, test keyword arguments, last
    z). The seed writes every block at depth 0 (0x1), greater-than at depth 1
    (0x11), and fails its own last draw per pixel without touching the
    buffer."""
    for name, kwargs, last_z in configs:
        greater = kwargs.get('greater', False)
        p.test(f'seed for {name}', [(p.whole(), 0.0, 0x1, WRITE)],
               greater=greater, init=[], last_hiz=0, last_z=0.75)
        p.test(name, init=[], last_hiz=0x2, last_z=last_z, **kwargs)


RAMP = (0.0, 1.0, 1.0)


def clear_configs(p):
    """Which clears write the buffer, and the value they write: a ramp of
    depth 0 to 1 along x tested over each."""
    configs = []
    for copy_name, copy in (('depth raw', DEPTH_RAW),
                            ('colour convert', COLOR_CONVERT)):
        for h in (None, 0x400, 0x200, 0x1):
            value = 'its depth' if h is None else f'{h:#x}'
            configs.append((f'{copy_name} clear hiz clear {value} ramp',
                            dict(clear_copy=copy, hiz_clear=h), RAMP))
        configs.append((f'{copy_name} clear without bit 0 ramp',
                        dict(clear_copy=copy, clear_hiz=0), RAMP))
        configs.append((f'{copy_name} clear greater hiz clear 0x700 ramp',
                        dict(clear_copy=copy, greater=True,
                             hiz_clear=hiz_clear_word(0.5, True)), RAMP))
    for copy_name, copy in (('depth convert', (4, 1)),
                            ('colour raw', (0, 0))):
        configs.append((f'{copy_name} clear hiz clear 0x400 ramp',
                        dict(clear_copy=copy, hiz_clear=0x400), RAMP))
    return configs


def part_b(p):
    t = p.test
    configs = clear_configs(p)
    # hiz8b's RB_HIZ_CLEAR values, through hiz1-hiz7's clear.
    fine = 2.0 ** -16
    for h in (0x400, 0x401, 0x200, 0x7FE, 0x7FF, 0x555, 0x800, 0xFFFFFFFF):
        configs.append((f'colour convert clear hiz clear {h:#x} sloped',
                        dict(clear_copy=COLOR_CONVERT, hiz_clear=h),
                        hiz8.slope(p, (h & 0x7FF) / 2048, fine)))
    clears(p, configs)
    # hiz8b's draw-written tests over depth 1 written by 0x1.
    w = lambda rect, z, hiz=0x3: (rect, z, hiz, WRITE)
    for z in (0.5, 0.5 + 3 * 2.0 ** -13, 0.5 + 2.0 ** -24, 0.3, 0.9):
        t(f'written {z!r} sloped', [w(p.whole(), z)],
          last_z=hiz8.slope(p, z, fine))
    t('written 0.5 by 0x1 sloped', [w(p.whole(), 0.5, 0x1)],
      last_z=hiz8.slope(p, 0.5, fine))
    for z in (0.5, 0.3):
        t(f'greater written {z} sloped', [w(p.whole(), z)], greater=True,
          last_z=hiz8.slope(p, z, fine))
    ramp = (0.25 - 0.5 / 256, 0.25 + (p.width - 0.5) / 256,
            0.25 + (p.width - 0.5) / 256)
    for m in range(166, 181, 2):
        t(f'written ramp tested at 0.25 + {m}/1024', [w(p.whole(), ramp)],
          last_z=0.25 + m / 1024)
    for x in (2, 4, 6):
        t(f'0.25 then 0.75 over x < {x}', [w(p.whole(), 0.25),
                                           w((0, 0, x, p.height), 0.75)],
          last_z=0.5)
    t('0.25 then 0.75 over x < 2 by 0x2', [w(p.whole(), 0.25),
                                           w((0, 0, 2, p.height), 0.75, 0x2)],
      last_z=0.5)


def part_e(p):
    configs = []
    for copy_name, copy in (('depth raw', DEPTH_RAW),
                            ('colour convert', COLOR_CONVERT)):
        for h in (None, 0x200):
            value = 'its depth' if h is None else f'{h:#x}'
            configs.append((f'{copy_name} clear hiz clear {value} ramp',
                            dict(clear_copy=copy, hiz_clear=h), RAMP))
    clears(p, configs)


def build(suite, part):
    if part == 'e':
        p = Part(suite, 64, 32)
        part_e(p)
        p.depth_base = 0x10
        configs = []
        for copy_name, copy in (('depth raw', DEPTH_RAW),
                                ('colour convert', COLOR_CONVERT)):
            configs.append((f'depth base 0x10 {copy_name} clear hiz clear '
                            'its depth ramp', dict(clear_copy=copy), RAMP))
        clears(p, configs)
        p.finish()
        return
    p = Part(suite, 128, 64)
    if part == 'a':
        p.init = far(p)
        hiz8.part_a(p)
    elif part == 'b':
        p.init = far(p)
        part_b(p)
    elif part == 'c':
        p.init = [(p.whole(), 0.25, 0x1, WRITE), (p.whole(), 1.0, 0x0, WRITE)]
        hiz8.part_c(p)
    elif part == 'd':
        p.init = far(p)
        hiz8.part_d(p)
    p.finish()
