"""Resolve geometry: which render target pixels a Resolve copies, and where
in the destination they land, for the rectangles, destination sizes and
offsets, window offsets, scissors, endianness and swaps and clears titles
resolve with.

Every test draws the 128x64 k_8_8_8_8 render target (surface pitch 160, two
EDRAM tiles) whole, pixel
(x, y) = bytes (x, y, 0x96, 0xC3) in R, G, B, A order (the rb suite's pixel
shader fetching texture 'coords' at its pixel position), and then resolves it
to a k_8_8_8_8 tiled texture at its output, so every destination texel names
the pixel it holds. The output starts out the 0xCDCDCDCD sentinel, which
marks the texels a Resolve leaves alone.

The rectangles are the vertices Direct3D's Resolve passes, (L, T), (R, T),
(R, B) less the half-pixel offset. They and the scissors stay on the 8-pixel
grid, the vertices on the half-pixel one: the console's GPU wedged in
Resolves of pixels (3, 5) to (37, 22), and of (8, 8) to (40, 24) with the
vertices 0.49 pixels up and to the left. Registers a test changes for its
Resolve alone are written before it and put back after (resolve_registers
`registers` and `after`), so the draw itself never sees them. The tests of
what titles' Resolves never do go last.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpugen  # noqa: E402
import pm4  # noqa: E402
import rb  # noqa: E402

WIDTH, HEIGHT = 128, 64
SURFACE_PITCH = 160
F32_32_32_32, K_8888, K_8888_FORMAT = 38, 6, 0
D24S8 = 0
MODE_CNTL = 0x00018006  # Direct3D's default PA_SU_SC_MODE_CNTL
SCISSOR_BR = (HEIGHT << 16) | WIDTH
SCREEN_SCISSOR_BR = 0x20002000  # Direct3D's default
# D24S8 words (depth << 8 | stencil) and k_8_8_8_8 colors the clears write.
DEPTH_CLEARS = (0xABCDEF5A, 0x1234565A, 0x0F1E2D3C, 0xFEDCBA98, 0x40404040,
                0x5A5A5A5A)
COLOR_CLEARS = (0xFF336699, 0xFF996633, 0x80112233, 0x80332211, 0x7F00FF00,
                0x7FFF00FF)
f2w = gpugen.f2w


def window_offset(x, y):
    """PA_SC_WINDOW_OFFSET: 15-bit two's complement x and y."""
    return (x & 0x7FFF) | ((y & 0x7FFF) << 16)


class Rects:
    """The suite's inputs: rectangles as 3 vertices of 2 floats each, the
    whole render target first."""

    def __init__(self):
        self.inputs = []
        self.rect(0, 0, WIDTH, HEIGHT)

    def rect(self, x0, y0, x1, y1):
        """The input dword of a rectangle over pixels x0 .. x1 - 1, y0 ..
        y1 - 1."""
        dword = len(self.inputs)
        for corner in ((x0, y0), (x1, y0), (x1, y1)):
            self.inputs += [f2w(v - 0.5) for v in corner]
        return dword


def coords_texture():
    words = []
    for y in range(HEIGHT):
        for x in range(WIDTH):
            words += [f2w(x / 255), f2w(y / 255), f2w(0x96 / 255),
                      f2w(0xC3 / 255)]
    return words


def registers(depth=False, pitch=SURFACE_PITCH):
    state = [('SQ_CONTEXT_MISC', [gpugen.SCREEN_XY]),
             ('RB_SURFACE_INFO', [pitch]),
             ('RB_COLOR_INFO', [K_8888_FORMAT << 16]),
             ('RB_COLOR_MASK', [0xF]),
             ('PA_SC_WINDOW_SCISSOR_BR', [SCISSOR_BR]),
             ('PA_SC_AA_CONFIG', [gpugen.AA_CONFIG[0]])]
    if depth:
        state.insert(3, ('RB_DEPTH_INFO', [rb.DEPTH_BASE | (D24S8 << 16)]))
    return state


def build(suite):
    rects = Rects()
    suite.add_texture('coords', coords_texture())
    fetch = {rb.SLOT: (gpugen.texture_constant(F32_32_32_32, WIDTH, HEIGHT),
                       'coords')}
    whole = (pm4.PRIM_RECTANGLE_LIST, 0, 3)

    def test(name, rect=(0, 0, WIDTH, HEIGHT), dest=(WIDTH, HEIGHT),
             draw=whole, depth=False, clear=None, room=0,
             pitch=SURFACE_PITCH, **resolve):
        resolve.setdefault('registers', ())
        resolve.setdefault('after', ())
        r = gpugen.resolve_registers(
            WIDTH, HEIGHT, K_8888, 4, dest=dest,
            rect=rects.rect(*rect), **resolve)
        # Room for texels a Resolve writes past the destination it's given.
        r['bytes'] += room
        suite.test(name, gpugen.resolve_vertex_shader(),
                   constants=rb.CONSTANTS, pixel_shader=rb.pixel_shader(),
                   program_cntl=rb.PROGRAM_CNTL,
                   registers=registers(depth, pitch),
                   fetch_constants=fetch, draw=draw, resolve=r, clear=clear)

    def around(registers, defaults):
        return dict(registers=registers, after=defaults)

    # Rectangles: at the origin, offset, and at every alignment titles use.
    test('origin whole dest')
    test('origin 64x32', (0, 0, 64, 32))
    test('offset 8 8 64x32', (8, 8, 72, 40))
    test('offset 32 16 64x32', (32, 16, 96, 48))
    test('offset 64 0 aligned 64', (64, 0, 128, 64))
    test('offset 16 16 aligned 16', (16, 16, 48, 48))
    test('offset 24 8 aligned 8', (24, 8, 56, 24))

    # Destinations other than the rectangle's size.
    test('dest 256x64', dest=(256, 64))
    test('dest 64x32 rect 64x32', (0, 0, 64, 32), dest=(64, 32))
    test('dest pitch 96 rect 96x32', (0, 0, 96, 32), dest=(96, 32))
    test('dest 128x40 rect 0 24 to 128 40', (0, 24, 128, 40), dest=(128, 40))
    test('dest 128x16 rect whole', dest=(128, 16), room=0x6000)

    # Window offsets during the Resolve, as titles' predicated tiling.
    default = [('PA_SC_WINDOW_OFFSET', [0])]
    for x, y, rect in ((0, -32, (0, 32, 128, 64)), (0, -24, (0, 24, 128, 40)),
                       (-64, 0, (64, 0, 128, 64)), (16, 8, (0, 0, 64, 32)),
                       (-16, -8, (16, 8, 80, 40))):
        test(f'window offset {x} {y} rect {rect[0]} {rect[1]} to {rect[2]} '
             f'{rect[3]}', rect,
             **around([('PA_SC_WINDOW_OFFSET', [window_offset(x, y)])],
                      default))
    test('window offset 0 -24 dest 128x40 rect 0 24 to 128 40',
         (0, 24, 128, 40), dest=(128, 40),
         **around([('PA_SC_WINDOW_OFFSET', [window_offset(0, -24)])],
                  default))
    test('window offset 0 -32 disabled rect 0 32 to 128 64',
         (0, 32, 128, 64),
         **around([('PA_SU_SC_MODE_CNTL', [MODE_CNTL & ~gpugen.MODE_CNTL]),
                   ('PA_SC_WINDOW_OFFSET', [window_offset(0, -32)])],
                  [('PA_SU_SC_MODE_CNTL', [MODE_CNTL])] + default))

    # Scissors during the Resolve.
    test('window scissor BR 40 24',
         **around([('PA_SC_WINDOW_SCISSOR_BR', [(24 << 16) | 40])],
                  [('PA_SC_WINDOW_SCISSOR_BR', [SCISSOR_BR])]))
    test('window scissor TL 8 8',
         **around([('PA_SC_WINDOW_SCISSOR_TL', [(8 << 16) | 8])],
                  [('PA_SC_WINDOW_SCISSOR_TL', [0])]))
    test('screen scissor BR 40 24',
         **around([('PA_SC_SCREEN_SCISSOR_BR', [(24 << 16) | 40])],
                  [('PA_SC_SCREEN_SCISSOR_BR', [SCREEN_SCISSOR_BR])]))

    # Destination endianness and red/blue swap.
    for endian in range(4):
        for swap in (False, True):
            test(f'endian {endian}{" swap" if swap else ""}', (0, 0, 64, 32),
                 dest=(64, 32), endian=endian, swap=swap)

    # Clearing Resolves: the clear, then a draw of part of the render target,
    # then the test's Resolve of all of it. Each clears to its own values, so
    # what one leaves alone shows the previous test's: both first, then color
    # alone, then depth alone.
    part = (pm4.PRIM_RECTANGLE_LIST, rects.rect(16, 8, 48, 24) // 2, 3)
    for k, (name, enable) in enumerate((('color depth', 3), ('color', 1),
                                        ('depth', 2))):
        for depth_resolve in (False, True):
            values = (DEPTH_CLEARS[2 * k + depth_resolve],
                      COLOR_CLEARS[2 * k + depth_resolve])
            clear = gpugen.resolve_registers(WIDTH, HEIGHT, K_8888, 4,
                                             clear=values,
                                             clear_enable=enable)
            if depth_resolve:
                test(f'clear {name} depth resolve', draw=part, depth=True,
                     clear=clear, command=0, source=4)
            else:
                test(f'clear {name}', draw=part, depth=True, clear=clear)

    # Destinations titles don't resolve to: unaligned pitch and addresses.
    test('dest pitch 100 rect 96x32', (0, 0, 96, 32), dest=(100, 32))
    for offset in (0x20, 0x100, 0x800, 0x1000):
        test(f'dest offset {offset:#x}', (0, 0, 64, 32), dest=(64, 32),
             offset=offset)

    # A surface pitch titles never use, not a whole number of 80-pixel tiles:
    # the console lays it out a tile a row, x 80 on in the next tile row.
    test('surface pitch 128', pitch=128)

    suite.set_inputs(rects.inputs, 2)
