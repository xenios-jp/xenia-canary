"""PA_SC_AA_MASK: which samples of which pixels a draw writes for each of the
mask's 16 bits, at 1x, 2x and 4x.

Every test fills its region (edramraw's, 80x32 pixels of k_8_8_8_8 at the
sample count) with FILL, draws texture 'bytes 0' over the whole region with
the test's mask, and reads every tile back through edramraw's one-sample
k_8_8_8_8 view (the converting Resolve that copies the EDRAM words
unchanged), so each sample of each pixel shows FILL or the texel. The masks
are each bit alone, then patterns: none, all, each nibble and alternating
bits, which say whether the bits go by sample, by pixel of a 2x2 quad or both.
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import edramraw  # noqa: E402
import gpugen  # noqa: E402
import rb  # noqa: E402
import rbdraw  # noqa: E402

MASKS = [1 << bit for bit in range(16)] + [
    0x0000, 0xFFFF, 0x000F, 0x00F0, 0x0F00, 0xF000, 0x5555, 0xAAAA, 0x3333,
    0xCCCC, 0x1111, 0x8888]
# (name, MSAA samples log2, base tile, tiles)
TARGETS = (('1x', 0, 0, 2), ('2x', 1, 300, 4), ('4x', 2, 700, 8))


def build(suite):
    suite.raw = True
    rng = random.Random(0xED7A)
    suite.add_texture('bytes 0', edramraw.texture(edramraw.bytes_texels(rng)))
    suite.add_texture('fill', rb.fill(edramraw.FILL))
    shapes = rbdraw.Shapes()
    write_rect = shapes.rectangles([(0, 0, edramraw.WIDTH, edramraw.HEIGHT)])
    for name, samples, base, tiles in TARGETS:
        rows = edramraw.TILE_HEIGHT * tiles
        fill_rect = shapes.rectangles([(0, 0, edramraw.TILE_WIDTH, rows)])
        read_rect = shapes.rectangles([(0, 0, edramraw.TILE_WIDTH, rows)])
        read = gpugen.resolve_registers(
            edramraw.TILE_WIDTH, rows, edramraw.K_8888, 4, command=1,
            rect=2 * read_rect[1], dest=(edramraw.DEST_PITCH, rows),
            registers=edramraw.surface(base, rows=rows))
        for mask in MASKS:
            passes = [(edramraw.surface(base, rows=rows), rb.fetch('fill'),
                       fill_rect)]
            suite.test(
                f'{name} mask {mask:#06x}', gpugen.resolve_vertex_shader(),
                constants=rb.CONSTANTS, pixel_shader=rb.pixel_shader(),
                program_cntl=rb.PROGRAM_CNTL, passes=passes,
                after_passes=edramraw.surface(base, samples=samples) +
                [('PA_SC_AA_MASK', [mask])],
                fetch_constants=edramraw.fetch('bytes 0'), draw=write_rect,
                resolve=read)
    suite.set_inputs(shapes.inputs, 2)
