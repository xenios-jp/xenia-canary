"""cpload's packets at the alignments titles read their memory at, which
cpload keeps on 128-byte lines (feature census: titles' LOAD_ALU_CONSTANT
sources sit on 16, 32 or 64 bytes as often as on 128, IM_LOAD shaders and
indirect buffers on 32 and 64, and indirect buffers end anywhere), and
Halo 3's CONTEXT_UPDATE 0. Untried on the console, so each in its own
segment, the least likely to stall first: a stall names its test.

Constants, shaders and outputs as cpload's: A from type 0 writes, B loaded.
- load 64, load 32, load 16: LOAD_ALU_CONSTANT of B to c252-c255 (index
  1008, 16 dwords) from 64, 32 and 16 bytes past a 128-byte line.
- ps load 32: the pixel shader's c252-c255 (index 2032) from 32 bytes past,
  Halo 3's commonest form.
- im_load 64, im_load 32: IM_LOAD of cpload's reversed vertex shader.
- ib unpadded: an indirect buffer of B's type 0 write, on a line, ending
  where its packets do (20 dwords).
- ib 64, ib 32: that indirect buffer 64 and 32 bytes past a line.
- ib nested 32: an indirect buffer 32 bytes past a line calling one 32 bytes
  past another that draws.
- context_update: CONTEXT_UPDATE 0 between the constants and the draw.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cpload  # noqa: E402
import cpraw as cp  # noqa: E402
import gpugen  # noqa: E402
import pixel  # noqa: E402
import pm4  # noqa: E402


def build(suite):
    cp.install(suite)
    suite.set_inputs(pixel.vertex((0.0, 0.0), (0.0, 0.0, 0.0, 0.0)), 8)
    main = [252, 253, 254, 255]
    vs = cpload.vertex_shader(main)
    reverse = cpload.vertex_shader(main[::-1])
    constants = cpload.a_constants()
    b = cpload.group(2000, 252, 4)

    def test(name, packets, draw=None):
        suite.test(name, vs, constants=constants, elements=4,
                   after_passes=[cp.entry(*packets)], draw=draw)

    for skew in (64, 32, 16):
        test(f'load {skew}', cp.load_alu_constant(cp.Data(b, skew=skew),
                                                  4 * 252))
    suite.test('ps load 32', gpugen.interpolating_vertex_shader(8),
               constants=cpload.a_constants(256), elements=4,
               pixel_shader=cpload.pixel_shader(main),
               program_cntl=pixel.PROGRAM_CNTL, registers=pixel.REGISTERS,
               after_passes=[cp.entry(cp.load_alu_constant(
                   cp.Data(b, skew=32), 4 * (256 + 252)))],
               draw=cpload.POINT)
    for skew in (64, 32):
        test(f'im_load {skew}', cp.im_load(
            pm4.VERTEX, cp.Data(reverse.words(), skew=skew)))
    constants_b = cp.float_constants(252, b)
    test('ib unpadded', cp.ib(cp.body(constants_b, lines=False)))
    for skew in (64, 32):
        test(f'ib {skew}', cp.ib(cp.body(constants_b, lines=False,
                                         skew=skew)))
    inner = cp.body(cp.draw_auto(pm4.PRIM_POINT_LIST, 1), lines=False,
                    skew=32)
    test('ib nested 32', cp.ib(cp.body(cp.ib(inner), lines=False, skew=32)),
         draw=cpload.EMPTY)
    test('context_update', cp.float_constants(252, b) +
         cp.packet(cp.CONTEXT_UPDATE, 0))
