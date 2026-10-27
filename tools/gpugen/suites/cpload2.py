"""cpload's pixel tests, which a stall after its MEM_WRITE of three dwords
kept from running on the console, and probes of what stalled it: what
SET_CONSTANT2 and a MEM_WRITE of more than one dword do (cpload: the
console left c252-c255 alone under SET_CONSTANT2 0x43F0, and wrote only the
first of three dwords). Constants as cpload's.
- ps type0, ps load c508 16, ps load c504 32, ps im_load: as cpload's.
- set_constant2 2102: SET_CONSTANT2 0x2102 of 1, the titles' form (a
  context register, here VGT_INDX_OFFSET), before an auto-index draw that
  doesn't set it: the vertex exports to slot 1 (output + 64) if it took.
- mem_write type2: MEM_WRITE of 0x11223344 and 0x80000000 (a type 2 packet,
  if the command processor reads it as one) to output + 128.
- mem_write packet: MEM_WRITE of 0x11223344 and then a type 0 write of B to
  c252-c255 (17 dwords) to output + 128, then a draw exporting c252-c255 (B
  if the command processor ran the extra words as packets, A if not).
- set_constant2 4102: as 2102 at 0x4102, which is c64.z if the index is a
  register and VGT_INDX_OFFSET if only its low bits count. Last, being the
  least known.
- after: ps type0 again, to show the stream went on.
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

POINT = cpload.POINT
EMPTY = cpload.EMPTY


def build(suite):
    cp.install(suite)
    suite.set_inputs(pixel.vertex((0.0, 0.0), (0.0, 0.0, 0.0, 0.0)), 8)
    # Room for a vertex exporting from VGT_INDX_OFFSET 1, and for MEM_WRITE
    # beside the exports.
    suite.export_slots = 4
    main = [252, 253, 254, 255]
    vs = cpload.vertex_shader(main)
    constants = cpload.a_constants()
    b = cpload.group(2000, 252, 4)
    ps = cpload.pixel_shader(main)

    def ps_test(name, packets=()):
        suite.test(name, gpugen.interpolating_vertex_shader(8),
                   constants=cpload.a_constants(256), elements=4,
                   pixel_shader=ps,
                   program_cntl=pixel.PROGRAM_CNTL,
                   registers=pixel.REGISTERS,
                   after_passes=[cp.entry(*packets)] if packets else (),
                   draw=POINT)

    def ps_load(first, count):
        return cp.load_alu_constant(
            cp.Data(cpload.group(2000, first, count)), 4 * (256 + first))

    ps_test('ps type0')
    ps_test('ps load c508 16', ps_load(252, 4))
    ps_test('ps load c504 32', ps_load(248, 8))
    ps_test('ps im_load', cp.im_load(pm4.PIXEL, cp.Data(
        cpload.pixel_shader(main[::-1]).words())))

    def test(name, packets, draw=EMPTY):
        suite.test(name, vs, constants=constants, elements=4,
                   after_passes=[cp.entry(*packets)], draw=draw)

    # An auto-index draw of one point that leaves VGT_INDX_OFFSET alone.
    draw = (cp.type0('COHER_DEST_BASE_0', 0) +
            cp.type0('WAIT_UNTIL', 0x00020000) +
            cp.packet(pm4.DRAW_INDX_2, pm4.PRIM_POINT_LIST |
                      (pm4.SOURCE_AUTO_INDEX << 6) | (1 << 16)) +
            cp.type0('VGT_INDX_OFFSET', 0) +
            cp.type0('COHER_DEST_BASE_0', 0) +
            cp.packet(pm4.EVENT_WRITE, 6))
    def set_constant2(index):
        test(f'set_constant2 {index:x}', cp.type0('VGT_INDX_OFFSET', 0) +
             cp.packet(cp.SET_CONSTANT2, index, 1) + draw)

    set_constant2(0x2102)
    test('mem_write type2', cp.packet(
        cp.MEM_WRITE, cp.Ref(output='mem_write type2', offset=128, low=2),
        0x11223344, cp.TYPE2))
    # The type 0 header right after the first dword, its payload on 16
    # bytes.
    test('mem_write packet', [cp.Pad(4, 0)] + cp.packet(
        cp.MEM_WRITE, cp.Ref(output='mem_write packet', offset=128, low=2),
        0x11223344, *cp.type0(gpugen.FLOAT_CONSTANTS + 4 * 252, *b)) +
        cp.draw_auto(pm4.PRIM_POINT_LIST, 1))
    set_constant2(0x4102)
    ps_test('after')
