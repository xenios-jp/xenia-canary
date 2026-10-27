"""Command processor packets that load state from memory or call other
packets, as titles issue them (feature census: IM_LOAD, LOAD_ALU_CONSTANT,
INDIRECT_BUFFER, NOP, SET_SHADER_CONSTANTS, SET_CONSTANT2, MEM_WRITE), every
address on a 128-byte line (cpalign has the titles' other alignments).

Vertex tests: one generated vertex whose shader exports float constants,
one element each (so an element shows what the shader saw), after the test's
packets. Type 0 writes start c248-c255 at A (1000 + 4 n + i for c[n]
component i) and c8-c83, c144-c159 at their own A; the packets then load B
(2000 + ...) or C (3000 + ...) over them.
Pixel tests: a one-pixel point whose pixel shader exports its constants
c252-c255 the same way, A from type 0 writes of c508-c511.
MEM_WRITE tests write dwords into the output directly; the test draws
nothing.
- type0: no packet, A.
- load c252 16 (index 1008, 16 dwords, the titles' commonest), c248 32
  (992, 32), c8 144 (32, 144), c36 192 (144, 192), c144 64 (576, 64): B over
  the range, A next to it.
- load twice: B then C over c252-c255; load then type0: a type 0 write of
  C to c252-c255 after the load.
- ib constants, ib nested constants: type 0 writes of B in an indirect buffer
  (in one an indirect buffer calls). ib load, ib nested load: the load in one.
  ib two: two in a row, B to c248-c251 and C to c252-c255. Every indirect
  buffer's packets fill whole 128-byte lines (type 2 after them).
- ib draw, ib nested draw: the draw itself in the indirect buffer.
- nop: a NOP whose payload is a type 0 write of B: A.
- im_load vs: IM_LOAD of a vertex shader exporting c255-c252 instead, over
  the immediate one; im_load vs ib, im_load vs nested ib: from an indirect
  buffer; im_load vs invalidate 7fff: after Direct3D's INVALIDATE_STATE.
- set_shader_constants, set_constant2: B through those packets, register
  index 0x43F0 (c252) and 16 dwords.
- mem_write endian 0-3: 0x11223344 at output + 16 e, endian e; mem_write
  three: three dwords 8in32 at output + 64. (Before the pixel tests: a trace
  dump plays a frame only up to its last draw.)
- ps type0, ps load c508 16 (2032, 16), ps load c504 32 (2016, 32), ps
  im_load: the pixel shader's.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cpraw as cp  # noqa: E402
import gpugen  # noqa: E402
import pixel  # noqa: E402
import pm4  # noqa: E402
import xenos as x  # noqa: E402

f2w = gpugen.f2w
EMPTY = (pm4.PRIM_POINT_LIST, 0, 0)
POINT = (pm4.PRIM_POINT_LIST, 0, 1)
RANGES = [(248, 8), (8, 36), (144, 16)]  # type 0 A: (first, count)


def value(base, n, i):
    return f2w(float(base + 4 * n + i))


def group(base, first, count):
    """Dwords of c[first] .. c[first + count - 1] at `base`."""
    return [value(base, n, i) for n in range(first, first + count)
            for i in range(4)]


def a_constants(bank=0):
    out = {}
    for first, count in RANGES:
        for n in range(first, first + count):
            out[bank + n] = [value(1000, n, i) for i in range(4)]
    return out


def const_exports(indices, bank=0):
    return [[x.alu(vector_op=x.V_MAX, vector_dest=x.EXPORT_DATA0,
                   vector_mask=0xF, export=True, src1=x.c(n), src2=x.c(n))]
            for n in indices]


def vertex_shader(indices):
    return gpugen.memexport_shader(0, [], const_exports(indices), fetches=[])


def pixel_shader(indices):
    return gpugen.memexport_pixel_shader([], const_exports(indices),
                                         x.r(0, 'x'))


def build(suite):
    cp.install(suite)
    # One vertex at pixel (0, 0), attribute 0, for the pixel tests.
    suite.set_inputs(pixel.vertex((0.0, 0.0), (0.0, 0.0, 0.0, 0.0)), 8)
    main = [252, 253, 254, 255]
    vs = vertex_shader(main)
    reverse = vertex_shader(main[::-1])
    constants = a_constants()

    def test(name, packets=(), indices=main, shader=None, draw=None):
        indices = list(indices)
        suite.test(name, shader or (vertex_shader(indices) if indices != main
                                    else vs),
                   constants=constants, elements=len(indices),
                   after_passes=[cp.entry(*packets)] if packets else (),
                   draw=draw)

    def load(base, first, count, **keywords):
        return cp.load_alu_constant(
            cp.Data(group(base, first, count), **keywords), 4 * first)

    def constants_b(base, first, count):
        return cp.float_constants(first, group(base, first, count))

    test('type0')
    test('load c252 16', load(2000, 252, 4), [251, 252, 255])
    test('load c248 32', load(2000, 248, 8), [248, 251, 252, 255])
    test('load c8 144', load(2000, 8, 36), [8, 9, 42, 43])
    test('load c36 192', load(2000, 36, 48), [35, 36, 43, 83])
    test('load c144 64', load(2000, 144, 16), [144, 151, 159])
    test('load twice', load(2000, 252, 4) + load(3000, 252, 4))
    test('load then type0', load(2000, 252, 4) + constants_b(3000, 252, 4))
    body = cp.body(constants_b(2000, 252, 4))
    test('ib constants', cp.ib(body))
    test('ib nested constants', cp.ib(cp.body(cp.ib(body))))
    load_body = cp.body(load(2000, 252, 4))
    test('ib load', cp.ib(load_body))
    test('ib nested load', cp.ib(cp.body(cp.ib(load_body))))
    test('ib two', cp.ib(cp.body(constants_b(2000, 248, 4))) +
         cp.ib(cp.body(constants_b(3000, 252, 4))), [248, 251, 252, 255])
    draw_body = cp.body(cp.draw_auto(pm4.PRIM_POINT_LIST, 1))
    test('ib draw', cp.ib(draw_body), draw=EMPTY)
    test('ib nested draw', cp.ib(cp.body(cp.ib(draw_body))), draw=EMPTY)
    hidden = cp.type0(gpugen.FLOAT_CONSTANTS + 4 * 252,
                      *group(2000, 252, 4))
    test('nop', cp.packet(cp.NOP, *hidden))
    shader = cp.Data(reverse.words())
    test('im_load vs', cp.im_load(pm4.VERTEX, shader))
    test('im_load vs ib', cp.ib(cp.body(cp.im_load(pm4.VERTEX, shader))))
    test('im_load vs nested ib', cp.ib(cp.body(cp.ib(cp.body(
        cp.im_load(pm4.VERTEX, shader))))))
    test('im_load vs invalidate 7fff',
         cp.packet(pm4.INVALIDATE_STATE, 0x7FFF) +
         cp.im_load(pm4.VERTEX, shader))
    test('set_shader_constants',
         cp.packet(cp.SET_SHADER_CONSTANTS, 0x43F0, *group(2000, 252, 4)))
    test('set_constant2',
         cp.packet(cp.SET_CONSTANT2, 0x43F0, *group(2000, 252, 4)))

    # MEM_WRITE.
    def mem_test(name, packets):
        suite.test(name, vs, constants=constants, elements=8,
                   after_passes=[cp.entry(*packets)], draw=EMPTY)

    mem_test('mem_write endian 0-3', [
        w for e in range(4)
        for w in cp.packet(cp.MEM_WRITE,
                           cp.Ref(output='mem_write endian 0-3',
                                  offset=16 * e, low=e), 0x11223344)])
    mem_test('mem_write three', cp.packet(
        cp.MEM_WRITE, cp.Ref(output='mem_write three', offset=64, low=2),
        0x11223344, 0x55667788, 0x99AABBCC))

    # Pixel shader constants: c508-c511 (pixel c252-c255).
    ps = pixel_shader(main)
    ps_constants = a_constants(256)

    def ps_test(name, packets=(), shader=ps):
        suite.test(name, gpugen.interpolating_vertex_shader(8),
                   constants=ps_constants, elements=4, pixel_shader=shader,
                   program_cntl=pixel.PROGRAM_CNTL,
                   registers=pixel.REGISTERS,
                   after_passes=[cp.entry(*packets)] if packets else (),
                   draw=POINT)

    def ps_load(base, first, count):
        return cp.load_alu_constant(
            cp.Data(group(base, first, count)), 4 * (256 + first))

    ps_test('ps type0')
    ps_test('ps load c508 16', ps_load(2000, 252, 4))
    ps_test('ps load c504 32', ps_load(2000, 248, 8))
    ps_test('ps im_load', cp.im_load(pm4.PIXEL,
                                     cp.Data(pixel_shader(main[::-1]).words())))
