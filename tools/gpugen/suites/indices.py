"""Indexed draws (DRAW_INDX from an index buffer) through the pixel suite's
probe: which vertices an index buffer draws for each index size and
endianness, the primitive reset index, VGT_INDX_OFFSET and the
VGT_MIN/MAX_VTX_INDX clamp.

Each test's indices are its own buffer, written so the GPU reads the listed
indices with the test's swap if it swaps each 16-bit index (8in16) or 32-bit
word (8in32, 16in32) of memory and then reads little-endian.
- points: 64 one-pixel points, vertex k at pixel (k % 16, k // 16) with
  attribute (k, 0, 0, 7), so the drawn pixels are the vertices the indices
  fetched. VGT_MAX_VTX_INDX 63 unless a test says otherwise, so an index out
  of range draws vertex 63 if it's clamped.
- buffer read as: the 8in16 buffer of 0-47 read with every swap.
- strip, triangle list, fan and line strip: primitives' blocks (one-hot
  weights, W the vertex) with reset indices among the indices, drawn from
  VGT_INDX_OFFSET = the block's first vertex, the clamp at the block's ends.
"""

import os
import random
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpugen  # noqa: E402
import pixel  # noqa: E402
import pm4  # noqa: E402
import primitives  # noqa: E402

I16, I32 = pm4.INDEX_INT16, pm4.INDEX_INT32
NONE, E8IN16, E8IN32, E16IN32 = (pm4.ENDIAN_NONE, pm4.ENDIAN_8IN16,
                                 pm4.ENDIAN_8IN32, pm4.ENDIAN_16IN32)
SWAPS = {NONE: 'none', E8IN16: '8in16', E8IN32: '8in32', E16IN32: '16in32'}
MULTI_PRIM_IB_ENA = 1 << 21
CULL_FRONT, CULL_BACK = primitives.CULL_FRONT, primitives.CULL_BACK
R16, R32 = 0xFFFF, 0xFFFFFFFF
POINTS = 64

STRIP = [(10 * (k // 2) + 0.25, 14.25 if k % 2 else 0.25) for k in range(10)]
BLOCKS = [
    ('points', pm4.PRIM_POINT_LIST,
     [pixel.vertex((k % 16, k // 16), (k, 0.0, 0.0, 7.0))
      for k in range(POINTS)]),
    ('strip', pm4.PRIM_TRIANGLE_STRIP,
     primitives.weighted(STRIP, lambda k: k % 3)),
    ('fan', pm4.PRIM_TRIANGLE_FAN,
     primitives.weighted(primitives.FAN,
                         lambda k: 0 if k == 0 else 1 + (k - 1) % 2)),
    ('line strip', pm4.PRIM_LINE_STRIP,
     primitives.weighted(primitives.LINE_STRIP, lambda k: k % 2)),
]


def memory(indices, index_format, swap):
    """Words (stored big-endian) the GPU reads as `indices`."""
    size = 2 if index_format == I16 else 4
    read = b''.join(struct.pack('<H' if size == 2 else '<I', i)
                    for i in indices)
    read += bytes(-len(read) % 4)
    out = bytearray()
    for w in range(0, len(read), 4):
        s = read[w:w + 4]
        out += {NONE: s, E8IN16: bytes((s[1], s[0], s[3], s[2])),
                E8IN32: s[::-1],
                E16IN32: bytes((s[2], s[3], s[0], s[1]))}[swap]
    return list(struct.unpack(f'>{len(out) // 4}I', out))


def build(suite):
    draws = pixel.set_blocks(suite, BLOCKS)
    probe = pixel.Probe()

    def test(name, block, indices, index_format=I16, swap=E8IN16,
             offset=0, clamp=None, reset=None, mode=0, words=None):
        primitive, first, count = draws[block]
        low, high = clamp or (first, first + count - 1)
        registers = [('VGT_MAX_VTX_INDX', [high, low]),
                     ('PA_SU_SC_MODE_CNTL', [
                         gpugen.MODE_CNTL | mode |
                         (MULTI_PRIM_IB_ENA if reset is not None else 0)])]
        if reset is not None:
            registers.append(('VGT_MULTI_PRIM_IB_RESET_INDX', [reset]))
        suite.add_texture(name, words or memory(indices, index_format, swap))
        probe.test(suite, name, (primitive, first + offset, len(indices)),
                   registers)
        suite.tests[-1].indices = (name, index_format, swap)

    rng = random.Random(0x1D)
    shuffled = rng.sample(range(48), 48)
    # Index sizes and endianness, Direct3D's first.
    test('points 16-bit 8in16', 'points', shuffled)
    test('points 32-bit 8in32', 'points', shuffled, I32, E8IN32)
    for swap in (NONE, E8IN32, E16IN32):
        test(f'points 16-bit {SWAPS[swap]}', 'points', shuffled, swap=swap)
    for swap in (NONE, E8IN16, E16IN32):
        test(f'points 32-bit {SWAPS[swap]}', 'points', shuffled, I32, swap)
    words = memory(shuffled, I16, E8IN16)
    for swap in (NONE, E8IN32, E16IN32):
        test(f'points 8in16 buffer read as {SWAPS[swap]}', 'points',
             shuffled, swap=swap, words=words)
    words = memory(shuffled, I32, E8IN32)
    for swap in (NONE, E8IN16, E16IN32):
        test(f'points 8in32 buffer read as 32-bit {SWAPS[swap]}', 'points',
             shuffled, I32, swap, words=words)
    # 32-bit indices past 24 bits.
    test('points 32-bit high byte', 'points',
         [k | (k % 4 + 1) << 24 for k in range(32)], I32, E8IN32)
    test('points 32-bit bit 23', 'points',
         [k | 0x800000 for k in range(32)], I32, E8IN32)
    # The clamp and VGT_INDX_OFFSET: which goes first, and the offset's sign.
    test('points out of range', 'points', list(range(40)) + [64, 100, R16])
    test('points clamp max 40', 'points', list(range(48)), clamp=(0, 40))
    test('points clamp min 8', 'points', list(range(16)), clamp=(8, 63))
    test('points offset 16', 'points', list(range(48)), offset=16)
    test('points offset 20 clamp max 40', 'points', list(range(48)),
         offset=20, clamp=(0, 40))
    test('points offset -16', 'points', list(range(16, 64)),
         offset=0xFFFFF0)
    test('points offset -16 32-bit', 'points', list(range(16, 64)), I32,
         E8IN32, offset=0xFFFFF0)
    test('points offset wraps 32-bit', 'points',
         [0xFFFFF0 + k for k in range(16)], I32, E8IN32, offset=16)
    # Reset indices in a point list, and against the offset.
    with_resets = [R16 if k % 5 == 4 else k for k in range(40)]
    test('points reset', 'points', with_resets, reset=R16)
    test('points reset off', 'points', with_resets)
    test('points reset 5 offset 10', 'points', list(range(40)), offset=10,
         reset=5)
    test('points reset 32-bit', 'points',
         [R32 if k % 5 == 4 else k for k in range(40)], I32, E8IN32,
         reset=0xFFFFFF)

    # Strips: a reset restarts the strip, and with it the winding parity.
    test('strip', 'strip', list(range(10)))
    test('strip 32-bit', 'strip', list(range(10)), I32, E8IN32)
    test('strip reset', 'strip', [0, 1, 2, 3, R16, 4, 5, 6, 7, 8, 9],
         reset=R16)
    for cull, bit in (('front', CULL_FRONT), ('back', CULL_BACK)):
        test(f'strip reset odd cull {cull}', 'strip',
             [0, 1, 2, R16, 3, 4, 5, 6, 7], reset=R16, mode=bit)
    test('strip reset short', 'strip', [0, 1, R16, 2, R16, R16, 3, 4, 5],
         reset=R16)
    test('strip reset off', 'strip', [0, 1, 2, 3, R16, 4, 5, 6, 7])
    test('strip reset 3', 'strip', list(range(10)), reset=3)
    test('strip reset 32-bit', 'strip',
         [0, 1, 2, 3, R32, 4, 5, 6, 7, 8, 9], I32, E8IN32, reset=0xFFFFFF)
    test('strip reset 32-bit 24 bits', 'strip',
         [0, 1, 2, 3, 0xFFFFFF, 4, 5, 6, 7, 8, 9], I32, E8IN32,
         reset=0xFFFFFF)
    test('strip reset 32-bit ffff', 'strip',
         [0, 1, 2, 3, R16, 4, 5, 6, 7, 8, 9], I32, E8IN32, reset=R16)
    # Lists: whether a reset index cuts a triangle list.
    for name, indices in (('triangle list', [0, 1, 2, 3, 4, 5, 6, 7, 8]),
                          ('triangle list reset', [0, 1, R16, 2, 3, 4, 5,
                                                   6, 7]),
                          ('triangle list reset off', [0, 1, R16, 2, 3, 4,
                                                       5, 6, 7])):
        primitive, first, count = draws['strip']
        draws[name] = (pm4.PRIM_TRIANGLE_LIST, first, count)
        test(name, name, indices,
             reset=R16 if name == 'triangle list reset' else None)
    test('fan reset', 'fan', [0, 1, 2, 3, R16, 0, 4, 5, 6], reset=R16)
    test('line strip reset', 'line strip', [0, 1, 2, R16, 3, 4], reset=R16)
