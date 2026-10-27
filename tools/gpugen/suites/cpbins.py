"""Bin predication, as titles' predicated tiling uses it (feature census:
SET_BIN_MASK_LO/HI, SET_BIN_SELECT_LO/HI, SET_BIN_MASK, SET_BIN_SELECT and
the predicate bit, bit 0 of a type 3 header, on DRAW_INDX, DRAW_INDX_2,
INDIRECT_BUFFER, SET_CONSTANT and SET_CONSTANT2).

Each test sets the bin mask and select with the titles' values, then issues a
predicated packet: a draw of one generated vertex exporting c252-c255 (A =
1000 + 4 n + i, from type 0 writes), so its output is A if the draw ran and
the sentinel if not, or a write of B (2000 + ...) to c252-c255 before an
unpredicated draw. Then it puts back the bins titles leave between tiles:
mask and select 0xFFFFFFFF, high words 0. Names give mask and select as 64
bits, hi:lo where the high word isn't 0.
- unpredicated: mask 0x80000000, select 0xC, no predicate bit.
- draw2 / draw: DRAW_INDX_2 / DRAW_INDX (VIZ token 0) predicated under
  mask/select pairs titles set, every 32-bit pair, and the high words.
- set_bin_mask 1 0, set_bin_select 1 0: the two-dword packets' order, the
  other one set by halves.
- ib bins: SET_BIN_MASK_LO 3 in an indirect buffer; the draw after it.
- ib pass, ib skip: a predicated INDIRECT_BUFFER holding the draw.
- set_constant alu pass / skip: SET_CONSTANT of B to c252-c255, predicated.
- set_constant regs pass / skip: SET_CONSTANT of registers (bank 4, the
  form titles predicate), VGT_INDX_OFFSET 1, so the draw's vertex is 1 and
  it exports to the second slot.
- set_constant2 pass / skip: SET_CONSTANT2 of B to c252-c255 (register
  0x43F0), predicated.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cpload  # noqa: E402
import cpraw as cp  # noqa: E402
import gpugen  # noqa: E402
import pm4  # noqa: E402

EMPTY = cpload.EMPTY
# Every (mask, select) titles predicate with, and the high words.
PAIRS = [(0xFFFFFFFF, 0xFFFFFFFF), (0xFFFFFFFF, 0x80000003),
         (0x80000000, 0xC), (0x80000000, 0x80000003), (0x80000000, 0x30),
         (0x800000FF, 0xC), (0x800000FF, 0xC0), (0x8000000F, 0x80000003),
         (0x3, 0xC), (0xC, 0xC), (0x30, 0xC0), (0xC0, 0xC0), (0x3, 0x3),
         (0, 0xFFFFFFFF), (0xFFFFFFFF, 0),
         (1 << 32, 1 << 32), (1 << 32, 1), (1, 1 << 32),
         (0xFFFFFFFF | 1 << 32, 1 << 32), (1 << 63 | 0x3F << 32, 1 << 63)]


def name(v):
    return f'{v >> 32:x}:{v & 0xFFFFFFFF:x}' if v >> 32 else f'{v:x}'


def restore():
    return cp.bins(*cp.BINS_DEFAULT)


def draw_indx(predicate):
    """DRAW_INDX auto-index with VIZ token 0, in gpugen's bracket."""
    initiator = pm4.PRIM_POINT_LIST | (pm4.SOURCE_AUTO_INDEX << 6) | (1 << 16)
    return (cp.type0('COHER_DEST_BASE_0', 0) +
            cp.type0('WAIT_UNTIL', 0x00020000) +
            cp.type0('VGT_INDX_OFFSET', 0) +
            cp.packet(pm4.DRAW_INDX, 0, initiator, predicate=predicate) +
            cp.type0('COHER_DEST_BASE_0', 0) + cp.packet(pm4.EVENT_WRITE, 6))


def build(suite):
    cp.install(suite)
    suite.set_generated(1)
    # Two slots: a draw from VGT_INDX_OFFSET 1 exports to the second.
    suite.export_slots = 2
    vs = cpload.vertex_shader([252, 253, 254, 255])
    constants = cpload.a_constants()
    b = cpload.group(2000, 252, 4)

    def test(test_name, *packets):
        words = [w for p in packets for w in p] + restore()
        suite.test(test_name, vs, constants=constants, elements=4,
                   after_passes=[cp.entry(*words)], draw=EMPTY)

    point = cp.draw_auto(pm4.PRIM_POINT_LIST, 1)
    test('unpredicated', cp.bins(0x80000000, 0xC), point)
    for mask, select in PAIRS:
        test(f'draw2 mask {name(mask)} select {name(select)}',
             cp.bins(mask, select),
             cp.draw_auto(pm4.PRIM_POINT_LIST, 1, predicate=True))
    for mask, select in PAIRS[:6] + PAIRS[15:18]:
        test(f'draw mask {name(mask)} select {name(select)}',
             cp.bins(mask, select), draw_indx(True))
    predicated = cp.draw_auto(pm4.PRIM_POINT_LIST, 1, predicate=True)
    for which, op, other in (('mask', cp.SET_BIN_MASK, 'select'),
                             ('select', cp.SET_BIN_SELECT, 'mask')):
        for half in ('lo', 'hi'):
            value = 1 if half == 'lo' else 1 << 32
            test(f'set_bin_{which} 1 0 {other} {half} 1',
                 cp.bins(**{other: value}), cp.packet(op, 1, 0), predicated)
    test('ib bins', cp.bins(0xFFFFFFFF, 0xC),
         cp.ib(cp.body(cp.packet(cp.SET_BIN_MASK_LO, 3))), predicated)
    ib = cp.body(cp.draw_auto(pm4.PRIM_POINT_LIST, 1))
    test('ib pass', cp.bins(0xFFFFFFFF, 0x80000003), cp.ib(ib, True))
    test('ib skip', cp.bins(0x80000000, 0xC), cp.ib(ib, True))
    for result, select in (('pass', 0x80000003), ('skip', 0xC)):
        test(f'set_constant alu {result}', cp.bins(0x80000000, select),
             [cp.Pad(4, 2)] +
             cp.packet(pm4.SET_CONSTANT, (pm4.CONSTANT_ALU << 16) | 4 * 252,
                       *b, predicate=True), point)
        test(f'set_constant regs {result}', cp.bins(0x80000000, select),
             cp.type0('COHER_DEST_BASE_0', 0) +
             cp.type0('WAIT_UNTIL', 0x00020000) +
             cp.type0('VGT_INDX_OFFSET', 0) +
             cp.packet(pm4.SET_CONSTANT, (pm4.CONSTANT_REGISTERS << 16) |
                       (pm4.register('VGT_INDX_OFFSET') - 0x2000), 1,
                       predicate=True) +
             cp.packet(pm4.DRAW_INDX_2, pm4.PRIM_POINT_LIST |
                       (pm4.SOURCE_AUTO_INDEX << 6) | (1 << 16)) +
             cp.type0('VGT_INDX_OFFSET', 0) +
             cp.type0('COHER_DEST_BASE_0', 0) +
             cp.packet(pm4.EVENT_WRITE, 6))
        test(f'set_constant2 {result}', cp.bins(0x80000000, select),
             [cp.Pad(4, 2)] +
             cp.packet(cp.SET_CONSTANT2, 0x43F0, *b, predicate=True), point)
