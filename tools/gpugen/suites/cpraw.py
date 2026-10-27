"""Raw PM4 packets for the cp* suites (cpload, cpbins, cpext, cpalign):
packets a gpugen Test has no field for, written where the test's `registers`
or `after_passes` go. A Raw stands in for a register name there; the stream
gets its packets instead of a type 0 write.

The memory a packet reads (an indirect buffer's packets, IM_LOAD's shader,
LOAD_ALU_CONSTANT's constants) goes in the ring just before it, as the
payload of a NOP the command processor skips, starting at `align` bytes plus
`skew` (default a 128-byte line), so the console and a trace replay both have
it in place before the packet reads it. Addresses are relocated like the
rest of the stream's: Ref(data) is a Data's block offset, Ref(output=name) a
test's output.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pm4  # noqa: E402

NOP = 0x10
IM_LOAD = 0x27
LOAD_ALU_CONSTANT = 0x2F
INDIRECT_BUFFER = 0x3F
MEM_WRITE = 0x3D
EVENT_WRITE_EXT = 0x5A
SET_CONSTANT2 = 0x55
SET_SHADER_CONSTANTS = 0x56
CONTEXT_UPDATE = 0x5E
SET_BIN_MASK = 0x50
SET_BIN_SELECT = 0x51
SET_BIN_MASK_LO, SET_BIN_MASK_HI = 0x60, 0x61
SET_BIN_SELECT_LO, SET_BIN_SELECT_HI = 0x62, 0x63
SCREEN_EXT_INIT, SCREEN_EXT_RPT = 25, 26
TYPE2 = 0x80000000

# The block layout of the build in progress (gpugen.build's), for Refs to
# outputs.
LAYOUT = {}


class Ref:
    """A dword holding an address: of `data` (a Data) or of test `output`'s
    output, plus `offset` bytes, with `low` ORed into bits 0-1 (an
    endianness or shader type)."""

    def __init__(self, data=None, offset=0, low=0, output=None):
        self.data, self.offset, self.low, self.output = data, offset, low, output


class Size:
    """A dword holding a Data's length in dwords, as placed."""

    def __init__(self, data):
        self.data = data


class Pad:
    """Type 2 filler until the next dword's ring index is `remainder` mod
    `modulus` (the ring is 64 KiB aligned): Pad(4, 3) before a type 0 write
    of float constants puts its payload on 16 bytes, as Direct3D does."""

    def __init__(self, modulus, remainder):
        self.modulus, self.remainder = modulus, remainder


class Data:
    """Words (ints, Refs, Sizes and Pads) the GPU reads from memory, placed
    `skew` bytes past an `align`-byte boundary; with `lines`, padded with
    type 2 to whole 128-byte lines, as play.py pads what its indirect
    buffers call."""

    def __init__(self, words, align=128, skew=0, lines=False):
        self.words = list(words)
        self.align, self.skew, self.lines = align, skew, lines


class Raw(int):
    """Packets as a `registers` entry: (Raw(words...), ()). Words are ints,
    Refs, Sizes or Pads."""

    def __new__(cls, *words):
        self = super().__new__(cls, 0xFFFFFFFF)
        self.packet_words = [w for group in words for w in
                             (group if isinstance(group, (list, tuple))
                              else [group])]
        return self

    def emit(self, s):
        placed = {}  # id(Data) -> (ring index, dwords)

        def expand(words, start):
            out = []
            for w in words:
                if isinstance(w, Pad):
                    while (start + len(out)) % w.modulus != w.remainder:
                        out.append(TYPE2)
                else:
                    out.append(w)
            return out

        def place(data):
            if id(data) in placed:
                return
            for w in data.words:
                if isinstance(w, (Ref, Size)) and w.data is not None:
                    place(w.data)
            # The payload of a NOP, at the boundary plus skew.
            start = len(s.words) + 1
            while (4 * start - data.skew) % data.align:
                start += 1
            words = expand(data.words, start)
            if data.lines:
                words += [TYPE2] * (-len(words) % 32)
            s.words += [TYPE2] * (start - 1 - len(s.words))
            s.words.append(pm4.type3(NOP, len(words)))
            placed[id(data)] = (start, len(words))
            put(words)

        def put(words):
            for w in words:
                if isinstance(w, Ref):
                    if w.data is not None:
                        value = 4 * placed[id(w.data)][0] + w.offset
                    else:
                        value = LAYOUT['outputs'][w.output] + w.offset
                    s.relocations.append((len(s.words), 0))
                    s.words.append(value | w.low)
                elif isinstance(w, Size):
                    s.words.append(placed[id(w.data)][1])
                else:
                    s.words.append(w & 0xFFFFFFFF)

        for w in self.packet_words:
            if isinstance(w, (Ref, Size)) and w.data is not None:
                place(w.data)
        put(expand(self.packet_words, len(s.words)))


_reg = pm4.Stream.reg


def _raw_reg(self, name, *values):
    if isinstance(name, Raw):
        name.emit(self)
    else:
        _reg(self, name, *values)


if not getattr(pm4.Stream.reg, 'raw', False):
    _raw_reg.raw = True
    pm4.Stream.reg = _raw_reg


def install(suite):
    """Has the build that made `suite` keep its block layout in LAYOUT."""
    main = sys.modules[type(suite).__module__]
    build_stream = main.build_stream
    if getattr(build_stream, 'raw', False):
        return

    def keep_layout(suite, layout):
        LAYOUT.clear()
        LAYOUT.update(layout)
        return build_stream(suite, layout)

    keep_layout.raw = True
    main.build_stream = keep_layout


def entry(*words):
    """A `registers` entry of raw packet words."""
    return (Raw(*words), ())


def packet(op, *payload, predicate=False):
    return [pm4.type3(op, len(payload)) | int(predicate), *payload]


def type0(register, *values):
    return [pm4.type0(pm4.register(register), len(values)), *values]


def ib(data, predicate=False, size=None):
    """INDIRECT_BUFFER of a Data's packets."""
    return packet(INDIRECT_BUFFER, Ref(data),
                  Size(data) if size is None else size, predicate=predicate)


def im_load(shader_type, data):
    """IM_LOAD of a Data holding the shader's words, start 0."""
    return packet(IM_LOAD, Ref(data, low=shader_type), Size(data))


def load_alu_constant(data, index, count=None, kind=0):
    """LOAD_ALU_CONSTANT of `count` dwords (default the Data's) from a Data
    to constant dword `index` of bank `kind` (0 ALU: dword 4 n + i is c[n]
    component i, the pixel shader's from 1024)."""
    return packet(LOAD_ALU_CONSTANT, Ref(data), (kind << 16) | index,
                  Size(data) if count is None else count)


def body(*words, **keywords):
    """A Data of packets for an indirect buffer, in whole 128-byte lines."""
    keywords.setdefault('lines', True)
    return Data([w for group in words for w in group], **keywords)


def float_constants(first, values):
    """A type 0 write of float constants from c[first] (whole groups of 4,
    as Direct3D writes them), its payload on 16 bytes."""
    if first % 4 or len(values) % 16:
        raise ValueError('float constants go in whole aligned groups of 4')
    return [Pad(4, 3)] + type0(0x4000 + 4 * first, *values)


def draw_auto(primitive, count, predicate=False, first=0):
    """gpugen's draw bracket, as raw words: wait for idle, the index offset,
    DRAW_INDX_2 auto-index, the offset back, a cache flush."""
    initiator = primitive | (pm4.SOURCE_AUTO_INDEX << 6) | (count << 16)
    return (type0('COHER_DEST_BASE_0', 0) + type0('WAIT_UNTIL', 0x00020000) +
            type0('VGT_INDX_OFFSET', first) +
            packet(pm4.DRAW_INDX_2, initiator, predicate=predicate) +
            type0('VGT_INDX_OFFSET', 0) + type0('COHER_DEST_BASE_0', 0) +
            packet(pm4.EVENT_WRITE, 6))


def bins(mask=None, select=None):
    """SET_BIN_MASK_LO/HI and SET_BIN_SELECT_LO/HI of 64-bit values."""
    words = []
    if mask is not None:
        words += packet(SET_BIN_MASK_LO, mask & 0xFFFFFFFF)
        words += packet(SET_BIN_MASK_HI, mask >> 32)
    if select is not None:
        words += packet(SET_BIN_SELECT_LO, select & 0xFFFFFFFF)
        words += packet(SET_BIN_SELECT_HI, select >> 32)
    return words


# The bin state games leave between tiles: every bin, low word.
BINS_DEFAULT = (0xFFFFFFFF, 0xFFFFFFFF)
