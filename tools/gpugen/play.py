#!/usr/bin/env python3
"""Plays a captured trace on a console through gpugen's executor, keeping
what every resolve wrote: one stage each, numbered by the trace's packets and
by Xenia's commands, so a backend's replay of the trace stopped after that
command compares with it byte for byte.

    python3 tools/gpugen/play.py TRACE.xtr OUTDIR [--microcode IMAGE]
        [--results NAME] [--next NAME] [--watchdog SECONDS] [--reboot]
        [--no-edram] [--no-zero]

OUTDIR gets gpugen.xex; gpugen.mem, the trace's memory, which the title reads
as it plays from game:\\, named by its SHA-1; gpugen.json; and gpugen.xtr, the trace
itself, which run.py replays a backend's stages from.

The build packs every page the trace reads or the GPU writes into the
executor's block, each run of pages at the same offset in its 64 KiB, and
rewrites every address the GPU uses to its page's block offset, which the
executor adds the block's physical address to: address-holding registers
(fetch constants, the resolve destination, the occlusion query report) and
the address fields of the packets that read or write memory. The packets go
into a command area the ring calls as an indirect buffer; the trace's own
indirect buffers and the constants LOAD_ALU_CONSTANT reads are copied there
too, prepared the same way, so the trace's memory reaches the GPU as it was
recorded. Writes of the display's and the kernel's registers are left out,
and bin predication is evaluated as host replays evaluate it, from Xenia's
initial state. A memory record ends a segment and is read in before the
next, with the GPU idle, which then invalidates its fetch caches; a resolve
ends one too, and once its fence lands the executor copies the destination
out as that stage.

Before the trace, unless --no-edram, a segment of its own reads the whole
eDRAM out with one converting Resolve (edramraw's 'span whole convert') to
an output of its own ahead of the stages, EDRAM_NAME, which run.py replays
the trace from instead of eDRAM all zeros and all ones. It writes nothing in
the trace's pages, and puts every register it wrote back as the setup left
it, so the trace starts from the state it would without it.

The setup also reads zeros, unless --no-zero, into every page of the trace's
that no read before its first packet fills whole: those the GPU writes and
the trace never reads, and the parts of those it reads only in part or
later, as a host replay has zeros there. Without it they hold whatever the
console's memory held, which differs from run to run.
"""

import argparse
import array
import hashlib
import json
import os
import shutil
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'suites'))
import edramraw  # noqa: E402
import gpugen  # noqa: E402
import pm4  # noqa: E402

# trace_protocol.h's TraceCommandType and MemoryEncodingFormat.
PRIMARY_START, PRIMARY_END, INDIRECT_START, INDIRECT_END = 0, 1, 2, 3
PACKET_START, PACKET_END, MEMORY_READ, MEMORY_WRITE = 4, 5, 6, 7
EDRAM_SNAPSHOT, EVENT, REGISTERS, GAMMA_RAMP = 8, 9, 10, 11
EDRAM_BYTES = 10 << 20
ENCODING_NONE = 0
ENCODING_SNAPPY = 1
TRACE_HEADER = struct.Struct('<I40sI')  # version, build commit, title id

PAGE = 0x1000
RUN_ALIGNMENT = 0x10000
# Zeros in the block, after the trace's pages, that addresses outside them
# point at, so the GPU reads none of the console's own memory, as a host
# replay reads zeros there: enough for any texture a draw fetches from one.
NULL_BYTES = 16 << 20
INDIRECT_BUFFER = 0x3F
NOP = 0x10

# Registers: RB_MODECONTROL, RB_COPY_CONTROL, RB_COPY_DEST_BASE,
# RB_COPY_DEST_PITCH, RB_COPY_DEST_INFO, RB_SAMPLE_COUNT_ADDR.
RB_MODECONTROL = 0x2208
RB_COPY_CONTROL, RB_COPY_DEST_BASE = 0x2318, 0x2319
RB_COPY_DEST_PITCH, RB_COPY_DEST_INFO = 0x231A, 0x231B
RB_COPY_SURFACE_SLICE = 0x2323
RB_SAMPLE_COUNT_ADDR = 0x2325
FETCH_CONSTANTS, FETCH_DWORDS = 0x4800, 192
# SET_CONSTANT and LOAD_ALU_CONSTANT's types that hold addresses.
CONSTANT_BASES = {1: FETCH_CONSTANTS, 4: 0x2000}
# Bytes per pixel of the formats a resolve writes.
COPY_BPP = {2: 1, 8: 1, 9: 1, 3: 2, 4: 2, 5: 2, 10: 2, 15: 2, 24: 2, 30: 2,
            6: 4, 7: 4, 14: 4, 16: 4, 17: 4, 25: 4, 31: 4, 36: 4,
            26: 8, 32: 8, 37: 8, 38: 16}
# The packets a trace may hold besides those prepare() rewrites or drops: no
# memory addresses in them.
PLAIN_OPCODES = {NOP, 0x23, 0x2B, 0x36, 0x3B, 0x46}
TYPE2_NOP = 0x80000000
LINE_WORDS = 32
# The registers of a trace's starting state a title gets: the rendering
# context and the constants, as a console trace holds them. Xenia's holds its
# whole register file, which has registers that act when written (WAIT_UNTIL,
# the coherency and VGT initiator registers).
STATE_RANGES = ((0x2000, 0x200F), (0x2080, 0x2082), (0x2100, 0x2114),
                (0x2180, 0x2184), (0x2200, 0x220B), (0x2280, 0x2294),
                (0x2300, 0x2325), (0x2380, 0x239F), (0x4000, 0x48BF),
                (0x4900, 0x4927))
# The eDRAM before the trace: its 2048 tiles of 80 x 16 words as one 1280 x
# 2048 one-sample k_8_8_8_8 surface, read by a converting Resolve to a tiled
# k_8_8_8_8 texture of big-endian words, which are the eDRAM's unchanged.
EDRAM_NAME = 'edram_before'
EDRAM_PITCH, EDRAM_ROWS = edramraw.SPAN_PITCH, edramraw.SPAN_ROWS
EDRAM_TILE_WIDTH, EDRAM_TILE_HEIGHT = edramraw.TILE_WIDTH, edramraw.TILE_HEIGHT
# Registers that act when written: the dump doesn't write them back.
ACTION_REGISTERS = {pm4.register('WAIT_UNTIL'),
                    pm4.register('COHER_STATUS_HOST')}


class TraceError(Exception):
    pass


def unsnappy(data):
    """Snappy's raw format decoded, as Xenia's trace writer compresses."""
    size = shift = o = 0
    while True:
        byte = data[o]
        o += 1
        size |= (byte & 127) << shift
        shift += 7
        if not byte & 128:
            break
    out = bytearray()
    while o < len(data):
        tag = data[o]
        o += 1
        kind = tag & 3
        if kind == 0:
            length = tag >> 2
            if length >= 60:
                extra = length - 59
                length = int.from_bytes(data[o:o + extra], 'little')
                o += extra
            length += 1
            out += data[o:o + length]
            o += length
            continue
        if kind == 1:
            length = 4 + (tag >> 2 & 7)
            offset = (tag >> 5) << 8 | data[o]
            o += 1
        else:
            width = 2 if kind == 2 else 4
            length = (tag >> 2) + 1
            offset = int.from_bytes(data[o:o + width], 'little')
            o += width
        if not 0 < offset <= len(out):
            raise TraceError('a bad snappy copy')
        start = len(out) - offset
        if offset >= length:
            out += out[start:start + length]
        else:
            for i in range(length):
                out.append(out[start + i])
    if len(out) != size:
        raise TraceError('a snappy block of the wrong size')
    return bytes(out)


def decode(encoding, data, size):
    """A record's data, decoded to `size` bytes."""
    if encoding == ENCODING_SNAPPY:
        data = unsnappy(data)
    elif encoding != ENCODING_NONE:
        raise TraceError(f'encoding {encoding}')
    if len(data) != size:
        raise TraceError(f'a record of {len(data)} bytes, not {size}')
    return data


def read_trace(path):
    """The trace's title id and its records: (kind, address or first
    register, packet words, memory bytes or register values). A packet
    comes at its end, after the memory it read; one starting an indirect
    buffer Xenia recorded the packets of doesn't come, its packets do, as
    Xenia's trace player runs them."""
    with open(path, 'rb') as f:
        data = f.read()
    version, _, title_id = TRACE_HEADER.unpack_from(data)
    if version != 1:
        raise TraceError(f'{path}: trace version {version}, not 1')

    def records():
        o = TRACE_HEADER.size
        packet = None
        while o < len(data):
            kind = struct.unpack_from('<I', data, o)[0]
            if kind == PACKET_START:
                base, count = struct.unpack_from('<II', data, o + 4)
                packet = kind, base, list(struct.unpack_from(
                    f'>{count}I', data, o + 12))
                o += 12 + 4 * count
            elif kind == PACKET_END:
                if packet:
                    yield packet
                packet = None
                o += 4
            elif kind in (PRIMARY_START, INDIRECT_START):
                if kind == INDIRECT_START:
                    packet = None
                count = struct.unpack_from('<I', data, o + 8)[0]
                o += 12 + 4 * count
            elif kind in (PRIMARY_END, INDIRECT_END):
                o += 4
            elif kind in (MEMORY_READ, MEMORY_WRITE):
                base, encoding, encoded, decoded = struct.unpack_from(
                    '<IIII', data, o + 4)
                # Writes are what the GPU wrote, not needed to play.
                yield kind, base, (decode(encoding, data[o + 20:o + 20 +
                                                         encoded], decoded)
                                   if kind == MEMORY_READ else b'')
                o += 20 + encoded
            elif kind == REGISTERS:
                first, count, _, encoding, size = struct.unpack_from(
                    '<IIB3xII', data, o + 4)
                values = decode(encoding, data[o + 24:o + 24 + size],
                                4 * count)
                yield kind, first, list(struct.unpack(f'<{count}I', values))
                o += 24 + size
            elif kind == EVENT:
                o += 8
            elif kind == EDRAM_SNAPSHOT:
                # The console's EDRAM can't be loaded: stages of what was in
                # it before the trace hold the console's own leftovers.
                encoding, encoded = struct.unpack_from('<II', data, o + 4)
                o += 12 + encoded
            elif kind == GAMMA_RAMP:
                # Scanout's, which no stage reads.
                encoding, encoded = struct.unpack_from('<4xII', data, o + 4)
                o += 16 + encoded
            else:
                raise TraceError(f'record {kind} at {o}, which only Xenia\'s '
                                 'own traces have')
    return title_id, records


def packet_dwords(header):
    kind = header >> 30
    return 1 if kind == 2 else 3 if kind == 1 else ((header >> 16) & 0x3FFF) + 2


def packets(words):
    i = 0
    while i < len(words):
        count = packet_dwords(words[i])
        if i + count > len(words):
            raise TraceError('a packet runs past its indirect buffer')
        yield words[i:i + count]
        i += count


def replayable_register(reg):
    """Rendering state, the shader GPR split (SQ_GPR_MANAGEMENT), and the
    cache and synchronization controls titles write in their streams."""
    return (reg >= 0x2000 or reg in (0x5C8, 0xD00, 0xE00, 0xE40, 0xF01) or
            0xA2F <= reg <= 0xA31)


def system_register(reg):
    """Scanout, and the kernel driver's interrupt acknowledgement and scratch
    writeback, which belong to the executor's own display and fences; waits
    that observed scratch values replay from their recorded memory."""
    return (0x1800 <= reg < 0x2000 or reg in (0x1DC, 0x1DD, 0x45E) or
            0x578 <= reg < 0x580)


class Memory:
    """The trace's memory as its records leave it at each point."""

    def __init__(self):
        self.pages = {}

    def write(self, address, data):
        address &= 0x1FFFFFFF
        done = 0
        while done < len(data):
            at = address + done
            offset = at & (PAGE - 1)
            count = min(PAGE - offset, len(data) - done)
            page = self.pages.setdefault(at >> 12, bytearray(PAGE))
            page[offset:offset + count] = data[done:done + count]
            done += count

    def words(self, address, count):
        out = []
        for i in range(count):
            at = (address + 4 * i) & 0x1FFFFFFF
            page = self.pages.get(at >> 12)
            if page is None:
                raise TraceError(f'{at:08X} read before the trace wrote it')
            out.append(struct.unpack_from('>I', page, at & (PAGE - 1))[0])
        return out


class Player:
    """Walks the trace as the console plays it. The first walk (`place`
    None) finds the pages the trace uses; the second rewrites the packets
    into the command area for the pages' block offsets in `place`."""

    def __init__(self, place=None, copies_offset=0, null_offset=0):
        self.place = place
        self.copies_offset = copies_offset
        self.null_offset = null_offset
        self.used = set()
        self.memory = Memory()
        self.registers = {}
        self.fetch = [0] * FETCH_DWORDS
        # How each fetch dword's address was rewritten, if it was.
        self.fetch_kind = [None] * FETCH_DWORDS
        # Bin predication from Xenia's initial state.
        self.bin_mask = self.bin_select = 0xFFFFFFFF
        self.packet = self.command = 0
        self.commands = []        # the command area's words
        self.relocations = []     # command area dword indices
        self.copies = []          # the out-of-line copies' words
        self.copy_relocations = []
        self.segment_start = 0
        self.segments = []        # (reads, first word, words, stages)
        self.reads = []           # (address, bytes, file offset)
        self.stages = []
        self.memory_bytes = 0
        # The pages reads before the first packet fill whole.
        self.preloaded = set()
        # The shader types (0 vertex, 1 pixel) the trace has loaded.
        self.shaders = set()
        self.counts = {'system register packets': 0, 'bin packets': 0,
                       'interrupts': 0, 'addresses outside the trace': 0,
                       'draws before the shaders': 0,
                       'registers outside the state': 0, 'swaps': 0}

    # Addresses.

    def mark(self, address, size):
        address &= 0x1FFFFFFF
        for page in range(address >> 12, (address + size - 1 >> 12) + 1):
            self.used.add(page)

    def offset(self, address, what):
        """The block offset of a trace address the GPU uses; for one
        outside the trace's pages, which a fetch may hold unused, the zeros
        at null_offset."""
        address &= 0x1FFFFFFF
        page = self.place.get(address >> 12)
        if page is None:
            if what:
                raise TraceError(f'packet {self.packet}: {what} {address:08X} '
                                 'outside the trace\'s pages')
            self.counts['addresses outside the trace'] += 1
            page = self.null_offset
        return page + (address & (PAGE - 1))

    def rewrite(self, value, mask, what=None):
        """VALUE with its address bits, `mask`, moved to the block offset;
        whether it was."""
        # Zero is no address, to the GPU as to Xenia (a texture's mips
        # following its base): the title had it there too.
        if self.place is None or not value & mask:
            return value, False
        return (value & ~mask & 0xFFFFFFFF) | self.offset(value & mask,
                                                          what), True

    # Registers.

    def track(self, reg, value):
        self.registers[reg] = value

    def register_value(self, reg, value):
        """VALUE written to REG, an address in it rewritten; whether it
        was. The registers keep the trace's values."""
        self.track(reg, value)
        moved = False
        if FETCH_CONSTANTS <= reg < FETCH_CONSTANTS + FETCH_DWORDS:
            index = reg - FETCH_CONSTANTS
            k = index % 6
            self.fetch_kind[index] = None
            # A slot is a texture's six dwords or three vertex fetches'
            # pairs, as its first dword says.
            texture = self.fetch[index - k] & 3 == 2 if k else value & 3 == 2
            if not index & 1 and value & 3 == 3 and not texture:
                value, moved = self.rewrite(value, 0x1FFFFFFC)   # vertex
                self.fetch_kind[index] = 'vertex'
            elif k in (1, 5) and texture:
                value, moved = self.rewrite(value, 0x1FFFF000)   # texture
                self.fetch_kind[index] = 'texture'
            self.fetch[index] = self.registers[reg]
        elif reg in (RB_COPY_DEST_BASE, RB_SAMPLE_COUNT_ADDR):
            value, moved = self.rewrite(value, 0x1FFFFFFC)
        return value, moved

    def check_fetches(self):
        """Refuses a draw any of whose fetch constants was rewritten as
        another kind of fetch than its slot now is: written while the slot
        was one, it holds an address left as the trace had it, or a word
        that isn't one rewritten."""
        for slot in range(0, FETCH_DWORDS, 6):
            if self.fetch[slot] & 3 == 2:
                kinds = {1: 'texture', 5: 'texture'}
            else:
                kinds = {i: 'vertex' for i in (0, 2, 4)
                         if self.fetch[slot + i] & 3 == 3}
            for i in range(6):
                if self.fetch_kind[slot + i] != kinds.get(i):
                    raise TraceError(f'packet {self.packet}: fetch constant '
                                     f'dword {slot + i} rewritten as '
                                     f'{self.fetch_kind[slot + i]} for a '
                                     f'{kinds.get(i)} fetch')

    # Packets.

    def evaluate_bins(self, words):
        """False for a packet the GPU doesn't get: the bin packets, and
        packets predicated off."""
        header = words[0]
        if header >> 30 != 3:
            return True
        op = header >> 8 & 127
        if op in (0x50, 0x51) and len(words) == 3:
            value = words[1] << 32 | words[2]
            if op == 0x50:
                self.bin_mask = value
            else:
                self.bin_select = value
        elif 0x60 <= op <= 0x63 and len(words) == 2:
            old = self.bin_mask if op < 0x62 else self.bin_select
            new = ((old & 0xFFFFFFFF) | words[1] << 32 if op & 1 else
                   (old & ~0xFFFFFFFF) | words[1])
            if op < 0x62:
                self.bin_mask = new
            else:
                self.bin_select = new
        elif header & 1:
            if not self.bin_mask & self.bin_select:
                self.counts['bin packets'] += 1
                return False
            words[0] = header & ~1
            return True
        else:
            return True
        self.counts['bin packets'] += 1
        return False

    def register_packet_kind(self, words):
        """'system' for a packet writing only system registers (left out),
        'state' for one writing only replayable ones, None for others."""
        header = words[0]
        kind = header >> 30
        if kind == 0:
            first = header & 0x7FFF
            regs = [first + (0 if header & 0x8000 else i)
                    for i in range(len(words) - 1)]
        elif kind == 1:
            regs = [header & 0x7FF, header >> 11 & 0x7FF]
        elif kind == 3 and header >> 8 & 127 == 0x21:
            regs = [words[1] & 0x1FFF]
        elif kind == 3 and header >> 8 & 127 in (0x55, 0x56):
            regs = [(words[1] & 0xFFFF) + i
                    for i in range(len(words) - 2)]
        else:
            return None
        system = [r for r in regs if system_register(r)]
        other = [r for r in regs if not system_register(r)]
        for reg in other:
            if not replayable_register(reg):
                raise TraceError(f'packet {self.packet} writes register '
                                 f'{reg:X}')
        if system and other:
            raise TraceError(f'packet {self.packet} mixes system and other '
                             'registers')
        return 'system' if system else 'state'

    def resolve(self):
        """(base, size) of the destination a draw now resolves to, if it is
        in copy mode and writes memory."""
        regs = self.registers
        if (regs.get(RB_MODECONTROL, 0) & 7 != 6 or
                regs.get(RB_COPY_CONTROL, 0) >> 20 & 3 == 3):
            return None
        info = regs.get(RB_COPY_DEST_INFO, 0)
        bpp = COPY_BPP.get(info >> 7 & 63)
        if not bpp:
            raise TraceError(f'packet {self.packet}: resolve format '
                             f'{info >> 7 & 63}')
        pitch = regs.get(RB_COPY_DEST_PITCH, 0)
        size = (((pitch & 0x3FFF) + 31 & ~31) *
                ((pitch >> 16 & 0x3FFF) + 31 & ~31) * bpp)
        if info & 8:
            width, height = pitch & 0x3FFF, pitch >> 16 & 0x3FFF
            storage_height = self.resolve_storage_height()
            # Keep the full eight-slice tile group. Slice addressing interleaves
            # pages, so a 2D-sized capture can miss both writes and pixels.
            size = max(size, ((width + 31) & ~31) *
                       ((max(height, storage_height) + 31) & ~31) * 8 * bpp,
                       4096)
        return regs.get(RB_COPY_DEST_BASE, 0) & 0x1FFFF000, size

    def resolve_storage_height(self):
        pitch = self.registers.get(RB_COPY_DEST_PITCH, 0)
        width = pitch & 0x3FFF
        area = self.registers.get(RB_COPY_SURFACE_SLICE, 0)
        return area // width if area and width else pitch >> 16 & 0x3FFF

    def prepare(self, words, depth):
        """The packet as the console gets it, at `depth` 0 in the command
        area or 1 in an indirect buffer, and the indices of its words that
        hold block offsets; None for one left out."""
        words = list(words)
        if not self.evaluate_bins(words):
            return None
        kind = self.register_packet_kind(words)
        if kind == 'system':
            self.counts['system register packets'] += 1
            return None
        header = words[0]
        moved = []
        if header >> 30 == 0:
            first = header & 0x7FFF
            for i in range(1, len(words)):
                reg = first + (0 if header & 0x8000 else i - 1)
                words[i], m = self.register_value(reg, words[i])
                if m:
                    moved.append(i)
            return words, moved
        if header >> 30 == 1:
            for i, reg in ((1, header & 0x7FF), (2, header >> 11 & 0x7FF)):
                words[i], m = self.register_value(reg, words[i])
                if m:
                    moved.append(i)
            return words, moved
        if header >> 30 == 2:
            return words, moved
        op = header >> 8 & 127
        count = len(words)
        if op in (0x27, 0x2B) and count >= 2:
            self.shaders.add(words[1] & 3)
        # A draw before the trace loads its shaders would run whatever the
        # GPU last had, the setup's: Xenia draws nothing there, and the
        # console wedges on it.
        if op in (0x22, 0x36) and not {0, 1} <= self.shaders:
            self.counts['draws before the shaders'] += 1
            return None

        def field(i, what):
            words[i], m = self.rewrite(words[i], 0x1FFFFFFC, what)
            if m:
                moved.append(i)

        if op == 0x21:
            reg = words[1] & 0x1FFF
            if reg in (RB_MODECONTROL, RB_COPY_CONTROL, RB_COPY_DEST_BASE,
                       RB_COPY_DEST_PITCH, RB_COPY_DEST_INFO,
                       RB_COPY_SURFACE_SLICE, RB_SAMPLE_COUNT_ADDR) or reg >= FETCH_CONSTANTS:
                raise TraceError(f'packet {self.packet}: REG_RMW of {reg:X}')
        elif op == 0x22:
            if count >= 5 and not words[2] >> 6 & 3:
                field(3, 'index buffer')
        elif op == 0x27:
            if count >= 3:
                field(1, 'shader')
        elif op in (0x2D, 0x55, 0x56):
            base = (0 if op in (0x55, 0x56) else
                    CONSTANT_BASES.get(words[1] >> 16 & 255))
            for i in range(2, count):
                if base is not None:
                    reg = base + (words[1] & (0xFFFF if op in (0x55, 0x56) else
                                             0x7FF)) + i - 2
                    words[i], m = self.register_value(reg, words[i])
                    if m:
                        moved.append(i)
        elif op == 0x2F:
            if count != 4:
                raise TraceError(f'packet {self.packet}: LOAD_ALU_CONSTANT '
                                 f'of {count} words')
            base = CONSTANT_BASES.get(words[2] >> 16 & 255)
            values = self.memory.words(words[1], words[3] & 0xFFF)
            if self.place is None:
                self.mark(words[1], 4 * len(values))
            copy, copy_moved = [], []
            for i, value in enumerate(values):
                if base is not None:
                    value, m = self.register_value(
                        base + (words[2] & 0x7FF) + i, value)
                    if m:
                        copy_moved.append(i)
                copy.append(value)
            words[1] = self.copy(copy, copy_moved)
            moved.append(1)
        elif op == 0x3C:
            if count == 6 and words[1] & 0x10:
                if self.place is None:
                    self.mark(words[2], 4)
                field(2, 'wait')
        elif op == INDIRECT_BUFFER:
            if count != 3 or depth:
                raise TraceError(f'packet {self.packet}: nested or odd '
                                 'indirect buffer')
            contents, contents_moved = [], []
            for inner in packets(self.memory.words(words[1] & 0x1FFFFFFC,
                                                   words[2] & 0xFFFFF)):
                prepared = self.prepare(inner, 1)
                if prepared is None:
                    continue
                if inner[0] >> 30 == 3 and \
                        inner[0] >> 8 & 127 in (0x22, 0x36) and self.resolve():
                    raise TraceError(f'packet {self.packet}: a resolve inside '
                                     'an indirect buffer')
                inner, inner_moved = prepared
                contents_moved += [len(contents) + i for i in inner_moved]
                contents += inner
            if not contents:
                return None
            words[1] = self.copy(contents, contents_moved)
            words[2] = len(contents)
            moved.append(1)
        elif op == 0x64:
            # Xenia's own packet for VdSwap: scanout's, which no stage reads.
            self.counts['swaps'] += 1
            return None
        elif op == 0x54:
            self.counts['interrupts'] += 1
            return [pm4.type3(NOP, count - 1)] + [0] * (count - 1), []
        elif op == 0x58:
            if count == 4:
                if self.place is None:
                    self.mark(words[2], 4)
                field(2, 'fence')
        elif op == 0x5A:
            if count != 3:
                raise TraceError(f'packet {self.packet}: EVENT_WRITE_EXT '
                                 f'of {count} words')
            if self.place is None:
                self.mark(words[2] & 0x1FFFFFFC, 12)
            field(2, 'screen extent')
        elif op == 0x5B:
            address = self.registers.get(RB_SAMPLE_COUNT_ADDR, 0)
            if self.place is None:
                self.mark(address & 0x1FFFFFE0, 32)
        elif op not in PLAIN_OPCODES:
            raise TraceError(f'packet {self.packet}: opcode {op:#x}')
        if op in (0x22, 0x36):
            self.check_fetches()
            target = self.resolve()
            if target and self.place is None:
                self.mark(*target)
        return words, moved

    def copy(self, words, moved):
        """Puts WORDS after the command area, its `moved` indices holding
        block offsets; their block offset."""
        # On a 128-byte line, as the command processor reads them: one
        # loading constants from elsewhere left it waiting.
        self.copies += [0] * (-len(self.copies) % LINE_WORDS)
        offset = self.copies_offset + 4 * len(self.copies)
        self.copy_relocations += [len(self.copies) + i for i in moved]
        self.copies += words
        return offset

    # Segments.

    def end_segment(self):
        words = len(self.commands) - self.segment_start
        if words or self.reads or self.pending_stages:
            self.segments.append((self.reads, self.segment_start, words,
                                  self.pending_stages))
        self.reads = []
        self.pending_stages = []
        # The command processor hangs on an indirect buffer that doesn't
        # start on a 128-byte line, as the one before ended (libxenon starts
        # each on one, libxenon:drivers/xenos/xe.c:123).
        self.commands += [TYPE2_NOP] * (-len(self.commands) % LINE_WORDS)
        self.segment_start = len(self.commands)

    def submit(self, words):
        header = words[0]
        prepared = self.prepare(words, 0)
        if prepared is not None:
            words, moved = prepared
            self.relocations += [len(self.commands) + i for i in moved]
            self.commands += words
        if header >> 30 == 3 and header >> 8 & 127 in (0x22, 0x36):
            target = self.resolve()
            if target and prepared is not None and self.place is not None:
                base, size = target
                self.pending_stages.append({
                    'packet': self.packet, 'command': self.command,
                    'base': base, 'size': size,
                    'control': self.registers.get(RB_COPY_CONTROL, 0),
                    'pitch': self.registers.get(RB_COPY_DEST_PITCH, 0),
                    'info': self.registers.get(RB_COPY_DEST_INFO, 0),
                    'storage_height': self.resolve_storage_height(),
                    'source': self.offset(base, 'resolve')})
                self.end_segment()
            self.command += 1

    def play(self, records):
        self.pending_stages = []
        for kind, base, payload in records():
            if kind == MEMORY_READ:
                self.memory.write(base, payload)
                if self.place is None:
                    self.mark(base, len(payload))
                    if not self.packet:
                        start = (base & 0x1FFFFFFF) + PAGE - 1 >> 12
                        end = (base & 0x1FFFFFFF) + len(payload) >> 12
                        self.preloaded.update(range(start, end))
                else:
                    if len(self.commands) > self.segment_start:
                        self.end_segment()
                    self.reads.append((self.offset(base, 'memory'),
                                       len(payload), self.memory_bytes))
                self.memory_bytes += len(payload)
            elif kind == PACKET_START:
                self.submit(payload)
                self.packet += 1
            elif kind == REGISTERS:
                # Written as runs of system registers (left out) or state
                # ones, the rest left out.
                run = []
                for reg, value in enumerate(payload, base):
                    kind = ('system' if system_register(reg) else 'state'
                            if any(first <= reg <= last for first, last in
                                   STATE_RANGES) else None)
                    if run and (kind != run_kind or len(run) == 0x4000 or
                                reg != run_first + len(run)):
                        self.submit([pm4.type0(run_first, len(run))] + run)
                        run = []
                    if kind is None:
                        self.counts['registers outside the state'] += 1
                        continue
                    if not run:
                        run_first, run_kind = reg, kind
                    run.append(value)
                if run:
                    self.submit([pm4.type0(run_first, len(run))] + run)
        self.end_segment()


def zero_runs(pages, place):
    """(block offset, bytes) runs covering `pages` at their block offsets,
    none longer than the zeros the memory file ends with."""
    runs = []
    for page in sorted(pages):
        block = place[page]
        if runs and runs[-1][0] + runs[-1][1] == block and \
                runs[-1][1] < NULL_BYTES:
            runs[-1][1] += PAGE
        else:
            runs.append([block, PAGE])
    return [tuple(run) for run in runs]


def place_pages(used, start):
    """{page: block offset} for the used pages, each run of contiguous ones
    at the same offset in its 64 KiB as in the trace, from block offset
    `start`; and the block offset past them."""
    place, cursor = {}, start
    pages = sorted(used)
    i = 0
    while i < len(pages):
        j = i
        while j + 1 < len(pages) and pages[j + 1] == pages[j] + 1:
            j += 1
        base = pages[i] << 12
        cursor += (base - cursor) % RUN_ALIGNMENT
        for k, page in enumerate(pages[i:j + 1]):
            place[page] = cursor + (k << 12)
        cursor += (j + 1 - i) << 12
        i = j + 1
    return place, cursor


# Stages.

def tiled2d(x, y, pitch_aligned, bpp_log2):
    """Xenia's texture_address::Tiled2D: the byte offset of pixel (x, y) of
    a tiled 2D surface, its pitch aligned to 32."""
    outer = ((y >> 5) * (pitch_aligned >> 5) + (x >> 5)) << 6
    inner = ((y >> 1) & 7) << 3 | (x & 7)
    oib = (outer | inner) << bpp_log2
    bank, pipe = (y >> 4) & 1, ((x >> 3) & 3) ^ (((y >> 3) & 1) << 1)
    return ((y & 1) << 4 | pipe << 6 | bank << 11 | (oib & 15) |
            ((oib >> 4) & 1) << 5 | ((oib >> 5) & 7) << 8 | (oib >> 8) << 12)


def tiled3d(x, y, z, pitch_aligned, height_aligned, bpp_log2):
    """texture_address::Tiled3D, including the slice-dependent bank xor."""
    outer = ((((z >> 2) * (height_aligned >> 4) + (y >> 4)) *
              (pitch_aligned >> 5)) + (x >> 5)) << 7
    inner = ((z & 3) << 5) | (((y >> 1) & 3) << 3) | (x & 7)
    oib = (outer | inner) << bpp_log2
    bank = ((y >> 3) ^ (z >> 2)) & 1
    pipe = ((x >> 3) & 3) ^ (bank << 1)
    return (((y & 1) << 4) | (pipe << 6) | (bank << 11) |
            (oib & 15) | (((oib >> 4) & 1) << 5) |
            (((oib >> 5) & 7) << 8) | ((oib >> 8) << 12))


def stage_pixels(stage):
    """(x, y, byte offset, bytes) of each pixel of a stage's destination."""
    pitch, height = stage['pitch'] & 0x3FFF, stage['pitch'] >> 16 & 0x3FFF
    bpp = COPY_BPP[stage['info'] >> 7 & 63]
    aligned = (pitch + 31) & ~31
    volume = bool(stage['info'] & 8)
    slice_index = (stage['info'] >> 4) & 7
    storage_height = (stage.get('storage_height', height) + 31) & ~31
    for y in range(height):
        for x in range(pitch):
            offset = (tiled3d(x, y, slice_index, aligned, storage_height,
                              bpp.bit_length() - 1) if volume else
                      tiled2d(x, y, aligned, bpp.bit_length() - 1))
            yield x, y, offset, bpp


def compare_stage(stage, expected, actual, low=None, high=None, mask=True):
    """The pixels of a stage whose bytes differ, the first of them, the
    pixels two replays starting from eDRAM all zeros and all ones (`low`,
    `high`) disagree on, as they show what eDRAM held before the trace, and
    how many of those are the same. With `mask` those are left out as
    undefined; without, for a replay starting from the console's own eDRAM,
    they are compared like the rest."""
    if expected == actual and low == high:
        return 0, None, 0, 0
    differing, first, undefined, same = 0, None, 0, 0
    # Whole stages compared once, not for every pixel.
    replays_differ = low != high
    for x, y, o, bpp in stage_pixels(stage):
        differs = expected[o:o + bpp] != actual[o:o + bpp]
        if replays_differ and low[o:o + bpp] != high[o:o + bpp]:
            undefined += 1
            same += not differs
            if mask:
                continue
        if differs:
            differing += 1
            if first is None:
                first = (x, y)
    return differing, first, undefined, same


def seeded_trace(trace, path, edram):
    """A copy of the trace starting from eDRAM `edram`: a byte it is filled
    with, or an EDRAM_SNAPSHOT's EDRAM_BYTES (edram_snapshot's)."""
    if isinstance(edram, int):
        edram = bytes([edram]) * EDRAM_BYTES
    if len(edram) != EDRAM_BYTES:
        raise TraceError(f'an eDRAM snapshot of {len(edram)} bytes')
    with open(trace, 'rb') as f:
        data = f.read()
    with open(path, 'wb') as f:
        f.write(data[:TRACE_HEADER.size])
        f.write(struct.pack('<III', EDRAM_SNAPSHOT, ENCODING_NONE,
                            EDRAM_BYTES))
        f.write(edram)
        f.write(data[TRACE_HEADER.size:])


def edram_offsets():
    """For each word of Xenia's eDRAM snapshot, tile-major (a tile's 80 x 16
    words row by row, 16 tiles a row of the 1280-wide surface), the byte
    offset of its texel in the dump."""
    tiles_across = EDRAM_PITCH // EDRAM_TILE_WIDTH
    out = array.array('I', bytes(EDRAM_BYTES))
    for y in range(EDRAM_ROWS):
        row = ((y // EDRAM_TILE_HEIGHT) * tiles_across * EDRAM_TILE_HEIGHT +
               y % EDRAM_TILE_HEIGHT) * EDRAM_TILE_WIDTH
        for x in range(EDRAM_PITCH):
            out[row + (x // EDRAM_TILE_WIDTH) * EDRAM_TILE_WIDTH *
                EDRAM_TILE_HEIGHT + x % EDRAM_TILE_WIDTH] = \
                gpugen.tiled_offset(x, y, EDRAM_PITCH, 4)
    return out


def edram_snapshot(dump):
    """The EDRAM_SNAPSHOT record's data, Xenia's eDRAM as little-endian
    words, from the dump: big-endian words of a tiled 1280 x 2048
    texture."""
    if len(dump) != EDRAM_BYTES:
        raise TraceError(f'an eDRAM dump of {len(dump)} bytes')
    words = array.array('I', dump)
    if sys.byteorder == 'little':
        words.byteswap()
    out = array.array('I', [words[o >> 2] for o in edram_offsets()])
    if sys.byteorder == 'big':
        out.byteswap()
    return out.tobytes()


def replay_seeded(directory, dump):
    """Whether DIRECTORY's play replay (run.py replay's) started from
    `dump`, the console's eDRAM dump, as its xenia_provenance.json says."""
    try:
        with open(os.path.join(directory, 'xenia_provenance.json')) as f:
            seed = json.load(f).get('edram_seed') or {}
    except OSError:
        return False
    return dump is not None and \
        seed.get('sha1') == hashlib.sha1(dump).hexdigest()


# The eDRAM dump.

def edram_resolve():
    """edramraw's read() of the whole eDRAM: copy command 1 to k_8_8_8_8,
    every register of the one-sample k_8_8_8_8 view at base 0 written
    first."""
    return gpugen.resolve_registers(
        EDRAM_PITCH, EDRAM_ROWS, edramraw.K_8888, 4, command=1,
        dest=(EDRAM_PITCH, EDRAM_ROWS),
        registers=edramraw.surface(0, pitch=EDRAM_PITCH, rows=EDRAM_ROWS))


def edram_vertices():
    """The Resolve's rectangle over the whole surface, as rbdraw's
    rectangles(): its edges half a pixel before the pixel centers."""
    return [gpugen.f2w(v - 0.5) for v in (0, 0, EDRAM_PITCH, 0, EDRAM_PITCH,
                                          EDRAM_ROWS)]


def register_state(s, start=0):
    """{register: (value, relocation shift or None)} as the stream's type 0
    packets from dword `start` leave them."""
    shifts = dict(s.relocations)
    state, i = {}, start
    for words in packets(s.words[start:]):
        if words[0] >> 30 == 0:
            first = words[0] & 0x7FFF
            for k, value in enumerate(words[1:]):
                reg = first + (0 if words[0] & 0x8000 else k)
                state[reg] = value, shifts.get(i + 1 + k)
        i += len(words)
    return state


def emit_edram_dump(s, dest, vertices):
    """The dump's segment: the Resolve of the whole eDRAM to block offset
    `dest`, its rectangle at block offset `vertices`; then every register
    it wrote as the setup left it, and the setup's shaders loaded again, as
    the setup loads them. Returns the registers it put back."""
    setup = register_state(s)
    start = len(s.words)
    s.reg('WAIT_UNTIL', 0x00020000)
    gpugen.emit_resolve(s, edram_resolve(), dest, vertices)
    written = sorted(set(register_state(s, start)) - ACTION_REGISTERS)
    missing = [reg for reg in written if reg not in setup]
    if missing:
        raise TraceError('the eDRAM dump writes registers the setup leaves '
                         'as the console had them: ' +
                         ', '.join(f'{reg:X}' for reg in missing))
    s.reg('WAIT_UNTIL', 0x00020000)
    i = 0
    while i < len(written):
        j = i
        while j + 1 < len(written) and written[j + 1] == written[j] + 1:
            j += 1
        s.words.append(pm4.type0(written[i], j + 1 - i))
        for reg in written[i:j + 1]:
            value, shift = setup[reg]
            if shift is not None:
                s.relocations.append((len(s.words), shift))
            s.words.append(value)
        i = j + 1
    s.packet3(pm4.INVALIDATE_STATE, [0x300])
    s.im_load_immediate(pm4.VERTEX, gpugen.gpr_dummy_vertex_shader().words())
    s.im_load_immediate(pm4.PIXEL, gpugen.trivial_pixel_shader().words())
    gpugen.emit_fence(s)
    return written


class RingCapacityError(TraceError):
    def __init__(self, required):
        self.required = required
        super().__init__(f'play ring requires {required} bytes')


def build(*args, **kwargs):
    # The executor embeds its physical layout while assembling. Grow the
    # power-of-two ring together with all following control locations, then
    # rebuild relocations; restore module defaults for other builds.
    names = ('RING_SIZE', 'RPTR_OFFSET', 'FENCE_OFFSET', 'DUMMY_OFFSET',
             'INPUT_OFFSET')
    saved = {name: getattr(gpugen, name) for name in names}
    try:
        try:
            return _build(*args, **kwargs)
        except RingCapacityError as error:
            size = 1 << error.required.bit_length()
            gpugen.RING_SIZE = size
            for name in names[1:]:
                setattr(gpugen, name, saved[name] + size - saved['RING_SIZE'])
            return _build(*args, **kwargs)
    finally:
        for name, value in saved.items():
            setattr(gpugen, name, value)


def _build(trace, outdir, image=None, results=gpugen.NAME, next_name=None,
          watchdog=None, reboot=False, edram=True, zero=True):
    os.makedirs(outdir, exist_ok=True)
    COMMAND_OFFSET = gpugen.INPUT_OFFSET
    title_id, records = read_trace(trace)
    scan = Player()
    scan.play(records)
    if not scan.used:
        raise TraceError(f'{trace}: no memory')
    # The command area and the copies after it, sized by a walk with every
    # page at 0: the placement changes no packet's size.
    estimate = Player({page: 0 for page in scan.used})
    estimate.play(records)
    copies_offset = COMMAND_OFFSET + (4 * len(estimate.commands) +
                                      4 * LINE_WORDS - 1 &
                                      -4 * LINE_WORDS)
    copies_end = copies_offset + 4 * len(estimate.copies)
    place, memory_end = place_pages(
        scan.used, (copies_end + RUN_ALIGNMENT - 1) & ~(RUN_ALIGNMENT - 1))
    player = Player(place, copies_offset, memory_end)
    player.play(records)
    if (len(player.commands), len(player.copies)) != (
            len(estimate.commands), len(estimate.copies)):
        raise TraceError('the command area changed size with the placement')

    # The zeros, then the outputs after a page for the results header, which
    # the dump's rectangle starts: the eDRAM before the trace, then the
    # stages.
    outputs_offset = memory_end + NULL_BYTES + PAGE
    vertices = outputs_offset - PAGE
    offset = outputs_offset + (EDRAM_BYTES if edram else 0)
    tests, stage_table = [], bytearray()
    # The setup reads the zeros from the end of the memory file, and them
    # into the pages no read fills whole before the first packet.
    zeros = [(memory_end, NULL_BYTES)]
    if zero:
        zeros += zero_runs(scan.used - scan.preloaded, place)
    read_table = bytearray()
    for block, size in zeros:
        read_table += struct.pack('>III', block, size, player.memory_bytes)
    stream = pm4.Stream()
    gpugen.emit_setup(stream)
    # The dump has no work.
    work = struct.pack('>IIII', 0, len(zeros), 0, 0)
    dump = None
    if edram:
        emit_edram_dump(stream, outputs_offset, vertices)
        work += struct.pack('>IIII', len(zeros), 0, 0, 0)
        dump = {'name': EDRAM_NAME, 'elements': 1, 'offset': 0,
                'element_bytes': EDRAM_BYTES, 'output_bytes': EDRAM_BYTES,
                'shader': [], 'segment': len(stream.segments) - 1,
                'pitch': EDRAM_PITCH, 'rows': EDRAM_ROWS}
    reads, stages = len(zeros), 0
    for n, (segment_reads, first, words, segment_stages) in enumerate(
            player.segments):
        if segment_reads:
            gpugen.emit_invalidate_fetch_caches(stream)
        if words:
            stream.words.append(pm4.type3(INDIRECT_BUFFER, 2))
            stream.relocations.append((len(stream.words), 0))
            stream.words += [COMMAND_OFFSET + 4 * first, words]
        gpugen.emit_fence(stream)
        for block, size, file_offset in segment_reads:
            read_table += struct.pack('>III', block, size, file_offset)
        for stage in segment_stages:
            name = f'stage{len(tests):03d}_command{stage["command"]}'
            stage_table += struct.pack('>III', stage['source'], offset,
                                       stage['size'] // 4)
            tests.append({'name': name, 'elements': 1,
                          'offset': offset - outputs_offset,
                          'element_bytes': stage['size'],
                          'output_bytes': stage['size'], 'shader': [],
                          'segment': len(stream.segments) - 1,
                          'stage': {k: stage[k] for k in (
                              'packet', 'command', 'base', 'size', 'control',
                              'pitch', 'info', 'storage_height')}})
            offset += stage['size'] + 15 & ~15
        work += struct.pack('>IIII', reads, len(segment_reads), stages,
                            len(segment_stages))
        reads += len(segment_reads)
        stages += len(segment_stages)
    gpugen.emit_put_back(stream, gpugen.QUERY_REGISTERS)
    # The put-back has no work.
    work += bytes(gpugen.PLAY_WORK_BYTES)
    output_words = (offset - outputs_offset) // 4
    block_size = (offset + 0xFFFF) & ~0xFFFF

    relocations = list(stream.relocations)
    relocations += [(COMMAND_OFFSET // 4 + i, 0) for i in player.relocations]
    relocations += [(copies_offset // 4 + i, 0)
                    for i in player.copy_relocations]
    blobs = [(gpugen.RING_OFFSET, stream.words),
             (COMMAND_OFFSET, player.commands),
             (copies_offset, player.copies)]
    if edram:
        blobs.append((vertices, edram_vertices()))
    if watchdog is None:
        # Each segment flushes its log four times.
        watchdog = (gpugen.WATCHDOG_SECONDS + len(stream.segments) // 2 +
                    (block_size + player.memory_bytes +
                     sum(size for _, size in zeros[1:])) //
                    gpugen.WATCHDOG_BYTES_PER_SECOND)
    # The memory file, named by its contents: a console that has it already
    # needn't get it again.
    digest = hashlib.sha1()
    with open(os.path.join(outdir, 'gpugen.mem'), 'wb') as f:
        for kind, base, payload in records():
            if kind == MEMORY_READ:
                f.write(payload)
                digest.update(payload)
        f.write(bytes(NULL_BYTES))
        digest.update(bytes(NULL_BYTES))
    memory = f'{digest.hexdigest()[:16]}.mem'
    if 4 * len(stream.words) >= gpugen.RING_SIZE:
        raise RingCapacityError(4 * len(stream.words))
    xex, console_ready, watchdog = gpugen.write_title(
        outdir, stream, blobs, relocations, block_size, outputs_offset,
        output_words, image, results, next_name, watchdog, reboot, title_id,
        (bytes(work), bytes(read_table), bytes(stage_table)), memory)
    xtr = os.path.join(outdir, 'gpugen.xtr')
    if os.path.exists(xtr):
        os.remove(xtr)
    try:
        os.link(trace, xtr)
    except OSError:
        shutil.copyfile(trace, xtr)
    meta = {'suite': os.path.splitext(os.path.basename(trace))[0],
            'title_id': title_id, 'version': gpugen.VERSION,
            'control': gpugen.CONTROL, 'segments': len(stream.segments),
            'raw': True, 'console_ready': console_ready, 'results': results,
            'next': next_name, 'reboot': reboot, 'watchdog': watchdog,
            'memory': memory, 'tests': tests,
            **({'edram': dump} if dump else {}),
            'play': {'packets': player.packet, 'commands': player.command,
                     'memory_bytes': player.memory_bytes,
                     'block_size': block_size, 'ring_size': gpugen.RING_SIZE,
                     **player.counts,
                     **({'zeroed_bytes': sum(size for _, size in zeros[1:])}
                        if zero else {})}}
    with open(os.path.join(outdir, 'gpugen.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    print(f'{len(tests)} stages over {player.packet} packets and '
          f'{player.command} commands, {len(stream.segments)} segments, '
          f'{xex:#x} bytes, block {block_size:#x}, memory '
          f'{player.memory_bytes:#x} bytes')


def add_arguments(parser):
    parser.add_argument('trace')
    parser.add_argument('outdir')
    parser.add_argument('--microcode', metavar='IMAGE',
                        help='as gpugen.py build\'s; needed to run on a '
                        'console')
    parser.add_argument('--results', default=gpugen.NAME, metavar='NAME',
                        help='write the results to NAME.bin')
    parser.add_argument('--next', metavar='NAME',
                        help='launch game:\\NAME.xex after a run that gave '
                        'the GPU back')
    parser.add_argument('--watchdog', type=int, metavar='SECONDS',
                        help='reboot the console this long after the start '
                        'unless the run has finished (default: from the '
                        'size)')
    parser.add_argument('--reboot', action='store_true',
                        help='reboot the console after the run instead of '
                        'going back to the dashboard')
    parser.add_argument('--no-edram', action='store_true',
                        help='don\'t read the eDRAM out before the trace')
    parser.add_argument('--no-zero', action='store_true',
                        help='leave the pages no read fills before the '
                        'trace as the console\'s memory had them')


def run_from_args(args):
    image = None
    if args.microcode:
        with open(args.microcode, 'rb') as f:
            image = f.read()
    try:
        build(args.trace, args.outdir, image, args.results, args.next,
              args.watchdog, args.reboot, not args.no_edram,
              not args.no_zero)
    except TraceError as e:
        sys.exit(f'play: {e}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    add_arguments(parser)
    run_from_args(parser.parse_args(argv))


if __name__ == '__main__':
    main()
