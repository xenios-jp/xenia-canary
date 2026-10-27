#!/usr/bin/env python3
"""Records what the Xenos GPU computes by running test draws in a title.

A suite is a list of tests drawn over the suite's input vertices. A test's
shader writes its results with memory export: the vertex shader, as points
without rasterization, or the pixel shader of a rasterized draw. A test can
instead draw into EDRAM and end with a Resolve, whose destination is its
result. `build` writes OUTDIR/gpugen.xex, a fixed executor with the suite's
command stream in its data pages, and OUTDIR/gpugen.json describing it. The
executor allocates a block of physical memory, brings up the command
processor through the kernel the way Direct3D does, copies the inputs,
textures and command stream into the block, adds its physical address to the
stream's relocations, and submits the stream one test (segment) at a time,
each closed by a fence it polls. Then it writes the results like xamgen does,
to cache:\\gpugen.bin or game:\\gpugen.bin.

    python3 tools/gpugen/gpugen.py build SUITE.py OUTDIR [--microcode IMAGE]
    python3 tools/gpugen/gpugen.py decode OUTDIR/gpugen.json RESULTS.bin

A suite file defines `build(suite)` and adds tests to it.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'xex'))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'xamgen'))
sys.path.insert(0, HERE)
import xexbuild  # noqa: E402
from ppcasm import Assembler  # noqa: E402
import xamgen  # noqa: E402
import pm4  # noqa: E402
import xenos  # noqa: E402

LOAD_ADDRESS = 0x82000000
PAGE_SIZE = xexbuild.PAGE_SIZE
CODE_BASE = LOAD_ADDRESS + PAGE_SIZE
IMPORTS_BASE = CODE_BASE + 0xC000
CONTROL = CODE_BASE + PAGE_SIZE
DATA = CONTROL + PAGE_SIZE
TITLE_ID = 0xFFFF7E5D

DONE_MAGIC = 0x600DF00D
EXIT_MAGIC = 0xE817E817
DATA_MAGIC = 0x47504744     # GPGD
RESULTS_MAGIC = 0x47504752  # GPGR
VERSION = 1
SENTINEL = 0xCDCDCDCD

# Physical block layout.
RING_OFFSET = 0
RING_SIZE = 0x40000
RPTR_OFFSET = 0x40000       # read pointer write-back
FENCE_OFFSET = 0x40100
DUMMY_OFFSET = 0x40200     # 8 zero dwords the GPR dummy draws fetch
INPUT_OFFSET = 0x50000

FENCE_POLLS = 2000          # 1 ms each
# A wedged GPU can't be handed back to XAM, and a title ending without
# handing it back freezes the console, so the title reboots it instead
# (HalReturnToFirmware's HalRebootRoutine). A watchdog thread reboots it too
# if the title hasn't finished WATCHDOG_SECONDS plus WATCHDOG_SEGMENT_SECONDS
# a segment (and a second per WATCHDOG_BYTES_PER_SECOND of block) after it
# started, or hasn't ended WATCHDOG_EXIT_SECONDS after it finished.
HAL_REBOOT_ROUTINE = 1
WATCHDOG_SECONDS = 60
WATCHDOG_SEGMENT_SECONDS = 3
WATCHDOG_BYTES_PER_SECOND = 4 << 20
WATCHDOG_EXIT_SECONDS = 60
WATCHDOG_STACK = 0x4000

MMIO_REGISTERS = 0x7FC80000
MMIO_CP_RB_WPTR = MMIO_REGISTERS + 4 * pm4.register('CP_RB_WPTR')
# VdInitializeEngines' first argument, an XDK build number << 16; the kernel
# ignores it.
ENGINES_VERSION = 0x50140000
# Direct3D's ME_INIT payload.
ME_INIT_PAYLOAD = [0x000003FF, 0, 0, 0, 0x00000080, 0x00000100, 0x00000180,
                   0x00000200, 0x00000280, 0x00000300, 0x00000380, 0x00010800,
                   0x00000007, 0, 0, 0, 0, 0]
# The command processor microcode Direct3D passes to VdInitializeEngines
# (SHA-1, dword count, SHA-1 of its first 16 bytes), the same in every
# Direct3D checked. It isn't distributed: `build --microcode` finds it in a
# decoded XEX image that links Direct3D. The kernel's own, loaded when the
# pointers are 0, doesn't run these streams on a console.
PFP_MICROCODE = ('224fefcc72b1ae72dcfc03bb8630ef9377f6b79a', 0x120,
                 '5383e1e102d495ade88d718461e7c1dafc92ffff')
ME_MICROCODE = ('e6264800ba80dd68cd09cd3bf137e37b4a26cf40', 0x900,
                'ee0fda871fbcb124b72a9284264299ce15ea3669')
PAGE_READWRITE = 0x04
# COHER_STATUS_HOST's vc_action_ena and tc_action_ena
# (xenia:src/xenia/gpu/registers.h).
COHER_VERTEX_CACHE, COHER_TEXTURE_CACHE = 1 << 24, 1 << 25
ALL_MEMORY_BYTES = 0x20000000
PAGE_WRITECOMBINE = 0x400
MEM_LARGE_PAGES = 0x20000000

# Every register the setup and the tests change that the title gives back as
# it found them before VdShutdownEngines hands the GPU back to XAM. Leaving
# the GPU without VdShutdownEngines froze a Peer Pressure console on the way
# back to Aurora.
PUT_BACK_REGISTERS = (
    'CP_PERFMON_CNTL', 'RBBM_PERFCOUNTER1_SELECT', 'SCLK_PWRMGT_CNTL2_REG',
    'SCLK_PWRMGT_CNTL3_REG', 'PA_CL_ENHANCE', 'SQ_GPR_MANAGEMENT',
    'SQ_FLOW_CNTL', 'SQ_INST_STORE_MANAGEMENT', 'SQ_EO_RT', 'VC_FIFO_DEPTHS',
    'RB_BC_CONTROL', 'RB_SURFACE_INFO', 'RB_COLOR_INFO', 'RB_DEPTH_INFO',
    'COHER_DEST_BASE_7', 'PA_SC_SCREEN_SCISSOR_TL', 'PA_SC_SCREEN_SCISSOR_BR',
    'PA_SC_WINDOW_OFFSET', 'PA_SC_WINDOW_SCISSOR_TL',
    'PA_SC_WINDOW_SCISSOR_BR', 'VGT_MAX_VTX_INDX', 'VGT_MIN_VTX_INDX',
    'VGT_INDX_OFFSET', 'VGT_OUTPUT_PATH_CNTL', 'VGT_HOS_CNTL',
    'VGT_HOS_MAX_TESS_LEVEL', 'VGT_HOS_MIN_TESS_LEVEL',
    'VGT_HOS_REUSE_DEPTH', 'VGT_GROUP_PRIM_TYPE', 'VGT_GROUP_FIRST_DECR',
    'VGT_GROUP_DECR', 'VGT_GROUP_VECT_0_CNTL', 'VGT_GROUP_VECT_1_CNTL',
    'VGT_GROUP_VECT_0_FMT_CNTL', 'VGT_GROUP_VECT_1_FMT_CNTL',
    'VGT_DMA_BASE', 'VGT_DMA_SIZE', 'RB_COLOR_MASK', 'PA_CL_VPORT_XSCALE',
    'PA_CL_VPORT_XOFFSET', 'PA_CL_VPORT_YSCALE', 'PA_CL_VPORT_YOFFSET',
    'PA_CL_VPORT_ZSCALE', 'PA_CL_VPORT_ZOFFSET', 'SQ_PROGRAM_CNTL',
    'SQ_CONTEXT_MISC', 'SQ_INTERPOLATOR_CNTL', 'RB_DEPTHCONTROL', 'RB_HIZCONTROL',
    'PA_CL_CLIP_CNTL',
    'PA_SU_SC_MODE_CNTL', 'PA_CL_VTE_CNTL', 'RB_MODECONTROL',
    'PA_SU_POINT_SIZE', 'PA_SU_POINT_MINMAX', 'PA_SU_VTX_CNTL', 'SQ_VS_CONST',
    'SQ_PS_CONST', 'PA_SC_AA_CONFIG', 'PA_SC_AA_MASK', 'RB_COPY_CONTROL',
    'RB_COPY_DEST_BASE', 'RB_COPY_DEST_PITCH', 'RB_COPY_DEST_INFO',
    'RB_HIZ_CLEAR', 'RB_DEPTH_CLEAR', 'RB_COLOR_CLEAR', 'RB_COLOR_CLEAR_LO')

# Put back too after scratch writebacks.
SCRATCH_REGISTERS = ('SCRATCH_REG0', 'SCRATCH_ADDR', 'SCRATCH_UMSK')
# And after occlusion and VIZ queries.
QUERY_REGISTERS = ('RB_SAMPLE_COUNT_CTL', 'RB_SAMPLE_COUNT_ADDR',
                   'PA_SC_VIZ_QUERY')

# Control page layout.
C_DONE = 0x000
C_EXIT = 0x004
C_PROGRESS = 0x008
C_FILE_STATUS = 0x00C
C_FIRST_SLEEP = 0x010
C_POLL_SLEEP = 0x018
C_EXIT_SLEEP = 0x020
C_HANDLE = 0x028
C_IOSB = 0x030
C_OFFSET = 0x038
C_ATTRIBUTES = 0x040
C_CACHE_NAME = 0x050
C_GAME_NAME = 0x058
C_CACHE_PATH = 0x060
C_GAME_PATH = 0x080
C_NEXT_PATH = 0x0C0
C_BLOCK = 0x100
C_BLOCK_PHYSICAL = 0x104
C_ENGINES_IDLE = 0x108
C_WEDGED = 0x10C
C_WATCHDOG_HANDLE = 0x110
C_WATCHDOG_ID = 0x114
C_ELAPSED = 0x118           # seconds since the title started
C_DEADLINE = 0x11C          # the watchdog reboots at this many
C_WATCHDOG_SLEEP = 0x120
C_REBOOT = 0x128            # reboot at the end instead of exiting
# A trace's play title (`play.py`) reads the trace's memory from a file.
C_PLAY_HANDLE = 0x130
C_PLAY_IOSB = 0x138
C_PLAY_OFFSET = 0x140
C_PLAY_NAME = 0x148
C_PLAY_PATH = 0x150
# And it logs how far it got to a file of its own, flushed at each step of
# each segment, which outlasts a run that freezes the whole console.
C_PLAY_LOG_HANDLE = 0x190
C_PLAY_LOG_IOSB = 0x198
C_PLAY_LOG_OFFSET = 0x1A0
C_PLAY_LOG_WORDS = 0x1A8     # segment, step
C_PLAY_LOG_NAME = 0x1B0
C_PLAY_LOG_PATH = 0x1B8
C_PLAY_END = 0x1F8
# And a thread of its own answers a host's UDP datagram to STATUS_PORT with
# STATUS_MAGIC, the seconds since the start, the progress, the log's
# segment and step and its last receive error: a host sees the run as it goes, and at once when the
# console stops answering.
C_STATUS_HANDLE = 0x200
C_STATUS_ID = 0x204
C_NET_ONE = 0x208
C_NET_FROM_BYTES = 0x20C
C_NET_ADDRESS = 0x210        # sockaddr_in: the port bound
C_NET_FROM = 0x220           # sockaddr_in: the host's
C_NET_REPLY = 0x230
C_NET_PARAMS = 0x248         # XNetStartup's parameters
C_NET_REQUEST = 0x258
C_NET_ERROR = 0x268          # recvfrom's last error
STATUS_PORT = 7312
STATUS_MAGIC = 0x4750474E    # 'GPGN'
STATUS_REPLY_BYTES = 24
C_FORMAT = 0x400
# A play title's copy records, after its data header's play words; and the
# bytes of a segment's work entry.
PLAY_RECORDS = 0x80
PLAY_WORK_BYTES = 16

NAME = 'gpugen'
FORMAT = b'gpugen: %d segments done, last %d polls, results file %08X\n\0'

# Progress values in the order the executor reaches them, so a fault names
# the stage: PROGRESS[value - 1].
PROGRESS = ('allocate', 'engines', 'copy', 'submit', 'results')
P_ALLOCATE, P_ENGINES, P_COPY, P_SUBMIT, P_RESULTS = range(1, 6)

FLOAT_CONSTANTS = pm4.register('SHADER_CONSTANT_000_X')
FETCH_CONSTANTS = pm4.register('SHADER_CONSTANT_FETCH_00_0')
FLUSH_FETCH = pm4.register('SHADER_CONSTANT_FLUSH_FETCH_0')


def f2w(value):
    return struct.unpack('>I', struct.pack('>f', value))[0]


# Suite.

# Memory export formats: (ColorFormat, SurfaceNumberFormat, bytes).
EXPORT_FLOAT4 = (38, 7, 16)


class Test:
    def __init__(self, name, shader, constants, elements, export,
                 fetch_constants, stream=None, pixel_shader=None,
                 program_cntl=None, registers=(), draw=None, resolve=None,
                 passes=(), after_passes=(), clear=None, scratch=(),
                 indices=None, set_constants=(), zpd=None, viz_queries=None,
                 viz=None, zpd_control=(2, 2), zpd_reset=False):
        self.name = name
        self.shader = shader
        self.constants = constants  # first vec4 index -> words
        self.elements = elements    # exported elements per vertex
        self.export = export        # (color format, number format, bytes)
        # Fetch constant slot (1-31) -> (6 words, texture name or None); the
        # texture's block offset is added to word 1 (its base address field).
        self.fetch_constants = fetch_constants
        # Called with each export stream constant's words to change them.
        self.stream = stream
        # A pixel shader that does the memory export instead of the vertex
        # shader: its stream constants go to the pixel bank (c256 + i), the
        # same local indices (c0, c2 + i, c7).
        self.pixel_shader = pixel_shader
        self.program_cntl = program_cntl
        # (register, values) after the default draw state.
        self.registers = registers
        # (primitive, first vertex, count) instead of all vertices as points.
        self.draw = draw
        # A Resolve after the draw writing the output instead of memory
        # exports (see resolve_registers), of the rectangle in the suite's
        # first 6 input dwords.
        self.resolve = resolve
        # Draws before the test's own, each [(register, values)], fetch
        # constants and optionally (primitive, first vertex, count) instead of
        # the test's, then `after_passes` registers for the test's draw.
        self.passes = passes
        self.after_passes = after_passes
        # A clearing Resolve (resolve_registers with `clear`) before
        # everything else, under `registers`, to the output the test's own
        # Resolve then overwrites.
        self.clear = clear
        # Instead of drawing, (byte offset, SCRATCH_UMSK, value) writes of
        # SCRATCH_REG0, with SCRATCH_ADDR at the offset into the output.
        self.scratch = scratch
        # (texture name, pm4.INDEX_INT16 or INDEX_INT32, endianness): the
        # draw's vertices come from that texture's indices, `draw`'s first
        # vertex becoming VGT_INDX_OFFSET and its count the indices'.
        self.indices = indices
        # (pm4.CONSTANT_* bank, offset, words) SET_CONSTANTs after
        # `registers`: bool and loop constants.
        self.set_constants = set_constants
        # Instead of exports, Direct3D's occlusion query (emit_zpd): the
        # sample counts at the output + 32 before pass `zpd` (the test's own
        # draw if there are no more), at the output after the test's draw.
        self.zpd = zpd
        # RB_SAMPLE_COUNT_CTL of the reports at the start and end (Direct3D's
        # 2 both). Every report must copy (bit 1): after one that doesn't,
        # each report lags a draw, and the dashboard a run with them went
        # back to couldn't launch titles.
        if any(not control & 2 for control in zpd_control):
            raise ValueError(f'{name}: a report that does not copy')
        self.zpd_control = zpd_control
        # Reset the counters first, with a copy to the output the end
        # report then overwrites.
        self.zpd_reset = zpd_reset
        # {pass index: VIZ query ID 0-63}: those passes' draws in between a
        # VIZ_QUERY beginning and ending that query.
        self.viz_queries = viz_queries or {}
        # The VIZ token of the test's own draw.
        self.viz = viz

    @property
    def element_bytes(self):
        return self.export[2]

    def output_bytes(self, slots):
        if self.zpd is not None:
            return 64
        if self.scratch:
            return (max(offset for offset, _, _ in self.scratch) + 32) & ~31
        if self.resolve:
            return self.resolve['bytes']
        return self.element_bytes * slots * self.elements


class Suite:
    """Tests drawn over the same input vertices, `vertex_dwords` per vertex."""

    def __init__(self, name):
        self.name = name
        self.title_id = TITLE_ID
        self.tests = []
        self.inputs = []
        self.vertex_dwords = 4
        # Vertices of a suite whose shaders compute their inputs from r0.x.
        self.generated_vertices = 0
        # Elements' slots per test when not one per vertex (pixel shaders).
        self.export_slots = 0
        # Results as raw big-endian words per test instead of hex in JSON.
        self.raw = False
        self.textures = {}  # name -> words, each placed 4 KiB aligned
        # 8 VS registers, vertex fetch, multipass export (no rasterization,
        # as Direct3D's memexport pass) and its PS export mode.
        self.program_cntl = 7 | (1 << 16) | (7 << 24) | (2 << 27)

    def set_inputs(self, words, vertex_dwords):
        if len(words) % vertex_dwords:
            raise ValueError('inputs are not whole vertices')
        self.inputs = [w & 0xFFFFFFFF for w in words]
        self.vertex_dwords = vertex_dwords

    def set_generated(self, vertices):
        """No input buffer: `vertices` points whose shaders compute their
        inputs from the vertex index."""
        self.inputs = []
        self.generated_vertices = vertices

    @property
    def vertices(self):
        if self.generated_vertices:
            return self.generated_vertices
        return len(self.inputs) // self.vertex_dwords

    @property
    def slots(self):
        return self.export_slots or self.vertices

    def test(self, name, shader, constants=None, elements=1,
             export=EXPORT_FLOAT4, fetch_constants=None, stream=None,
             pixel_shader=None, program_cntl=None, registers=(), draw=None,
             resolve=None, passes=(), after_passes=(), clear=None,
             scratch=(), indices=None, set_constants=(), zpd=None,
             viz_queries=None, viz=None, zpd_control=(2, 2),
             zpd_reset=False):
        if any(test.name == name for test in self.tests):
            raise ValueError(f'two tests named {name!r}')
        if clear and (not resolve or clear['bytes'] > resolve['bytes']):
            raise ValueError(
                f'{name}: a clear needs a Resolve at least as big')
        self.tests.append(Test(name, shader, constants or {}, elements, export,
                               fetch_constants or {}, stream, pixel_shader,
                               program_cntl, registers, draw, resolve, passes,
                               after_passes, clear, scratch, indices,
                               set_constants, zpd, viz_queries, viz,
                               zpd_control, zpd_reset))

    def add_texture(self, name, words):
        self.textures[name] = [w & 0xFFFFFFFF for w in words]


# Shader building blocks the suites share. A test's vertex shader fetches its
# vertex into registers, computes, and exports `elements` float4s per vertex.

# Constants every test's draw sets: c0 = (0, elements, 0, 0) for eA, c1 = the
# culled position, c2 + i = the export stream of element i (the same buffer
# 16 bytes further on per element), c7 = zeros.
C_EA = 0
C_POSITION = 1
C_STREAM = 2
MAX_ELEMENTS = 5
C_FLUSH = 7     # zeros, for the trailing eA-only exports
C_FIRST_FREE = 8

VERTEX_FETCH_CONSTANT = 0
INPUT_REGISTER = 1


def fetch_inputs(vertex_dwords):
    """vfetches of the vertex into r1, r2, ... one float4 each: a full fetch
    prefetching up to 8 dwords, then mini fetches within them. A mini fetch
    past the prefetched dwords doesn't read the vertex on the console, so
    every 8 dwords start another full fetch."""
    fetches = []
    for i in range(vertex_dwords // 4):
        full = i % 2 == 0
        # Direct3D leaves the stride of mini fetches 0.
        fetches.append(xenos.vfetch(
            INPUT_REGISTER + i, 0, 'x', VERTEX_FETCH_CONSTANT, 0,
            xenos.VF_32_32_32_32_FLOAT, vertex_dwords if full else 0,
            offset=4 * i, mini=not full,
            prefetch=min(vertex_dwords - 4 * i, 8) if full else None))
    return fetches


def memexport_shader(vertex_dwords, body, exports, fetches=None,
                     prologue=(), flow=None):
    """A vertex shader: fetch, `body` ALU instructions, then one export per
    entry of `exports`, a list of ALU instructions writing eM0 (each export
    gets its own eA at element vertex * len(exports) + i), and a culled
    position. `fetches` replaces the default fetch of the vertex, `prologue`
    ALU instructions run before it, `flow` is called with the shader to add
    control flow after `body`. The tail is the one of Direct3D titles'
    memexport shaders: two more exports that only write eA from constants,
    which keep the data exports from being overwritten by what follows, then
    the position, an unwritten interpolator allocation and an empty exece."""
    s = xenos.Shader()
    if prologue:
        s.exec_(list(prologue))
    if fetches is None:
        fetches = fetch_inputs(vertex_dwords)
    if fetches:
        s.exec_(fetches)
    first = list(body)
    if first:
        first[0] = first[0].serialized()
        s.exec_(first)
    if flow:
        flow(s)
    emit_exports(s, exports, xenos.r(0, 'x'), not first)
    s.alloc(xenos.ALLOC_POSITION, 0)
    s.exec_([xenos.alu(vector_op=xenos.V_MAX,
                       vector_dest=xenos.EXPORT_POSITION, vector_mask=0xF,
                       export=True, src1=xenos.c(C_POSITION),
                       src2=xenos.c(C_POSITION))])
    s.alloc(xenos.ALLOC_INTERPOLATORS, 0)
    s.exec_([], end=True)
    return s


def export_copy(register):
    """max eM0, r`register`, r`register`: the register as the exported
    element."""
    return xenos.alu(vector_op=xenos.V_MAX, vector_dest=xenos.EXPORT_DATA0,
                     vector_mask=0xF, export=True, src1=xenos.r(register),
                     src2=xenos.r(register))


def export_vector(op, *sources):
    """Vector `op` on `sources` as the exported element; a one-source op
    reads it twice."""
    return [xenos.alu(vector_op=op, vector_dest=xenos.EXPORT_DATA0,
                      vector_mask=0xF, export=True, src1=sources[0],
                      src2=sources[1] if len(sources) > 1 else sources[0],
                      src3=sources[2] if len(sources) > 2 else None)]


def export_scalar_lanes(op, operands):
    """Lane i of the exported element = scalar `op` on operands(i): a source
    for src3, or the alu() keywords."""
    lanes = []
    for i in range(4):
        keywords = operands(i)
        if not isinstance(keywords, dict):
            keywords = {'src3': keywords}
        lanes.append(xenos.alu(vector_dest=xenos.EXPORT_DATA0, export=True,
                               scalar_op=op, scalar_mask=1 << i, **keywords))
    return lanes


def lane_operands(register):
    """export_scalar_lanes operands of a one-operand op: lane i of
    r`register` in src3's W slot, the one such ops read."""
    return lambda i: xenos.r(register, 'xxx' + 'xyzw'[i])


def emit_exports(s, exports, index, serialize):
    """The export clauses of `exports` (see memexport_shader) with element
    `index` (a source) * len(exports) + i, then the two eA-only exports."""
    if len(exports) > MAX_ELEMENTS:
        raise ValueError('too many exported elements')
    for i, writes in enumerate(exports):
        s.alloc(xenos.ALLOC_MEMORY, 1)
        # eA = index * (0, elements, 0, 0) + stream i.
        ea = xenos.alu(vector_op=xenos.V_MAD, vector_dest=xenos.EXPORT_ADDRESS,
                       vector_mask=0xF, export=True, src1=index,
                       src2=xenos.c(C_EA, 'xyxx'), src3=xenos.c(C_STREAM + i))
        if serialize and i == 0:
            ea.serialize = True
        s.exec_([ea] + list(writes))
    for components in ('yxwz', 'zyxw'):
        s.alloc(xenos.ALLOC_MEMORY, 1)
        s.exec_([xenos.alu(vector_op=xenos.V_MAD,
                           vector_dest=xenos.EXPORT_ADDRESS, vector_mask=0xF,
                           export=True, src1=xenos.c(C_FLUSH, components),
                           src2=xenos.c(C_FLUSH, components),
                           src3=xenos.c(C_FLUSH, components))])


def memexport_pixel_shader(body, exports, index, fetches=()):
    """A pixel shader: `fetches`, `body` ALU instructions, then `exports` as
    in memexport_shader at element `index` (a source), then oC0 = r0. Its
    constants are the pixel bank's, c256 + the same indices."""
    s = xenos.Shader()
    if fetches:
        s.exec_(list(fetches))
    first = list(body)
    if first:
        first[0] = first[0].serialized()
        s.exec_(first)
    emit_exports(s, exports, index, not first)
    s.alloc(xenos.ALLOC_COLORS, 0)
    s.exec_([xenos.alu(vector_op=xenos.V_MAX, vector_dest=0, vector_mask=0xF,
                       export=True, src1=xenos.r(0), src2=xenos.r(0))],
            end=True)
    return s


def interpolating_vertex_shader(vertex_dwords):
    """Fetches the vertex into r1, r2, ... and exports oPos = r1 and
    interpolator i = r2 + i, shaped like gpr_dummy_vertex_shader."""
    s = xenos.Shader()
    s.exec_(fetch_inputs(vertex_dwords))
    s.alloc(xenos.ALLOC_INTERPOLATORS, 0)
    s.exec_([xenos.alu(vector_op=xenos.V_MAX, vector_dest=i, vector_mask=0xF,
                       export=True, src1=xenos.r(INPUT_REGISTER + 1 + i),
                       src2=xenos.r(INPUT_REGISTER + 1 + i))
             for i in range(vertex_dwords // 4 - 1)])
    s.alloc(xenos.ALLOC_POSITION, 0)
    s.exec_([xenos.alu(vector_op=xenos.V_MAX,
                       vector_dest=xenos.EXPORT_POSITION, vector_mask=0xF,
                       export=True, src1=xenos.r(INPUT_REGISTER),
                       src2=xenos.r(INPUT_REGISTER))])
    s.exec_([xenos.alu(vector_op=xenos.V_MAX)], end=True)
    return s


def gpr_dummy_vertex_shader():
    """The vertex shader of the dummy draws Direct3D follows GPR changes with:
    a float3 and a float4 from 7 dwords into o0 and oPos."""
    s = xenos.Shader()
    s.exec_([xenos.vfetch(1, 0, 'x', 0, 0, xenos.VF_32_32_32_FLOAT, 7,
                          dst_swizzle='xyz1', prefetch=7),
             xenos.vfetch(0, 0, 'x', 0, 0, xenos.VF_32_32_32_32_FLOAT, 0,
                          offset=3, mini=True)])
    s.alloc(xenos.ALLOC_INTERPOLATORS, 0)
    s.exec_([xenos.alu(vector_op=xenos.V_MAX, vector_dest=0, vector_mask=0xF,
                       export=True, src1=xenos.r(0), src2=xenos.r(0))])
    s.alloc(xenos.ALLOC_POSITION, 0)
    s.exec_([xenos.alu(vector_op=xenos.V_MAX,
                       vector_dest=xenos.EXPORT_POSITION, vector_mask=0xF,
                       export=True, src1=xenos.r(1), src2=xenos.r(1))])
    s.exec_([xenos.alu(vector_op=xenos.V_MAX)], end=True)
    return s


def resolve_vertex_shader():
    """The vertex shader of Direct3D's Resolve: xy from vertex fetch constant
    0, k_32_32_FLOAT, z = w = 1, into oPos."""
    s = xenos.Shader()
    s.exec_([xenos.vfetch(0, 0, 'x', 0, 0, xenos.VF_32_32_FLOAT, 2,
                          dst_swizzle='xy11', prefetch=1)])
    s.alloc(xenos.ALLOC_INTERPOLATORS, 0)
    s.alloc(xenos.ALLOC_POSITION, 0)
    s.exec_([xenos.alu(vector_op=xenos.V_MAX,
                       vector_dest=xenos.EXPORT_POSITION, vector_mask=0xF,
                       export=True, src1=xenos.r(0), src2=xenos.r(0))],
            end=True)
    return s


def trivial_pixel_shader():
    """alloc colors; exece max oC0, r0, r0."""
    s = xenos.Shader()
    s.alloc(xenos.ALLOC_COLORS, 0)
    s.exec_([xenos.alu(vector_op=xenos.V_MAX, vector_dest=0, vector_mask=0xF,
                       export=True, src1=xenos.r(0), src2=xenos.r(0))],
            end=True)
    return s


# Render state.

SCREEN_XY = 2  # SQ_CONTEXT_MISC.sc_output_screen_xy
# PA_SU_SC_MODE_CNTL: vtx_window_offset_enable, and msaa_enable.
MODE_CNTL = 1 << 16
MSAA_ENABLE = 1 << 15
AA_CONFIG = (0, 0x8001, 0xC003)  # Direct3D's PA_SC_AA_CONFIG at 1x, 2x, 4x


# Textures.

SWIZZLE_XYZW = 0x688  # texture fetch constant swizzle: each from itself
DIMENSION_2D = 1 << 9  # in dword 5


def texture_constant(fmt, width, height=1, sign=0, num_format=0,
                     endian=pm4.ENDIAN_8IN32, filters=(0, 0, 0), clamp=2,
                     border_color=0, swizzle=SWIZZLE_XYZW, exp_adjust=0,
                     pitch=None, aniso=0, mips=(0, 0), lod_bias=0,
                     grad_exp_adjust=(0, 0), tri_clamp=0, aniso_bias=0,
                     packed_mips=False):
    """A 2D texture fetch constant (xenos.h xe_gpu_texture_fetch_t) for a
    linear texture, the base address left to the test's texture: `sign` for
    every component, `filters` (mag, min, mip), `clamp` one ClampMode or
    (x, y, z), `pitch` in texels, by default `width` rounded up to 32. A
    texture with mips gives the (min, max) levels; its mip address is left
    to the test's mip texture (see Test.fetch_constants). `lod_bias` is in
    1/32ths."""
    clamp = clamp if isinstance(clamp, tuple) else (clamp,) * 3
    pitch = (pitch or max(width, 32) + 31) // 32
    return [2 | ((sign * 0x55) << 2) | (clamp[0] << 10) | (clamp[1] << 13) |
            (clamp[2] << 16) | (pitch << 22),
            fmt | (endian << 6),
            (width - 1) | ((height - 1) << 13),
            num_format | (swizzle << 1) | ((exp_adjust & 0x3F) << 13) |
            (filters[0] << 19) | (filters[1] << 21) | (filters[2] << 23) |
            (aniso << 25),
            (mips[0] << 2) | (mips[1] << 6) | ((lod_bias & 0x3FF) << 12) |
            ((grad_exp_adjust[0] & 0x1F) << 22) |
            ((grad_exp_adjust[1] & 0x1F) << 27),
            border_color | (tri_clamp << 3) | ((aniso_bias & 0xF) << 5) |
            DIMENSION_2D | (int(packed_mips) << 11)]


def texture_words(*rows, swap=False):
    """A linear texture's words from rows of bytes, each padded to the 256
    bytes a row takes. With `swap` the rows hold texels in the byte order an
    8in32 fetch reads them, so each dword is stored reversed."""
    data = b''
    for row in rows:
        if swap:
            row += bytes(-len(row) % 4)
            row = b''.join(row[i:i + 4][::-1] for i in range(0, len(row), 4))
        data += row + bytes(-len(row) % 256)
    return list(struct.unpack(f'>{len(data) // 4}I', data))


def stream_constant(offset, count, endian=2, color_format=38, num_format=7):
    """Memory export stream constant words for `count` elements at block
    offset `offset` (relocated by the executor), 32_32_32_32_FLOAT 8in32."""
    return [0x40000000 | (offset >> 2), 0x4B000000,
            (0x4B0 << 20) | (num_format << 16) | (color_format << 8) | endian,
            0x4B000000 | count]


# Command stream.

# Direct3D's default render state after init, register -> value; every
# register of each block not listed is 0.
DEFAULT_STATE_BLOCKS = [(0x2200, 0x220B), (0x2180, 0x2184), (0x2100, 0x2114),
                        (0x2000, 0x200F), (0x2280, 0x2294), (0x2300, 0x2325)]
DEFAULT_STATE = {
    0x2200: 0x00700734, 0x2201: 0x00010001, 0x2202: 0x87000007,
    0x2204: 0x00080000, 0x2205: 0x00018006, 0x2206: 0x0000043F,
    0x2208: 0x00000004, 0x2209: 0x00010001, 0x220A: 0x00010001,
    0x220B: 0x00010001, 0x2100: 0x00FFFFFF, 0x2103: 0x0000FFFF,
    0x210C: 0x00FFFF00, 0x210D: 0x00FFFF00, 0x200F: 0x20002000,
    0x2280: 0x00080008, 0x2281: 0x04000010, 0x2282: 0x00000008,
    0x2285: 0x00000001, 0x2286: 0x3F800000, 0x2287: 0x3F800000,
    0x2288: 0x0000000E, 0x2302: 0x00000004, 0x2303: 0x40000000,
    0x2304: 0x3F800000, 0x2305: 0x40000000, 0x2306: 0x3F800000,
    0x2307: 0x000FF000, 0x2308: 0x000FF100, 0x2312: 0x0000FFFF,
    0x2316: 0x0000000E, 0x2317: 0x00000010, 0x2324: 0x00000002,
}


def emit_init(s):
    """What Direct3D writes after VdInitializeEngines: ME_INIT, setup
    registers, its GPR split and the default render state."""
    s.packet3(pm4.ME_INIT, ME_INIT_PAYLOAD)
    s.reg('SQ_INST_STORE_MANAGEMENT', 0x00010800)
    s.reg('PG_START', 0xC0100000, 0x07F00000, 0xC0000000, 0x00100000)
    s.reg('WAIT_UNTIL', 0x00020000)
    s.reg('SQ_EO_RT', 0)
    s.reg('SQ_FLOW_CNTL', 0x04000000)
    s.packet3(pm4.REG_RMW, [pm4.register('SCLK_PWRMGT_CNTL2_REG'),
                            0xFFFFFFFF, 0x80010000])
    s.packet3(pm4.REG_RMW, [pm4.register('SCLK_PWRMGT_CNTL3_REG'),
                            0xFFFFFFFF, 0])
    s.reg('VC_FIFO_DEPTHS', 0x1F60)
    s.reg('PA_CL_ENHANCE', 3)
    s.reg('PA_SC_WINDOW_OFFSET', 0, 0, 0)
    s.reg('WAIT_UNTIL', 0x00020000)
    s.reg('RB_BC_CONTROL', 0x200E)
    s.reg('WAIT_UNTIL', 0x00020000)
    for _ in range(8):
        s.reg('SQ_GPR_MANAGEMENT', 0x00040401)
    s.reg('SQ_GPR_MANAGEMENT', 0x00040400)
    s.reg('RB_HIZCONTROL', 0)
    for first, last in DEFAULT_STATE_BLOCKS:
        s.reg(first,
              *[DEFAULT_STATE.get(r, 0) for r in range(first, last + 1)])
    emit_invalidate_fetch_caches(s)


def emit_invalidate_fetch_caches(s):
    """The vertex and texture caches invalidated over all memory, so no
    line from an earlier title at a suite's addresses survives: a chained
    suite's vertex fetches read the one before's inputs without it. Then a
    wait for the coherency to finish."""
    s.reg('COHER_STATUS_HOST', COHER_VERTEX_CACHE | COHER_TEXTURE_CACHE)
    s.reg('COHER_SIZE_HOST', ALL_MEMORY_BYTES, 0)
    s.packet3(pm4.WAIT_REG_MEM,
              [3, pm4.register('COHER_STATUS_HOST'), 0, 0x80000000, 8])


def emit_draw_state(s, program_cntl):
    """The per-draw state of Direct3D's internal point draws, plus a surface
    pitch Xenia needs to run the draw at all."""
    s.reg('SQ_PROGRAM_CNTL', program_cntl)
    s.reg('SQ_CONTEXT_MISC', 0)
    s.reg('VGT_MAX_VTX_INDX', 0xFFFFFF, 0, 0)
    s.reg('PA_CL_CLIP_CNTL', 0x10000, 0x10000, 0x300)
    s.reg('PA_SC_AA_MASK', 0xFFFF)
    s.reg('COHER_DEST_BASE_7', 0)
    s.packet3(pm4.SET_BIN_MASK_LO, [0xFFFFFFFF])
    s.reg('RB_DEPTHCONTROL', 0)
    s.reg('RB_HIZCONTROL', 0)
    s.reg('RB_MODECONTROL', 4)
    s.reg('RB_COLOR_MASK', 0)
    s.reg('PA_SU_POINT_SIZE', 0x00080008)
    s.reg('PA_SU_VTX_CNTL', 4)
    s.reg('PA_SC_WINDOW_OFFSET', 0, 0, 0x00100010)
    s.reg('RB_SURFACE_INFO', 1280)


TYPE2_PAD = 0x80000000


def pad_to(s, modulus, remainder):
    """Type 2 filler until the next dword's index is `remainder` mod
    `modulus` (the ring is 64 KiB aligned, so the index sets the alignment)."""
    while len(s.words) % modulus != remainder:
        s.words.append(TYPE2_PAD)


def emit_fetch_constant(s, offset, dwords):
    """Vertex fetch constant 0 over `dwords` at block offset `offset` the way
    Direct3D writes them: the whole 6-dword slot with the payload 8-byte
    aligned, bit 28 set, then SHADER_CONSTANT_FLUSH_FETCH_0-2."""
    pad_to(s, 2, 1)
    s.words.append(pm4.type0(FETCH_CONSTANTS, 6))
    s.relocations.append((len(s.words), 0))
    s.words.append(offset | 3)
    s.words.append(0x10000000 | (dwords << 2) | pm4.ENDIAN_8IN32)
    s.words += [0, 0, 0, 0]
    s.reg(FLUSH_FETCH, 0, 0, 0)


def emit_float_constants(s, constants, relocations=()):
    """Float constants, {vec4 index: 4 words}, as Direct3D writes them: type
    0 packets covering whole groups of 4 constants, their payload 16-byte
    aligned, one per run of consecutive groups holding constants and never
    across banks (c0-c255 vertex, c256-c511 pixel). `relocations` are (vec4
    index, component, shift)."""
    groups = sorted({index // 4 for index in constants})
    runs = []
    for group in groups:
        if runs and group == runs[-1][-1] + 1 and group % 64:
            runs[-1].append(group)
        else:
            runs.append([group])
    positions = {}
    for run in runs:
        pad_to(s, 4, 3)
        s.words.append(pm4.type0(FLOAT_CONSTANTS + 16 * run[0],
                                 16 * len(run)))
        for group in run:
            for index in range(4 * group, 4 * group + 4):
                positions[index] = len(s.words)
                values = constants.get(index, [0, 0, 0, 0])
                s.words += [v & 0xFFFFFFFF for v in values]
    for index, component, shift in relocations:
        s.relocations.append((positions[index] + component, shift))


DRAW_LIMIT = 0xFFFE  # Direct3D's largest auto-index draw
PRIMITIVE_VERTICES = {pm4.PRIM_LINE_LIST: 2, pm4.PRIM_TRIANGLE_LIST: 3,
                      pm4.PRIM_RECTANGLE_LIST: 3}


def emit_draw(s, count, primitive=pm4.PRIM_POINT_LIST, first=0, viz=None):
    """Direct3D's draw bracket for its memexport pass: wait for idle, draws
    of at most DRAW_LIMIT vertices, whole primitives, from increasing
    VGT_INDX_OFFSET, then a cache flush event. `viz` is the draws' VIZ
    token."""
    limit = DRAW_LIMIT - DRAW_LIMIT % PRIMITIVE_VERTICES.get(primitive, 1)
    s.reg('COHER_DEST_BASE_0', 0)
    s.reg('WAIT_UNTIL', 0x00020000)
    for base in range(0, count, limit):
        s.reg('VGT_INDX_OFFSET', first + base)
        s.draw_auto(primitive, min(limit, count - base), viz)
    s.reg('VGT_INDX_OFFSET', 0)
    s.reg('COHER_DEST_BASE_0', 0)
    s.packet3(pm4.EVENT_WRITE, [6])  # CACHE_FLUSH


def emit_zpd(s, address, control=2):
    """Direct3D's sample count report to block offset `address`: the EDRAM
    in copy mode with no surface, RB_SAMPLE_COUNT_CTL (`control`) and _ADDR,
    then EVENT_WRITE_ZPD ZPASS_DONE. Its draw state follows."""
    s.reg('RB_MODECONTROL', 6)
    s.reg('RB_SURFACE_INFO', 0x20000, 0)
    s.words.append(pm4.type0(pm4.register('RB_SAMPLE_COUNT_CTL'), 2))
    s.words.append(control)
    s.relocations.append((len(s.words), 0))
    s.words.append(address)
    s.packet3(pm4.EVENT_WRITE_ZPD, [pm4.ZPASS_DONE])


def emit_indexed_draw(s, count, primitive, first, base, index_format, swap):
    """emit_draw's bracket around one draw of `count` indices at block
    offset `base`, VGT_INDX_OFFSET `first`."""
    s.reg('COHER_DEST_BASE_0', 0)
    s.reg('WAIT_UNTIL', 0x00020000)
    s.reg('VGT_INDX_OFFSET', first)
    s.draw_indexed(primitive, count, base, index_format, swap)
    s.reg('VGT_INDX_OFFSET', 0)
    s.reg('COHER_DEST_BASE_0', 0)
    s.packet3(pm4.EVENT_WRITE, [6])  # CACHE_FLUSH


def resolve_registers(width, height, dest_format, bpp, number=0, endian=2,
                      command=1, sample_select=0, exp_bias=0, source=0,
                      clear=None, clear_enable=3, rect=None, dest=None,
                      offset=0, swap=False, registers=(), after=(), hiz=0):
    """A Resolve of the `width` x `height` rectangle at the origin to a tiled
    texture of the same size at the test's output, as Direct3D computes the
    registers: RB_COPY_CONTROL (command 1 convert, 0 raw), RB_COPY_DEST_PITCH,
    RB_COPY_DEST_INFO and COHER_SIZE_HOST. With `clear`, (depth and stencil
    word, color word), it also clears what `clear_enable` says (1 color, 2
    depth), and RB_HIZ_CLEAR with the depth's top 11 bits.

    Beyond Direct3D's: `rect` is the rectangle's vertices at that
    input dword instead of the first, `dest` the destination's (width,
    height), `offset` bytes added to its address, `swap` RB_COPY_DEST_INFO's
    red and blue swap, `registers` and `after` (register, values) writes
    before and after the Resolve, and `hiz` its RB_HIZCONTROL: 1 to write
    the hierarchical buffer from RB_HIZ_CLEAR with a depth clear."""
    clears = {}
    if clear is not None:
        depth, color = clear
        clears = {'clear': [depth >> 21, depth, color, 0]}
    extra = {}
    if rect is not None:
        extra['rect'] = rect
    if offset:
        extra['offset'] = offset
    if registers or after:
        extra['registers'], extra['after'] = list(registers), list(after)
    if hiz:
        extra['hiz'] = hiz
    dest_width, dest_height = dest or (width, height)
    return {**clears, **extra,
            'control': ((command << 20) | (sample_select << 4) | source |
                        (clear_enable << 8 if clear is not None else 0)),
            'pitch': dest_width | (dest_height << 16),
            'info': (endian | (dest_format << 7) | (number << 13) |
                     ((exp_bias & 0x3F) << 16) | (int(swap) << 24)),
            'coherency': (offset + dest_height * dest_width * bpp + 0x1FFF) &
                         ~0xFFF,
            'bytes': offset + max(dest_width * dest_height * bpp,
                                  tiled_bytes(dest_width, dest_height, bpp))}


def tiled_offset(x, y, pitch, bpp):
    """The byte offset of texel (x, y) of a tiled 2D texture `pitch` texels
    wide (xenia:src/xenia/gpu/texture_address.h, Tiled2D)."""
    log2_bytes = bpp.bit_length() - 1
    pitch = (pitch + 31) & ~31
    macro = ((x >> 5) + (y >> 5) * (pitch >> 5)) << (log2_bytes + 7)
    micro = ((x & 7) + ((y & 0xE) << 2)) << log2_bytes
    offset = macro + ((micro & ~0xF) << 1) + (micro & 0xF) + ((y & 1) << 4)
    return (((offset & ~0x1FF) << 3) + ((y & 16) << 7) +
            ((offset & 0x1C0) << 2) +
            (((((y & 8) >> 2) + (x >> 3)) & 3) << 6) + (offset & 0x3F))


def tiled_bytes(width, height, bpp):
    """The bytes a tiled `width` x `height` texture spans: more than
    width * height * bpp when its tiles leave holes, as 8bpp ones at 64x32."""
    return max(tiled_offset(x, y, width, bpp)
               for y in range(height) for x in range(width)) + bpp


def emit_resolve(s, resolve, dest, vertices):
    """Direct3D's Resolve (its non-tiled path): the copy registers, its
    vertex shader drawing the rectangle list at block offset `vertices` (3 x 2
    floats) in copy mode, then the coherency wait for the destination at block
    offset `dest` (4 KiB aligned). The draw isn't predicated: Direct3D's is,
    on bin select state gpugen never sets."""
    clears = resolve.get('clear', [])
    vertices += 4 * resolve.get('rect', 0)
    emit_registers(s, resolve.get('registers', ()))
    s.words.append(pm4.type0(pm4.register('RB_COPY_CONTROL'),
                             4 + len(clears)))
    s.words.append(resolve['control'])
    s.relocations.append((len(s.words), 0))
    s.words += [dest + resolve.get('offset', 0), resolve['pitch'],
                resolve['info']] + clears
    s.packet3(pm4.INVALIDATE_STATE, [0x100])
    s.im_load_immediate(pm4.VERTEX, resolve_vertex_shader().words())
    s.reg('SQ_PROGRAM_CNTL', 0x00010002, 0)
    s.reg('RB_MODECONTROL', 6)
    # A clearing one tests depth and stencil ALWAYS with writes. The
    # hierarchical buffer stays off unless asked: testing it would reject the
    # copy.
    s.reg('RB_DEPTHCONTROL', 0x8777 if clears else 0)
    s.reg('WAIT_UNTIL', 0x00020000)
    s.reg('RB_HIZCONTROL', resolve.get('hiz', 0))
    s.reg('VGT_MAX_VTX_INDX', 0xFFFF, 0, 0)
    s.reg('PA_CL_CLIP_CNTL', 0x10000, 0x10000, 0x300)
    s.reg('PA_SC_AA_MASK', 0xFFFF)
    s.reg('COHER_DEST_BASE_7', 0)
    s.packet3(pm4.SET_BIN_MASK_LO, [0xFFFFFFFF])
    # Unlike its other fetch constant writes, Direct3D's Resolve doesn't pad
    # this one's payload to 8 bytes.
    s.words.append(pm4.type0(FETCH_CONSTANTS, 6))
    s.relocations.append((len(s.words), 0))
    s.words += [vertices | 3, 0x10000000 | (6 << 2) | pm4.ENDIAN_8IN32, 0, 0,
                0, 0]
    s.reg(FLUSH_FETCH, 0, 0, 0)
    s.draw_auto(pm4.PRIM_RECTANGLE_LIST, 3)
    s.packet3(pm4.EVENT_WRITE, [6])  # CACHE_FLUSH
    s.reg('COHER_DEST_BASE_1', 0)
    s.reg('COHER_STATUS_HOST', 0x03000100)
    s.words.append(pm4.type0(pm4.register('COHER_SIZE_HOST'), 2))
    s.words.append(resolve['coherency'])
    s.relocations.append((len(s.words), 0))
    s.words.append(dest)
    s.packet3(pm4.WAIT_REG_MEM,
              [3, pm4.register('COHER_STATUS_HOST'), 0, 0x80000000, 8])
    s.reg('RB_COPY_CONTROL', 0)
    s.reg('RB_COPY_DEST_INFO', 0)
    emit_registers(s, resolve.get('after', ()))


def emit_setup(s):
    """The setup segment: emit_init, then the dummy draws Direct3D follows
    every SQ_GPR_MANAGEMENT change with."""
    emit_init(s)
    s.packet3(pm4.INVALIDATE_STATE, [0x300])
    s.im_load_immediate(pm4.VERTEX, gpr_dummy_vertex_shader().words())
    s.im_load_immediate(pm4.PIXEL, trivial_pixel_shader().words())
    emit_draw_state(s, 0x1001000E)
    for _ in range(2):
        emit_fetch_constant(s, DUMMY_OFFSET, 7)
        for _ in range(12):
            s.draw_auto(pm4.PRIM_POINT_LIST, 1)
    emit_fence(s)


def build_stream(suite, layout):
    s = pm4.Stream()
    n = suite.vertices
    emit_setup(s)
    pixel = trivial_pixel_shader().words()
    for test in suite.tests:
        # Don't load over a shader a draw may still run.
        s.reg('WAIT_UNTIL', 0x00020000)
        output = layout['outputs'][test.name]
        if test.scratch:
            emit_scratch(s, test.scratch, output, layout)
            continue
        if test.clear:
            for register, values in test.registers:
                s.reg(register, *values)
            emit_resolve(s, test.clear, output, layout['input'])
        s.packet3(pm4.INVALIDATE_STATE, [0x300])
        s.im_load_immediate(pm4.VERTEX, test.shader.words())
        s.im_load_immediate(pm4.PIXEL, test.pixel_shader.words()
                            if test.pixel_shader else pixel)
        emit_draw_state(s, test.program_cntl or suite.program_cntl)
        for register, values in test.registers:
            s.reg(register, *values)
        for kind, offset, values in test.set_constants:
            s.set_constant(kind, offset, values)
        count = suite.slots * test.elements
        bank = 256 if test.pixel_shader else 0
        constants = {bank + C_EA: [0, f2w(float(test.elements)), 0, 0],
                     C_POSITION: [f2w(2.0), f2w(2.0), f2w(0.5), f2w(1.0)],
                     bank + C_FLUSH: [0, 0, 0, 0]}
        relocations = []
        for i in range(test.elements):
            color_format, num_format, size = test.export
            words = stream_constant(
                output + size * i, count, color_format=color_format,
                num_format=num_format)
            if test.stream:
                words = test.stream(words)
            constants[bank + C_STREAM + i] = words
            # Stream X holds the address in dwords.
            relocations.append((bank + C_STREAM + i, 0, 2))
        for index, values in test.constants.items():
            if index < C_FIRST_FREE or bank <= index < bank + C_FIRST_FREE:
                raise ValueError(f'{test.name}: c{index} is reserved')
            if len(values) % 4:
                raise ValueError(f'{test.name}: c{index} needs whole vec4s')
            for i in range(0, len(values), 4):
                constants[index + i // 4] = values[i:i + 4]
        emit_float_constants(s, constants, relocations)
        if suite.inputs:
            emit_fetch_constant(s, layout['input'], len(suite.inputs))
        primitive, first, vertices = test.draw or (pm4.PRIM_POINT_LIST, 0, n)

        def zpd_begin(index):
            if test.zpd != index:
                return
            if test.zpd_reset:
                emit_zpd(s, output, 3)
            emit_zpd(s, output + 32, test.zpd_control[0])
            emit_draw_state(s, test.program_cntl or suite.program_cntl)
            for register, values in test.registers:
                s.reg(register, *values)
        for index, (registers, fetch_constants, *draw) in enumerate(
                test.passes):
            zpd_begin(index)
            emit_registers(s, registers)
            emit_texture_fetch_constants(s, fetch_constants, layout)
            pass_primitive, pass_first, pass_vertices = (
                draw[0] if draw else (primitive, first, vertices))
            query = test.viz_queries.get(index)
            if query is not None:
                s.packet3(pm4.VIZ_QUERY, [query])
            emit_draw(s, pass_vertices, pass_primitive, pass_first)
            if query is not None:
                s.packet3(pm4.VIZ_QUERY, [0x100 | query])
        zpd_begin(len(test.passes))
        emit_registers(s, test.after_passes)
        emit_texture_fetch_constants(s, test.fetch_constants, layout)
        if test.indices:
            texture, index_format, swap = test.indices
            emit_indexed_draw(s, vertices, primitive, first,
                              layout['textures'][texture], index_format, swap)
        else:
            emit_draw(s, vertices, primitive, first, test.viz)
        if test.zpd is not None:
            emit_zpd(s, output, test.zpd_control[1])
        if test.resolve:
            emit_resolve(s, test.resolve, output, layout['input'])
        emit_fence(s)
    extra = SCRATCH_REGISTERS if any(
        test.scratch for test in suite.tests) else ()
    if any(test.zpd is not None or test.viz_queries
           for test in suite.tests):
        extra += QUERY_REGISTERS
    emit_put_back(s, extra)
    return s


def emit_registers(s, registers):
    """(register, values) writes, float constants (whole vec4s from a
    SHADER_CONSTANT_n_X) as emit_float_constants writes them."""
    for register, values in registers:
        index = pm4.register(register) - FLOAT_CONSTANTS
        if 0 <= index < 4 * 512:
            if index % 4 or len(values) % 4:
                raise ValueError(f'{register}: write whole vec4s')
            emit_float_constants(s, {index // 4 + i // 4: values[i:i + 4]
                                     for i in range(0, len(values), 4)})
        else:
            s.reg(register, *values)


def emit_scratch(s, writes, output, layout):
    """SCRATCH_REG0 writes under SCRATCH_UMSK masks (writeback enable bits,
    swap mode in bits 16-17), each with SCRATCH_ADDR at its offset into the
    output, then one of the setup's dummy draws, as Xenia's trace dump plays
    a frame only up to its last draw, and a fence."""
    for offset, mask, value in writes:
        s.words.append(pm4.type0(pm4.register('SCRATCH_ADDR'), 1))
        s.relocations.append((len(s.words), 0))
        s.words.append(output + offset)
        s.reg('SCRATCH_UMSK', mask)
        s.reg('SCRATCH_REG0', value)
    s.reg('SCRATCH_UMSK', 0)
    s.packet3(pm4.INVALIDATE_STATE, [0x300])
    s.im_load_immediate(pm4.VERTEX, gpr_dummy_vertex_shader().words())
    s.im_load_immediate(pm4.PIXEL, trivial_pixel_shader().words())
    emit_draw_state(s, 0x1001000E)
    emit_fetch_constant(s, layout['dummy'], 7)
    s.draw_auto(pm4.PRIM_POINT_LIST, 1)
    emit_fence(s)


def emit_put_back(s, extra=()):
    """The last segment: PUT_BACK_REGISTERS as the title found them. The
    executor reads each from the GPU before submitting and writes it over its
    placeholder, at the ring dword index s.saves names."""
    s.saves = []
    s.reg('WAIT_UNTIL', 0x00020000)
    for name in PUT_BACK_REGISTERS + tuple(extra):
        s.words.append(pm4.type0(pm4.register(name), 1))
        s.saves.append((len(s.words), pm4.register(name)))
        s.words.append(0)
    emit_fence(s)


def emit_texture_fetch_constants(s, fetch_constants, layout):
    """{slot: (6 words, texture name, None or (base, mips) names)}, with the
    texture's block offset added to the base address (and the mips' to the
    mip address)."""
    for slot, (words, texture) in sorted(fetch_constants.items()):
        pad_to(s, 2, 1)
        s.words.append(pm4.type0(FETCH_CONSTANTS + 6 * slot, 6))
        if texture is not None:
            base, mips = texture if isinstance(texture, tuple) else (
                texture, None)
            s.relocations.append((len(s.words) + 1, 0))
            words = list(words)
            words[1] += layout['textures'][base]
            if mips is not None:
                s.relocations.append((len(s.words) + 5, 0))
                words[5] += layout['textures'][mips]
        s.words += [w & 0xFFFFFFFF for w in words]
        s.reg(FLUSH_FETCH, 0, 0, 0)


def emit_fence(s):
    """Ends a segment with Direct3D's fence: wait for idle, then the event
    write the executor polls, of the segment's number."""
    s.reg('WAIT_UNTIL', 0x00020000)
    s.words.append(pm4.type3(pm4.EVENT_WRITE_SHD, 3))
    s.words.append(3)
    s.relocations.append((len(s.words), 0))
    s.words.append(FENCE_OFFSET | pm4.ENDIAN_8IN32)
    s.segments.append(len(s.words) + 1)
    s.words.append(len(s.segments))


# Executor.

def executor(a, thunk, play=False):
    """Allocates the block, brings up the command processor, copies the
    data's blobs into the block, relocates the stream, runs its segments and
    writes the results. r31 holds the control page, r28 the data header, r30
    and r29 the block's virtual and physical addresses.

    A play title's data header has three more words at 0x40, its copy
    records at PLAY_RECORDS: the offsets of a work table of PLAY_WORK_BYTES
    entries a segment, (first read, reads, first stage, stages), and of the
    read and stage tables. Before a segment its reads, (block offset, bytes,
    file offset), fill the block from the memory file; once its fence lands
    its stages, (block offset, block offset, words), copy what it
    resolved."""
    records = PLAY_RECORDS if play else 0x40
    a.b('start')
    # The callback VdShutdownEngines calls once; nothing to clean up.
    callback = a.address()
    a.blr()
    watchdog = emit_watchdog(a, thunk)
    if play:
        status = emit_status(a, thunk)
    a.label('start')
    a.stwu(1, -0x200, 1)
    a.load32(31, CONTROL)
    a.load32(28, DATA)
    a.addi(3, 31, C_WATCHDOG_HANDLE)
    a.li(4, WATCHDOG_STACK)
    a.addi(5, 31, C_WATCHDOG_ID)
    a.li(6, 0)                      # no XAPI startup: start runs directly
    a.load32(7, watchdog)
    a.li(8, 0)
    a.li(9, 0)
    a.call(thunk('ExCreateThread'))
    if play:
        a.addi(3, 31, C_STATUS_HANDLE)
        a.li(4, WATCHDOG_STACK)
        a.addi(5, 31, C_STATUS_ID)
        a.li(6, 0)
        a.load32(7, status)
        a.li(8, 0)
        a.li(9, 0)
        a.call(thunk('ExCreateThread'))
    # Give a host time to attach to XBDM's debug output after the launch.
    a.li(3, 1)
    a.li(4, 0)
    a.addi(5, 31, C_FIRST_SLEEP)
    a.call(thunk('KeDelayExecutionThread'))

    # The physical block.
    a.li(3, P_ALLOCATE)
    a.stw(3, C_PROGRESS, 31)
    a.li(3, 0)
    a.lwz(4, 0x08, 28)              # block size
    a.lwz(5, 0x38, 28)              # protection
    a.li(6, 0)
    a.load32(7, 0x1FFFFFFF)
    a.lis(8, 1)                     # 64 KiB alignment
    a.call(thunk('MmAllocatePhysicalMemoryEx'))
    a.stw(3, C_BLOCK, 31)
    a.cmpwi(3, 0)
    a.beq('fail')
    a.mr(30, 3)
    a.call(thunk('MmGetPhysicalAddress'))
    a.stw(3, C_BLOCK_PHYSICAL, 31)
    a.mr(29, 3)
    if play:
        # The memory file, before the engines: failing leaves nothing to
        # give back.
        open_file(a, thunk, C_PLAY_NAME, C_PLAY_HANDLE, C_PLAY_IOSB,
                  0x8010, 1, 1)     # GENERIC_READ, FILE_SHARE_READ, FILE_OPEN
        a.cmpwi(3, 0)
        a.bne('fail')
        open_file(a, thunk, C_PLAY_LOG_NAME, C_PLAY_LOG_HANDLE,
                  C_PLAY_LOG_IOSB, 0x4010, 0, 5)  # GENERIC_WRITE, OVERWRITE_IF
        a.cmpwi(3, 0)
        a.bne('fail')

    # The command processor first: initializing the ring may clear it.
    a.li(3, P_ENGINES)
    a.stw(3, C_PROGRESS, 31)
    bring_up(a, thunk, callback)

    # Copies: (block offset, data offset, words) records, then fills.
    a.li(3, P_COPY)
    a.stw(3, C_PROGRESS, 31)
    a.lwz(21, 0x0C, 28)             # copy count
    a.addi(22, 28, records)         # copy records
    a.label('copy')
    a.cmpwi(21, 0)
    a.beq('copies_done')
    a.lwz(4, 0, 22)
    a.add(4, 30, 4)                 # destination
    a.lwz(5, 4, 22)
    a.add(5, 28, 5)                 # source
    a.lwz(6, 8, 22)                 # words
    a.label('copy_word')
    a.cmpwi(6, 0)
    a.beq('copy_next')
    a.lwz(7, 0, 5)
    a.stw(7, 0, 4)
    a.addi(4, 4, 4)
    a.addi(5, 5, 4)
    a.addi(6, 6, -1)
    a.b('copy_word')
    a.label('copy_next')
    a.addi(22, 22, 12)
    a.addi(21, 21, -1)
    a.b('copy')
    a.label('copies_done')
    # Sentinel over the outputs, zero fence.
    a.lwz(4, 0x10, 28)
    a.add(4, 30, 4)
    a.lwz(6, 0x14, 28)              # output words
    a.load32(7, SENTINEL)
    a.label('fill')
    a.cmpwi(6, 0)
    a.beq('filled')
    a.stw(7, 0, 4)
    a.addi(4, 4, 4)
    a.addi(6, 6, -1)
    a.b('fill')
    a.label('filled')
    a.load32(4, FENCE_OFFSET)
    a.add(4, 30, 4)
    a.li(7, 0)
    a.stw(7, 0, 4)
    # Relocations: (ring dword index, shift).
    a.lwz(21, 0x18, 28)
    a.lwz(22, 0x1C, 28)
    a.add(22, 28, 22)
    a.label('relocate')
    a.cmpwi(21, 0)
    a.beq('relocated')
    a.lwz(4, 0, 22)
    a.slwi(4, 4, 2)
    a.add(4, 30, 4)                 # ring at block offset 0
    a.lwz(5, 4, 22)                 # shift, 0 or 2
    a.mr(6, 29)
    a.cmpwi(5, 0)
    a.beq('relocate_add')
    a.rlwinm(6, 29, 30, 2, 31)      # physical >> 2
    a.label('relocate_add')
    a.lwz(7, 0, 4)
    a.add(7, 7, 6)
    a.stw(7, 0, 4)
    a.addi(22, 22, 8)
    a.addi(21, 21, -1)
    a.b('relocate')
    a.label('relocated')
    # The registers to give back: (ring dword index, register) records after
    # the segment end table.
    a.lwz(21, 0x3C, 28)
    a.lwz(22, 0x30, 28)
    a.slwi(22, 22, 2)
    a.lwz(3, 0x34, 28)
    a.add(22, 22, 3)
    a.add(22, 28, 22)
    a.load32(23, MMIO_REGISTERS)
    a.label('save')
    a.cmpwi(21, 0)
    a.beq('saved')
    a.lwz(4, 0, 22)
    a.slwi(4, 4, 2)
    a.add(4, 30, 4)                 # ring at block offset 0
    a.lwz(5, 4, 22)
    a.slwi(5, 5, 2)
    a.lwzx(6, 23, 5)
    a.stw(6, 0, 4)
    a.addi(22, 22, 8)
    a.addi(21, 21, -1)
    a.b('save')
    a.label('saved')
    a.sync()

    # Segments one at a time: each ends with a fence of its number, so a hang
    # stops at the segment it is in and each starts on an idle GPU.
    a.li(3, P_SUBMIT)
    a.stw(3, C_PROGRESS, 31)
    a.load32(19, FENCE_OFFSET)
    a.add(19, 30, 19)
    a.lwz(25, 0x30, 28)             # segment count
    a.lwz(26, 0x34, 28)
    a.add(26, 28, 26)               # segment end table
    a.li(24, 0)
    a.label('segment')
    a.cmpw(24, 25)
    a.beq('fenced')
    if play:
        play_before(a, thunk)
    a.sync()
    a.load32(4, MMIO_CP_RB_WPTR)
    a.slwi(5, 24, 2)
    a.lwzx(5, 26, 5)
    a.stw(5, 0, 4)
    a.word(0x7C0006AC)              # eieio
    a.sync()
    a.addi(22, 24, 1)
    a.li(21, 0)
    a.label('poll')
    a.lwz(3, 0, 19)
    a.cmpw(3, 22)
    a.beq('segment_done')
    a.cmpwi(21, FENCE_POLLS)
    a.bge('fenced')
    a.li(3, 1)
    a.li(4, 0)
    a.addi(5, 31, C_POLL_SLEEP)
    a.call(thunk('KeDelayExecutionThread'))
    a.addi(21, 21, 1)
    a.b('poll')
    a.label('segment_done')
    if play:
        play_after(a, thunk)
    a.mr(24, 22)
    a.b('segment')
    a.label('fenced')
    # Every fence landed, the put-back's last, so the GPU is idle and can go
    # back to XAM; VdShutdownEngines would wait forever on a wedged one.
    a.li(3, 1)
    a.cmpw(24, 25)
    a.bne('wedged')
    a.stw(3, C_ENGINES_IDLE, 31)
    a.b('idle')
    a.label('wedged')
    a.stw(3, C_WEDGED, 31)
    a.label('idle')

    # Results: a header right before the outputs in the block, written out
    # with them.
    a.li(3, P_RESULTS)
    a.stw(3, C_PROGRESS, 31)
    a.lwz(20, 0x10, 28)
    a.add(20, 30, 20)
    a.addi(20, 20, -0x20)
    a.load32(3, RESULTS_MAGIC)
    a.stw(3, 0, 20)
    a.li(3, VERSION)
    a.stw(3, 4, 20)
    a.stw(24, 8, 20)                # segments completed
    a.stw(21, 0xC, 20)              # polls of the last one
    a.lwz(3, 0x14, 28)
    a.slwi(3, 3, 2)
    a.addi(3, 3, 0x20)
    a.stw(3, 0x10, 20)              # results size
    a.li(3, 0)
    for offset in (0x14, 0x18, 0x1C):
        a.stw(3, offset, 20)
    a.sync()
    a.b('finish')

    a.label('fail')
    a.lwz(20, 0x24, 28)
    a.load32(3, RESULTS_MAGIC)
    a.stw(3, 0, 20)
    a.li(3, VERSION)
    a.stw(3, 4, 20)
    a.li(3, 0)
    a.stw(3, 8, 20)
    a.li(21, -1)
    a.stw(21, 0xC, 20)
    a.li(3, 0x20)
    a.stw(3, 0x10, 20)
    a.li(3, 0)
    for offset in (0x14, 0x18, 0x1C):
        a.stw(3, offset, 20)
    finish(a, thunk)


def bring_up(a, thunk, callback):
    """The kernel calls that let the command processor run the ring, in
    Direct3D's order. The microcode pointers are 0 in a build without it,
    which loads the kernel's own. VdShutdownEngines calls `callback`."""
    a.load32(3, ENGINES_VERSION)
    a.load32(4, callback)
    a.li(5, 0)
    a.lwz(6, 0x28, 28)              # PFP microcode
    a.lwz(7, 0x2C, 28)              # ME microcode
    a.call(thunk('VdInitializeEngines'))
    a.load32(3, RING_OFFSET)
    a.add(3, 29, 3)
    a.li(4, (RING_SIZE // 8).bit_length() - 1)  # log2 of the size in qwords
    a.call(thunk('VdInitializeRingBuffer'))
    a.load32(3, RPTR_OFFSET)
    a.add(3, 29, 3)
    # log2 of the size in 512-byte blocks, as Direct3D passes it.
    a.li(4, (RING_SIZE // 512).bit_length() - 1)
    a.call(thunk('VdEnableRingBufferRPtrWriteBack'))


def finish(a, thunk):
    """Writes the results at r20, prints the summary and ends the title."""
    a.label('finish')
    write_results(a, thunk, C_CACHE_NAME, 'cache')
    a.cmpwi(3, 0)
    a.beq('written')
    write_results(a, thunk, C_GAME_NAME, 'game')
    a.label('written')
    a.stw(3, C_FILE_STATUS, 31)
    a.mr(6, 3)
    a.lwz(4, 8, 20)
    a.lwz(5, 0xC, 20)
    a.addi(3, 31, C_FORMAT)
    a.call(thunk('DbgPrint'))
    a.lwz(3, C_ELAPSED, 31)
    a.addi(3, 3, WATCHDOG_EXIT_SECONDS)
    a.stw(3, C_DEADLINE, 31)
    a.load32(3, DONE_MAGIC)
    a.sync()
    a.stw(3, C_DONE, 31)
    a.sync()
    a.load32(19, EXIT_MAGIC)
    a.lwz(3, C_FILE_STATUS, 31)
    a.li(21, 10)
    a.cmpwi(3, 0)
    a.beq('exit_wait')
    a.li(21, 300)
    a.label('exit_wait')
    a.lwz(3, C_EXIT, 31)
    a.cmpw(3, 19)
    a.beq('exit')
    a.cmpwi(21, 0)
    a.beq('exit')
    a.li(3, 1)
    a.li(4, 0)
    a.addi(5, 31, C_EXIT_SLEEP)
    a.call(thunk('KeDelayExecutionThread'))
    a.addi(21, 21, -1)
    a.b('exit_wait')
    a.label('exit')
    # A wedged GPU can't go back to XAM: reboot, the results written. So
    # does the end of a run asked to, whose GPU state the dashboard might
    # not survive.
    a.lwz(3, C_WEDGED, 31)
    a.cmpwi(3, 0)
    a.bne('reboot')
    a.lwz(3, C_REBOOT, 31)
    a.cmpwi(3, 0)
    a.beq('give_back')
    a.label('reboot')
    a.li(3, HAL_REBOOT_ROUTINE)
    a.call(thunk('HalReturnToFirmware'))
    a.label('give_back')
    # Give the GPU back to XAM, as a title must before it exits.
    a.lwz(3, C_ENGINES_IDLE, 31)
    a.cmpwi(3, 0)
    a.beq('terminate')
    a.call(thunk('VdShutdownEngines'))
    # The next title of a chain, straight from this one.
    a.lwz(3, C_NEXT_PATH, 31)
    a.cmpwi(3, 0)
    a.beq('terminate')
    a.addi(3, 31, C_NEXT_PATH)
    a.li(4, 0)
    a.call(thunk('XamLoaderLaunchTitle'))
    a.label('terminate')
    a.call(thunk('XamLoaderTerminateTitle'))
    a.label('hang')
    a.b('hang')


def open_file(a, thunk, name_offset, handle_offset, iosb_offset, access,
              share, disposition):
    """NtCreateFile of the path whose ANSI_STRING is at `name_offset`, its
    handle to `handle_offset`, with the high half of the access mask
    `access` (SYNCHRONIZE among them), leaving the status in r3."""
    a.li(3, -3)                     # ObDosDevices
    a.stw(3, C_ATTRIBUTES, 31)
    a.addi(3, 31, name_offset)
    a.stw(3, C_ATTRIBUTES + 4, 31)
    a.li(3, 0x40)                   # OBJ_CASE_INSENSITIVE
    a.stw(3, C_ATTRIBUTES + 8, 31)
    a.li(3, 0)
    a.stw(3, 0x50, 1)
    a.li(3, 0x60)                   # synchronous, not a directory
    a.stw(3, 0x54, 1)
    a.addi(3, 31, handle_offset)
    a.lis(4, access)
    a.addi(5, 31, C_ATTRIBUTES)
    a.addi(6, 31, iosb_offset)
    a.li(7, 0)
    a.li(8, 0x80)                   # FILE_ATTRIBUTE_NORMAL
    a.li(9, share)
    a.li(10, disposition)
    a.call(thunk('NtCreateFile'))


def play_log(a, thunk, step):
    """Logs segment r24's `step` (1 reading, 2 submitting, 3 fenced, 4
    staged) to the play title's log file, on the disk before it goes on."""
    a.stw(24, C_PLAY_LOG_WORDS, 31)
    a.li(3, step)
    a.stw(3, C_PLAY_LOG_WORDS + 4, 31)
    a.lwz(3, C_PLAY_LOG_HANDLE, 31)
    a.li(4, 0)
    a.li(5, 0)
    a.li(6, 0)
    a.addi(7, 31, C_PLAY_LOG_IOSB)
    a.addi(8, 31, C_PLAY_LOG_WORDS)
    a.li(9, 8)
    a.addi(10, 31, C_PLAY_LOG_OFFSET)
    a.call(thunk('NtWriteFile'))
    a.lwz(3, C_PLAY_LOG_HANDLE, 31)
    a.addi(4, 31, C_PLAY_LOG_IOSB)
    a.call(thunk('NtFlushBuffersFile'))


def play_before(a, thunk):
    """A play title's work before segment r24: its reads from the memory
    file. Leaves its work entry in r27; a failed read stops the run
    there."""
    play_log(a, thunk, 1)
    a.lwz(27, 0x40, 28)
    a.add(27, 28, 27)
    a.slwi(3, 24, 4)                # PLAY_WORK_BYTES
    a.add(27, 27, 3)
    a.lwz(16, 0, 27)
    a.lwz(18, 4, 27)
    a.slwi(3, 16, 2)                # 12-byte reads
    a.slwi(16, 16, 3)
    a.add(16, 16, 3)
    a.lwz(3, 0x44, 28)
    a.add(16, 16, 3)
    a.add(16, 28, 16)
    a.label('play_read')
    a.cmpwi(18, 0)
    a.beq('play_read_done')
    a.li(3, 0)
    a.stw(3, C_PLAY_OFFSET, 31)
    a.lwz(3, 8, 16)
    a.stw(3, C_PLAY_OFFSET + 4, 31)
    a.lwz(3, C_PLAY_HANDLE, 31)
    a.li(4, 0)
    a.li(5, 0)
    a.li(6, 0)
    a.addi(7, 31, C_PLAY_IOSB)
    a.lwz(8, 0, 16)
    a.add(8, 30, 8)
    a.lwz(9, 4, 16)
    a.addi(10, 31, C_PLAY_OFFSET)
    a.call(thunk('NtReadFile'))
    a.cmpwi(3, 0)
    a.bne('fenced')
    a.addi(16, 16, 12)
    a.addi(18, 18, -1)
    a.b('play_read')
    a.label('play_read_done')
    play_log(a, thunk, 2)


def play_after(a, thunk):
    """A play title's work once segment r24's fence lands: its stages, from
    its work entry in r27."""
    play_log(a, thunk, 3)
    a.lwz(16, 8, 27)
    a.lwz(18, 12, 27)
    a.slwi(3, 16, 2)                # 12-byte stages
    a.slwi(16, 16, 3)
    a.add(16, 16, 3)
    a.lwz(3, 0x48, 28)
    a.add(16, 16, 3)
    a.add(16, 28, 16)
    a.label('play_stage')
    a.cmpwi(18, 0)
    a.beq('play_staged')
    a.lwz(4, 0, 16)
    a.add(4, 30, 4)
    a.lwz(5, 4, 16)
    a.add(5, 30, 5)
    a.lwz(6, 8, 16)
    a.label('play_stage_word')
    a.cmpwi(6, 0)
    a.beq('play_stage_next')
    a.lwz(7, 0, 4)
    a.stw(7, 0, 5)
    a.addi(4, 4, 4)
    a.addi(5, 5, 4)
    a.addi(6, 6, -1)
    a.b('play_stage_word')
    a.label('play_stage_next')
    a.addi(16, 16, 12)
    a.addi(18, 18, -1)
    a.b('play_stage')
    a.label('play_staged')
    a.sync()
    play_log(a, thunk, 4)


def write_results(a, thunk, name_offset, suffix):
    """Writes the results at r20 (size at +0x10) to the path whose
    ANSI_STRING is at `name_offset`, leaving the status in r3."""
    open_file(a, thunk, name_offset, C_HANDLE, C_IOSB,
              0x4010, 0, 5)         # GENERIC_WRITE, FILE_OVERWRITE_IF
    a.cmpwi(3, 0)
    a.bne(f'write_{suffix}_done')
    a.lwz(3, C_HANDLE, 31)
    a.li(4, 0)
    a.li(5, 0)
    a.li(6, 0)
    a.addi(7, 31, C_IOSB)
    a.mr(8, 20)
    a.lwz(9, 0x10, 20)
    a.addi(10, 31, C_OFFSET)
    a.call(thunk('NtWriteFile'))
    a.mr(19, 3)
    # On the disk before any reboot.
    a.lwz(3, C_HANDLE, 31)
    a.addi(4, 31, C_IOSB)
    a.call(thunk('NtFlushBuffersFile'))
    a.lwz(3, C_HANDLE, 31)
    a.call(thunk('NtClose'))
    a.mr(3, 19)
    a.label(f'write_{suffix}_done')


def emit_watchdog(a, thunk):
    """The watchdog thread's start routine, at the address it returns: it
    counts seconds into C_ELAPSED and reboots the console once they reach
    C_DEADLINE, which the title moves on when it finishes."""
    start = a.address()
    a.stwu(1, -0x80, 1)
    a.load32(31, CONTROL)
    a.li(21, 0)
    a.label('watchdog_tick')
    a.li(3, 1)
    a.li(4, 0)
    a.addi(5, 31, C_WATCHDOG_SLEEP)
    a.call(thunk('KeDelayExecutionThread'))
    a.addi(21, 21, 1)
    a.stw(21, C_ELAPSED, 31)
    a.lwz(4, C_DEADLINE, 31)
    a.cmpw(21, 4)
    a.blt('watchdog_tick')
    a.li(3, HAL_REBOOT_ROUTINE)
    a.call(thunk('HalReturnToFirmware'))
    a.b('watchdog_tick')
    return start


def emit_status(a, thunk):
    """The status thread's start routine, at the address it returns: it
    binds STATUS_PORT and answers each datagram to it from the control
    page, for good, waiting a second after a failed receive; with no
    network it only sleeps."""
    start = a.address()
    a.stwu(1, -0x300, 1)
    a.load32(31, CONTROL)
    a.li(3, 2)
    a.addi(4, 31, C_NET_PARAMS)
    a.call(thunk('NetDll_XNetStartup'))
    a.cmpwi(3, 0)
    a.bne('status_idle')
    a.li(3, 2)
    a.li(4, 0x202)
    a.addi(5, 1, 0x100)             # WSADATA
    a.call(thunk('NetDll_WSAStartup'))
    a.cmpwi(3, 0)
    a.bne('status_idle')
    a.li(3, 2)
    a.li(4, 2)                      # AF_INET
    a.li(5, 2)                      # SOCK_DGRAM
    a.li(6, 17)                     # IPPROTO_UDP
    a.call(thunk('NetDll_socket'))
    a.cmpwi(3, 0)
    a.blt('status_idle')
    a.mr(30, 3)
    # Plain UDP, not XNet's secured traffic.
    a.li(3, 2)
    a.mr(4, 30)
    a.load32(5, 0xFFFF)             # SOL_SOCKET
    a.li(6, 0x5801)
    a.addi(7, 31, C_NET_ONE)
    a.li(8, 4)
    a.call(thunk('NetDll_setsockopt'))
    a.li(3, 2)
    a.mr(4, 30)
    a.addi(5, 31, C_NET_ADDRESS)
    a.li(6, 16)
    a.call(thunk('NetDll_bind'))
    a.cmpwi(3, 0)
    a.bne('status_idle')
    a.label('status_wait')
    a.li(3, 16)
    a.stw(3, C_NET_FROM_BYTES, 31)
    a.li(3, 2)
    a.mr(4, 30)
    a.addi(5, 31, C_NET_REQUEST)
    a.li(6, 16)
    a.li(7, 0)
    a.addi(8, 31, C_NET_FROM)
    a.addi(9, 31, C_NET_FROM_BYTES)
    a.call(thunk('NetDll_recvfrom'))
    a.cmpwi(3, 0)
    a.blt('status_retry')
    for i, offset in enumerate((C_ELAPSED, C_PROGRESS, C_PLAY_LOG_WORDS,
                                C_PLAY_LOG_WORDS + 4, C_NET_ERROR)):
        a.lwz(3, offset, 31)
        a.stw(3, C_NET_REPLY + 4 + 4 * i, 31)
    a.li(3, 2)
    a.mr(4, 30)
    a.addi(5, 31, C_NET_REPLY)
    a.li(6, STATUS_REPLY_BYTES)
    a.li(7, 0)
    a.addi(8, 31, C_NET_FROM)
    a.li(9, 16)
    a.call(thunk('NetDll_sendto'))
    a.b('status_wait')
    a.label('status_retry')
    a.call(thunk('NetDll_WSAGetLastError'))
    a.stw(3, C_NET_ERROR, 31)
    a.li(3, 1)
    a.li(4, 0)
    a.addi(5, 31, C_WATCHDOG_SLEEP)
    a.call(thunk('KeDelayExecutionThread'))
    a.b('status_wait')
    a.label('status_idle')
    a.li(3, 1)
    a.li(4, 0)
    a.addi(5, 31, C_WATCHDOG_SLEEP)
    a.call(thunk('KeDelayExecutionThread'))
    a.b('status_idle')
    return start


def control_page(results=NAME, next_name=None, deadline=WATCHDOG_SECONDS,
                 reboot=False, memory=None):
    """Results go to RESULTS.bin; with `next_name` the title then launches
    game:\\NEXT_NAME.xex in place of going back to the dashboard, with
    `reboot` it reboots the console instead. The watchdog reboots the console
    `deadline` seconds after the start unless the title has finished. A play
    title reads its trace's memory from game:\\MEMORY and logs its steps to
    game:\\RESULTS.log."""
    page = bytearray(C_FORMAT + len(FORMAT))
    if memory:
        for string, path, offset, end in (
                (C_PLAY_NAME, f'game:\\{memory}', C_PLAY_PATH,
                 C_PLAY_LOG_HANDLE),
                (C_PLAY_LOG_NAME, f'game:\\{results}.log', C_PLAY_LOG_PATH,
                 C_PLAY_END)):
            path = path.encode()
            if offset + len(path) >= end:
                raise ValueError(f'{path!r} too long')
            struct.pack_into('>HHI', page, string, len(path), len(path) + 1,
                             CONTROL + offset)
            page[offset:offset + len(path)] = path
        struct.pack_into('>I', page, C_NET_ONE, 1)
        # sockaddr_in: the family in the CPU's order, the port in the
        # network's, which are the same here.
        struct.pack_into('>HH', page, C_NET_ADDRESS, 2, STATUS_PORT)
        page[C_NET_PARAMS] = 13     # XNetStartup's parameters' size
        struct.pack_into('>I', page, C_NET_REPLY, STATUS_MAGIC)
    struct.pack_into('>I', page, C_DEADLINE, deadline)
    struct.pack_into('>I', page, C_REBOOT, int(reboot))
    struct.pack_into('>q', page, C_WATCHDOG_SLEEP, -1000 * 10000)
    struct.pack_into('>q', page, C_FIRST_SLEEP, -3000 * 10000)
    struct.pack_into('>q', page, C_POLL_SLEEP, -1 * 10000)
    struct.pack_into('>q', page, C_EXIT_SLEEP, -100 * 10000)
    for string, path, offset, end in (
            (C_CACHE_NAME, f'cache:\\{results}.bin', C_CACHE_PATH,
             C_GAME_PATH),
            (C_GAME_NAME, f'game:\\{results}.bin', C_GAME_PATH, C_NEXT_PATH)):
        path = path.encode()
        if offset + len(path) >= end:
            raise ValueError(f'results name {results!r} too long')
        struct.pack_into('>HHI', page, string, len(path), len(path) + 1,
                         CONTROL + offset)
        page[offset:offset + len(path)] = path
    if next_name:
        path = f'game:\\{next_name}.xex'.encode()
        if C_NEXT_PATH + len(path) >= C_BLOCK:
            raise ValueError(f'next name {next_name!r} too long')
        page[C_NEXT_PATH:C_NEXT_PATH + len(path)] = path
    page[C_FORMAT:] = FORMAT
    return bytes(page)


def find_blobs(image, specs):
    """Blobs found in a decoded XEX image by (SHA-1, dword count, SHA-1 of
    the first 16 bytes), at dword-aligned offsets."""
    view = memoryview(image)
    found = []
    for digest, dwords, head in specs:
        head = bytes.fromhex(head)
        for start in range(0, len(image) - 4 * dwords + 1, 4):
            if hashlib.sha1(view[start:start + 16]).digest() != head:
                continue
            blob = image[start:start + 4 * dwords]
            if hashlib.sha1(blob).hexdigest() == digest:
                found.append(blob)
                break
        else:
            raise ValueError(f'no blob with SHA-1 {digest} in the image')
    return found


def build(suite, outdir, image=None, results=NAME, next_name=None,
          watchdog=None, reboot=False):
    # Block layout.
    layout = {'input': INPUT_OFFSET, 'dummy': DUMMY_OFFSET, 'outputs': {},
              'textures': {}}
    offset = INPUT_OFFSET + ((4 * len(suite.inputs) + 0xFFF) & ~0xFFF)
    for name, words in suite.textures.items():
        layout['textures'][name] = offset
        offset += (4 * len(words) + 0xFFF) & ~0xFFF
    # A page for the results header before the outputs.
    offset += 0x1000
    outputs_offset = offset
    for test in suite.tests:
        if test.resolve or test.scratch or test.zpd is not None:
            offset = (offset + 0xFFF) & ~0xFFF
        layout['outputs'][test.name] = offset
        offset += (test.output_bytes(suite.slots) + 15) & ~15
        # Room past the end for exports a test aims out of range.
        offset += 0x1000
    output_words = (offset - outputs_offset) // 4
    block_size = (offset + 0xFFFF) & ~0xFFFF

    stream = build_stream(suite, layout)
    blobs = [(RING_OFFSET, stream.words), (INPUT_OFFSET, suite.inputs),
             (DUMMY_OFFSET, [0] * 8)]
    blobs += [(layout['textures'][name], words)
              for name, words in suite.textures.items()]
    xex, console_ready, watchdog = write_title(
        outdir, stream, blobs, stream.relocations, block_size,
        outputs_offset, output_words, image, results, next_name, watchdog,
        reboot, suite.title_id)
    meta = {'suite': suite.name, 'title_id': suite.title_id,
            'version': VERSION, 'control': CONTROL,
            'segments': len(stream.segments), 'raw': suite.raw,
            'console_ready': console_ready,
            'results': results, 'next': next_name, 'reboot': reboot,
            'watchdog': watchdog,
            'tests': [{'name': t.name, 'elements': t.elements,
                       'offset': layout['outputs'][t.name] - outputs_offset,
                       'element_bytes': t.element_bytes,
                       'output_bytes': t.output_bytes(suite.slots),
                       'shader': t.shader.words() if t.shader else []}
                      for t in suite.tests]}
    write_trace(os.path.join(outdir, 'gpugen.xtr'), stream, blobs,
                outputs_offset, output_words, suite.title_id)
    meta['trace'] = {'base': TRACE_BASE + outputs_offset,
                     'words': output_words}
    with open(os.path.join(outdir, 'gpugen.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    print(f'{len(suite.tests)} tests over {suite.vertices} vertices, '
          f'{xex:#x} bytes, stream {len(stream.words)} words, '
          f'results {0x20 + 4 * output_words:#x} bytes')


def write_title(outdir, stream, blobs, relocations, block_size,
                outputs_offset, output_words, image, results, next_name,
                watchdog, reboot, title_id, play=None, memory=None):
    """Writes OUTDIR/gpugen.xex: the executor and its data, copying `blobs`,
    (block offset, words), into the block and relocating the block dwords
    `relocations` name. `play` is a play title's (work, reads, stages)
    tables and `memory` the name of the memory file it reads. Returns the
    XEX's size, whether it carries the microcode and its watchdog."""
    exports = xamgen.load_exports()
    imports = xexbuild.Imports(IMPORTS_BASE)

    def thunk(name):
        return imports.function(*exports[name])

    microcode = None
    if image:
        microcode = find_blobs(image, (PFP_MICROCODE, ME_MICROCODE))
    if 4 * len(stream.words) > RING_SIZE:
        raise ValueError('command stream larger than the ring')

    asm = Assembler(CODE_BASE)
    executor(asm, thunk, play is not None)
    text = asm.code()
    if CODE_BASE + len(text) > IMPORTS_BASE:
        raise ValueError('executor runs into the import records')
    code = bytes(text) + bytes(IMPORTS_BASE - CODE_BASE - len(text))
    code += imports.data()

    # Data: header at 0, copy records at 0x40 (a play title's at
    # PLAY_RECORDS), then the blobs.
    records_offset = PLAY_RECORDS if play is not None else 0x40
    data = bytearray(records_offset + 12 * len(blobs))
    records = []
    for block_offset, words in blobs:
        data += bytes(-len(data) % 16)
        records.append((block_offset, len(data), len(words)))
        data += struct.pack(f'>{len(words)}I', *words)
    for i, record in enumerate(records):
        struct.pack_into('>III', data, records_offset + 12 * i, *record)
    data += bytes(-len(data) % 16)
    microcode_addresses = [0, 0]
    for i, blob in enumerate(microcode or []):
        microcode_addresses[i] = DATA + len(data)
        data += blob + bytes(-len(blob) % 16)
    relocations_offset = len(data)
    for index, shift in relocations:
        data += struct.pack('>II', index, shift)
    segments_offset = len(data)
    data += struct.pack(f'>{len(stream.segments)}I', *stream.segments)
    for index, register in stream.saves:
        data += struct.pack('>II', index, register)
    if play is not None:
        for i, table in enumerate(play):
            data += bytes(-len(data) % 16)
            struct.pack_into('>I', data, 0x40 + 4 * i, len(data))
            data += table
    # Only the failure path writes results from the image, just the header; a
    # run writes them from the block.
    failure_results = DATA + ((len(data) + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1))
    struct.pack_into('>IIIIIIIIIIII', data, 0, DATA_MAGIC, VERSION, block_size,
                     len(records), outputs_offset, output_words,
                     len(relocations), relocations_offset,
                     len(stream.words), failure_results,
                     *microcode_addresses)
    struct.pack_into('>II', data, 0x30, len(stream.segments), segments_offset)
    struct.pack_into('>I', data, 0x3C, len(stream.saves))
    # Large blocks take 64 KiB pages like Direct3D's big buffers (0x20000404):
    # with 4 KiB pages a 32 MiB block ends up near the top of the address
    # space, and the title faulted in the kernel.
    protect = PAGE_READWRITE | PAGE_WRITECOMBINE
    if block_size > 0x400000:
        protect |= MEM_LARGE_PAGES
    struct.pack_into('>I', data, 0x38, protect)

    watchdog = watchdog or (WATCHDOG_SECONDS +
                            WATCHDOG_SEGMENT_SECONDS * len(stream.segments) +
                            block_size // WATCHDOG_BYTES_PER_SECOND)
    pages = [(xexbuild.SECTION_CODE, code),
             (xexbuild.SECTION_DATA, control_page(
                 results, next_name, watchdog, reboot, memory))]
    for start in range(0, len(data), PAGE_SIZE):
        pages.append((xexbuild.SECTION_DATA,
                      bytes(data[start:start + PAGE_SIZE])))
    pages.append((xexbuild.SECTION_DATA, b''))  # the failure results
    xex = xexbuild.build(LOAD_ADDRESS, CODE_BASE, pages, title_id=title_id,
                         imports=imports)
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, 'gpugen.xex'), 'wb') as f:
        f.write(xex)
    return len(xex), microcode is not None, watchdog


# Where a trace places the block: Xenia replays the stream from there without
# running the executor.
TRACE_BASE = 0x10000000
# trace_protocol.h's TraceCommandType, MemoryEncodingFormat.
TRACE_PACKET_START, TRACE_PACKET_END, TRACE_MEMORY_READ = 4, 5, 6


def write_trace(path, stream, blobs, outputs_offset, output_words, title_id):
    """An .xtr of what the executor does, for Xenia's trace dump: the blobs
    and the output sentinels, then the stream's packets up to the put-back,
    relocated to TRACE_BASE."""
    words = list(stream.words)
    for index, shift in stream.relocations:
        words[index] = (words[index] + (TRACE_BASE >> shift)) & 0xFFFFFFFF
    out = bytearray(struct.pack('<I40sI', 1, b'0' * 40, title_id))

    def memory(offset, data):
        out.extend(struct.pack('<5I', TRACE_MEMORY_READ, TRACE_BASE + offset,
                               0, len(data), len(data)))
        out.extend(data)

    for offset, blob in blobs[1:]:
        memory(offset, struct.pack(f'>{len(blob)}I', *blob))
    memory(outputs_offset, struct.pack('>I', SENTINEL) * output_words)
    i, end = 0, stream.segments[-2]
    while i < end:
        header = words[i]
        kind = header >> 30
        count = 1 + (((header >> 16) & 0x3FFF) + 1 if kind in (0, 3) else
                     2 if kind == 1 else 0)
        out.extend(struct.pack('<3I', TRACE_PACKET_START,
                               TRACE_BASE + RING_OFFSET + 4 * i, count))
        out.extend(struct.pack(f'>{count}I', *words[i:i + count]))
        out.extend(struct.pack('<I', TRACE_PACKET_END))
        i += count
    with open(path, 'wb') as f:
        f.write(out)


def decode(meta, data):
    """The results as {'tests': {name: hex words}}, or for a raw suite
    {'raw': {name: bytes}}, with the segments done and the tests not run."""
    magic, version, done, polls, size = struct.unpack_from('>IIIiI', data, 0)
    if magic != RESULTS_MAGIC or version != VERSION:
        raise ValueError('not gpugen results')
    # Segment 0 is the setup, test i is segment i + 1; a play title's stages
    # name theirs, as its eDRAM dump does, decoded with them.
    tests = ([meta['edram']] if 'edram' in meta else []) + meta['tests']
    out = {'suite': meta['suite'], 'segments_done': done,
           'segments': meta['segments'], 'polls': polls, 'tests': {},
           'not_run': [t['name'] for i, t in enumerate(meta['tests'])
                       if done < t.get('segment', i + 1) + 1]}
    if 'edram' in meta and done < meta['edram']['segment'] + 1:
        out['not_run'].insert(0, meta['edram']['name'])
    if polls < 0:
        out['failed'] = 'no physical memory'
        return out
    body = data[0x20:size]
    for test in tests:
        length = (test['output_bytes'] + 3) & ~3
        if test['offset'] + length > len(body):
            raise ValueError('results from another build or cut short')
        if test['name'] in out['not_run']:
            continue
        output = body[test['offset']:test['offset'] + length]
        if meta['raw']:
            out.setdefault('raw', {})[test['name']] = output
        else:
            out['tests'][test['name']] = [
                f'{w:08X}' for w in struct.unpack(f'>{length // 4}I', output)]
    return out


def save_raw(decoded, directory, prefix):
    """Moves a raw suite's results to DIRECTORY/PREFIX_<test>.bin, naming
    each file in decoded['tests']."""
    for name, output in decoded.pop('raw', {}).items():
        path = os.path.join(directory, f'{prefix}_{name}.bin')
        with open(path, 'wb') as f:
            f.write(output)
        decoded['tests'][name] = os.path.basename(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('build')
    p.add_argument('suite')
    p.add_argument('outdir')
    p.add_argument('--microcode', metavar='IMAGE',
                   help='a decoded XEX image linking Direct3D to take the '
                   'command processor microcode from; needed to run on a '
                   'console')
    p.add_argument('--results', default=NAME, metavar='NAME',
                   help='write the results to NAME.bin')
    p.add_argument('--next', metavar='NAME',
                   help='launch game:\\NAME.xex after a run that gave the '
                   'GPU back, to run suites one after another')
    p.add_argument('--watchdog', type=int, metavar='SECONDS',
                   help='reboot the console this long after the start '
                   'unless the run has finished (default: from the size)')
    p.add_argument('--reboot', action='store_true',
                   help='reboot the console after the run instead of going '
                   'back to the dashboard')
    p = sub.add_parser('decode')
    p.add_argument('meta')
    p.add_argument('results')
    args = parser.parse_args()
    if args.command == 'build':
        spec = importlib.util.spec_from_file_location('suite', args.suite)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        suite = Suite(os.path.splitext(os.path.basename(args.suite))[0])
        module.build(suite)
        image = None
        if args.microcode:
            with open(args.microcode, 'rb') as f:
                image = f.read()
        build(suite, args.outdir, image, args.results, args.next,
              args.watchdog, args.reboot)
    else:
        with open(args.meta) as f:
            meta = json.load(f)
        with open(args.results, 'rb') as f:
            decoded = decode(meta, f.read())
        save_raw(decoded, os.path.dirname(os.path.abspath(args.results)),
                 os.path.splitext(os.path.basename(args.results))[0])
        json.dump(decoded, sys.stdout, indent=1)
        print()


if __name__ == '__main__':
    main()
