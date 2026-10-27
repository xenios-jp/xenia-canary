"""What a VIZ query leaves in the status registers: PA_SC_VIZ_QUERY_STATUS_0
and _1 (a bit for each of the 64 query IDs, set when visible, per
register_table.inc), read into memory with REG_TO_MEM right after the query
and again after a wait for the 3D engine to be idle, beside PA_SC_VIZ_QUERY
itself (to show REG_TO_MEM reads these registers at all). The queries are
vizpred.py's: a VIZ query over a draw of the whole 64x32 render target at
depth 0.5, visible with the depth test ALWAYS, occluded with NEVER, then the
predicated draw and the occlusion query over it.

The output is the queries suite's two reports (+0 end, +32 start) and then
for each read (+64, +80), the three registers as big-endian words: STATUS_0,
STATUS_1, PA_SC_VIZ_QUERY.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpugen  # noqa: E402
import pm4  # noqa: E402
import queries  # noqa: E402
import rb  # noqa: E402
import rbdraw  # noqa: E402

REG_TO_MEM = 0x3E
VIZ_ENABLE = queries.VIZ_ENABLE
KILL_HI_Z = 0x80
WAIT_3D_IDLE = 0x00020000
READS = 2
OUTPUT_BYTES = 64 + 16 * READS

# The suite's own packets, written where a test lists a READ_VIZ_STATUS
# `register`: REG_TO_MEM of the three registers to read `slot` of the output
# of the test whose occlusion query reports were written last (at output +
# 32, the start report).
_report = {}
# gpugen.py run as a script is __main__, and this suite's `import gpugen` a
# second copy of it: the one building the stream is the first.
_gpugen = sys.modules['__main__'] if hasattr(
    sys.modules['__main__'], 'build_stream') else gpugen
_emit_zpd = _gpugen.emit_zpd
_emit_registers = _gpugen.emit_registers
_output_bytes = gpugen.Test.output_bytes


def emit_zpd(s, address, control=2):
    _report['address'] = address
    _emit_zpd(s, address, control)


def emit_registers(s, registers):
    for register, values in registers:
        if register != 'READ_VIZ_STATUS':
            _emit_registers(s, [(register, values)])
            continue
        output = _report['address'] - 32 + 64 + 16 * values[0]
        for i, name in enumerate(('PA_SC_VIZ_QUERY_STATUS_0',
                                  'PA_SC_VIZ_QUERY_STATUS_1',
                                  'PA_SC_VIZ_QUERY')):
            s.words.append(pm4.type3(REG_TO_MEM, 2))
            s.words.append(pm4.register(name))
            s.relocations.append((len(s.words), 0))
            s.words.append((output + 4 * i) | pm4.ENDIAN_8IN32)


def output_bytes(self, slots):
    return OUTPUT_BYTES if getattr(self, 'reads', False) else _output_bytes(
        self, slots)


_gpugen.emit_zpd = emit_zpd
_gpugen.emit_registers = emit_registers


def build(suite):
    rects = rbdraw.Shapes()
    suite.add_texture('half', queries.texture(lambda x: 0.5))
    suite.add_texture('ramp', queries.texture(
        lambda x: (x + 0.5) / queries.WIDTH))
    whole = rb.RECT_DRAW
    nothing = (pm4.PRIM_RECTANGLE_LIST, 0, 0)
    reset = ([('RB_DEPTHCONTROL', [queries.RESET])], rb.fetch('half'), whole)
    on = VIZ_ENABLE | (1 << 1)

    def status(name, depthcontrol, query=1, during=None, draw=whole, hiz=0,
               fetched='ramp'):
        if during is None:
            during = VIZ_ENABLE | (query << 1) | KILL_HI_Z
        suite.test(name, gpugen.resolve_vertex_shader(),
                   constants=rb.CONSTANTS,
                   pixel_shader=rb.pixel_shader(depth=True),
                   program_cntl=queries.PROGRAM_CNTL,
                   registers=rb.registers(0, 0, depth_format=0) + [
                       ('RB_COLORCONTROL', [queries.COLORCONTROL])],
                   passes=[([('PA_SC_VIZ_QUERY', [0]),
                             ('RB_HIZCONTROL', [hiz])] + reset[0],
                            reset[1], reset[2]),
                           ([('RB_DEPTHCONTROL', [depthcontrol]),
                             ('PA_SC_VIZ_QUERY', [during])],
                            rb.fetch(fetched), draw)],
                   after_passes=[('READ_VIZ_STATUS', [0]),
                                 ('WAIT_UNTIL', [WAIT_3D_IDLE]),
                                 ('READ_VIZ_STATUS', [1]),
                                 ('PA_SC_VIZ_QUERY', [0]),
                                 ('RB_DEPTHCONTROL',
                                  [queries.zfunc(queries.ALWAYS)])],
                   fetch_constants=rb.fetch('half'), draw=whole, zpd=2,
                   viz_queries={1: query}, viz=0x100 | query)
        suite.tests[-1].reads = True
        type(suite.tests[-1]).output_bytes = output_bytes

    ALWAYS, NEVER, LESS = (queries.zfunc(f) for f in
                           (queries.ALWAYS, queries.NEVER, queries.LESS))
    status('status visible', ALWAYS)
    status('status occluded', NEVER)
    status('status less half', LESS)
    status('status nothing drawn', ALWAYS, draw=nothing)
    status('status visible id 33', ALWAYS, query=33)
    status('status occluded id 33', NEVER, query=33)
    status('status visible id 63', ALWAYS, query=63)
    status('status visible no kill', ALWAYS, during=on)
    status('status occluded no kill', NEVER, during=on)
    status('status visible hiz 0x3', ALWAYS, hiz=0x3)
    status('status occluded hiz 0x3', NEVER, hiz=0x3)
    suite.set_inputs(rects.inputs, 2)
