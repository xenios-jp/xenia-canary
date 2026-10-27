"""What a VIZ query leaves in the status registers, vizstatus.py's tests made
repeatable: PA_SC_VIZ_QUERY_STATUS_0 and _1 read right after the query and
again after a wait for the 3D engine to be idle, beside PA_SC_VIZ_QUERY.

vizstatus's three console runs differed in two ways, and this suite removes
both:
- the sample count reports are of free-running counters that start from
  whatever the console's earlier work left, so every test here resets them
  before the predicated draw's start report (RB_SAMPLE_COUNT_CTL 3, as
  queries.py's 'zpd after reset'). The reports' end minus start is the same on
  every run.
- a status register keeps the bit of every query ID until a query of that ID
  runs again: one run read bit 2 set in STATUS_0 all through, left by an ID 2
  query of the vizpred2 suite earlier in its chain. Every test here therefore
  begins with a VIZ query of each of the 64 IDs over an empty draw
  (PA_SC_VIZ_QUERY on for it, as 'status nothing drawn' does), which clears
  them all, so the registers hold the test's own bits alone. 'status nothing
  drawn' reads zero in both registers after them, which checks that clearing.

The output is vizstatus's: the queries suite's two reports (+0 end, +32 start)
and then for each read (+64, +80), STATUS_0, STATUS_1 and PA_SC_VIZ_QUERY as
big-endian words.
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
import vizstatus  # noqa: E402  (its READ_VIZ_STATUS register and output size)

VIZ_ENABLE = queries.VIZ_ENABLE
KILL_HI_Z = vizstatus.KILL_HI_Z
WAIT_3D_IDLE = vizstatus.WAIT_3D_IDLE
IDS = 64


def build(suite):
    rects = rbdraw.Shapes()
    suite.add_texture('half', queries.texture(lambda x: 0.5))
    suite.add_texture('ramp', queries.texture(
        lambda x: (x + 0.5) / queries.WIDTH))
    whole = rb.RECT_DRAW
    nothing = (pm4.PRIM_RECTANGLE_LIST, 0, 0)
    reset = ([('RB_DEPTHCONTROL', [queries.RESET])], rb.fetch('half'), whole)
    on = VIZ_ENABLE | (1 << 1)
    # A query of every ID over no draw, which clears that ID's status bit.
    clears = [([('PA_SC_VIZ_QUERY', [VIZ_ENABLE | (i << 1) | KILL_HI_Z])],
               rb.fetch('half'), nothing) for i in range(IDS)]

    def status(name, depthcontrol, query=1, during=None, draw=whole, hiz=0,
               fetched='ramp'):
        if during is None:
            during = VIZ_ENABLE | (query << 1) | KILL_HI_Z
        passes = clears + [
            ([('PA_SC_VIZ_QUERY', [0]), ('RB_HIZCONTROL', [hiz])] + reset[0],
             reset[1], reset[2]),
            ([('RB_DEPTHCONTROL', [depthcontrol]),
              ('PA_SC_VIZ_QUERY', [during])], rb.fetch(fetched), draw)]
        viz_queries = {i: i for i in range(IDS)}
        viz_queries[IDS + 1] = query
        suite.test(name, gpugen.resolve_vertex_shader(),
                   constants=rb.CONSTANTS,
                   pixel_shader=rb.pixel_shader(depth=True),
                   program_cntl=queries.PROGRAM_CNTL,
                   registers=rb.registers(0, 0, depth_format=0) + [
                       ('RB_COLORCONTROL', [queries.COLORCONTROL])],
                   passes=passes,
                   after_passes=[('READ_VIZ_STATUS', [0]),
                                 ('WAIT_UNTIL', [WAIT_3D_IDLE]),
                                 ('READ_VIZ_STATUS', [1]),
                                 ('PA_SC_VIZ_QUERY', [0]),
                                 ('RB_DEPTHCONTROL',
                                  [queries.zfunc(queries.ALWAYS)])],
                   fetch_constants=rb.fetch('half'), draw=whole,
                   zpd=len(passes), zpd_reset=True, viz_queries=viz_queries,
                   viz=0x100 | query)
        suite.tests[-1].reads = True
        type(suite.tests[-1]).output_bytes = vizstatus.output_bytes

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
