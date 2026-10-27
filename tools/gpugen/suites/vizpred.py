"""Predicated draws, past what the queries suite tried: a draw with a VIZ
token (DRAW_INDX, 0x100 | id) was never skipped there, with PA_SC_VIZ_QUERY
off for the predicated draw and nothing between the VIZ query's end and the
draw. This suite's queries and draws are the queries suite's (see its
docstring), with these changed, one at a time and together:
- a wait for the 3D engine to be idle (WAIT_UNTIL) between the query and the
  predicated draw, as the query's result may not have reached the command
  processor when the draw does;
- PA_SC_VIZ_QUERY left on, with the query's id or another, during the
  predicated draw (part 2, as no test has had it on there);
- PA_SC_VIZ_QUERY's other bits during the query: kill_pix_post_hi_z (the
  queries suite's) and kill_pix_post_detail_mask;
- RB_HIZCONTROL on, as titles survey.
Part 1 is the changes that leave PA_SC_VIZ_QUERY off at the draw, part 2 the
rest.
The output is an occlusion query's reports over the predicated draw alone
(queries.py): a skipped draw adds no samples. Every VIZ query is over a
draw of the whole 64x32 render target at depth 0.5 (ramp texture), occluded
when the depth test is NEVER and visible when ALWAYS.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpugen  # noqa: E402
import queries  # noqa: E402
import rb  # noqa: E402
import rbdraw  # noqa: E402

VIZ_ENABLE = queries.VIZ_ENABLE
KILL_HI_Z, DETAIL_MASK = 0x80, 0x100
WAIT_3D_IDLE = 0x00020000


def build(suite, part):
    rects = rbdraw.Shapes()
    suite.add_texture('half', queries.texture(lambda x: 0.5))
    suite.add_texture('ramp', queries.texture(lambda x: (x + 0.5) / queries.WIDTH))
    whole = rb.RECT_DRAW
    reset = ([('RB_DEPTHCONTROL', [queries.RESET])], rb.fetch('half'), whole)

    def viz(name, depthcontrol, query=1, during=None, final=0, token=None,
            idle=False, hiz=0, fetched='ramp', draw=whole):
        """A VIZ query `query` over a draw under `depthcontrol` with
        PA_SC_VIZ_QUERY `during`, then the predicated draw (VIZ token
        `token`) with PA_SC_VIZ_QUERY `final`, after an idle wait if asked.
        Every test starts by putting PA_SC_VIZ_QUERY and RB_HIZCONTROL
        back."""
        if during is None:
            during = VIZ_ENABLE | (query << 1) | KILL_HI_Z
        suite.test(name, gpugen.resolve_vertex_shader(),
                   constants=rb.CONSTANTS,
                   pixel_shader=rb.pixel_shader(depth=True),
                   program_cntl=queries.PROGRAM_CNTL,
                   registers=queries.rb.registers(0, 0, depth_format=0) + [
                       ('RB_COLORCONTROL', [queries.COLORCONTROL])],
                   passes=[([('PA_SC_VIZ_QUERY', [0]),
                             ('RB_HIZCONTROL', [hiz])] + reset[0],
                            reset[1], reset[2]),
                           ([('RB_DEPTHCONTROL', [depthcontrol]),
                             ('PA_SC_VIZ_QUERY', [during])],
                            rb.fetch(fetched), draw)],
                   after_passes=(
                       ([('WAIT_UNTIL', [WAIT_3D_IDLE])] if idle else []) +
                       [('PA_SC_VIZ_QUERY', [final]),
                        ('RB_DEPTHCONTROL', [queries.zfunc(queries.ALWAYS)])]),
                   fetch_constants=rb.fetch('half'), draw=whole, zpd=2,
                   viz_queries={1: query},
                   viz=0x100 | query if token is None else token)

    ALWAYS, NEVER, LESS = (queries.zfunc(f) for f in
                           (queries.ALWAYS, queries.NEVER, queries.LESS))
    on = VIZ_ENABLE | (1 << 1)
    if part == 1:
        # The queries suite's own, as a baseline for this build.
        viz('viz visible', ALWAYS)
        viz('viz occluded', NEVER)
        # An idle wait before the predicated draw.
        viz('viz visible idle', ALWAYS, idle=True)
        viz('viz occluded idle', NEVER, idle=True)
        viz('viz less half idle', LESS, idle=True)
        # Other bits of PA_SC_VIZ_QUERY during the query.
        viz('viz occluded no kill idle', NEVER, during=on, idle=True)
        viz('viz occluded detail mask idle', NEVER, during=on | DETAIL_MASK,
            idle=True)
        viz('viz occluded kill and detail mask idle', NEVER,
            during=on | KILL_HI_Z | DETAIL_MASK, idle=True)
        # With the hierarchical depth buffer on.
        viz('viz occluded hiz 0x3 idle', NEVER, hiz=0x3, idle=True)
    else:
        # PA_SC_VIZ_QUERY left on for the predicated draw, which no
        # earlier test did.
        viz('viz visible on', ALWAYS, final=on)
        viz('viz occluded on', NEVER, final=on)
        viz('viz occluded on idle', NEVER, final=on, idle=True)
        viz('viz occluded on id 2', NEVER, final=VIZ_ENABLE | (2 << 1))
        viz('viz occluded hiz 0x13 on idle', NEVER, hiz=0x13, final=on,
            idle=True)
    suite.set_inputs(rects.inputs, 2)
