"""Occlusion and VIZ queries: the sample counts EVENT_WRITE_ZPD reports, as
Direct3D's occlusion queries begin and end them, and which draws a VIZ query
lets through when they're predicated on it.

Every test draws a 64x32 k_8_8_8_8 render target with a D24S8 depth buffer
(the rb suite's, surface pitch 80) in rectangles of the rb suite's pixel
shader also exporting oDepth = the texel's X, from 64x32 textures:
- 'half': depth 0.5 everywhere;
- 'ramp': depth (x + 0.5) / 64 at pixel x, so half the pixels are nearer
  than 0.5.
The first pass resets every pixel to depth 0.5, stencil 0x5A. The output is
the report at its start (+32) and at its end (+0) of an occlusion query over
the passes after the reset ('zpd' tests) or over the draw after a VIZ query
('viz' tests: the query's draw is the second pass, between VIZ_QUERY begin
and end with PA_SC_VIZ_QUERY on, and the test's draw is predicated on it).
A report is 8 little-endian dwords, Total, ZFail, ZPass and StencilFail
each in two halves A and B, from counters that run freely across reports.

A VIZ query with PA_SC_VIZ_QUERY off wedged the console's GPU. A report with
RB_SAMPLE_COUNT_CTL 0 or 1 (no copy) made every report after it lag a draw,
and after a run with them went back to the dashboard, it couldn't launch
titles. No test has either.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpugen  # noqa: E402
import pm4  # noqa: E402
import rb  # noqa: E402
import rbdraw  # noqa: E402

WIDTH, HEIGHT = rb.WIDTH, rb.HEIGHT
F32_32_32_32 = 38
# RB_DEPTHCONTROL fields.
STENCIL, Z, Z_WRITE = 1, 2, 4
NEVER, LESS, LEQUAL, GREATER, ALWAYS = 0, 1, 3, 4, 7
KEEP, REPLACE = 0, 2
RESET = (STENCIL | Z | Z_WRITE | (ALWAYS << 4) | (ALWAYS << 8) |
         (REPLACE << 14))
PROGRAM_CNTL = (rb.PROGRAM_CNTL & ~(0xF << 27)) | (3 << 27)  # oDepth
VIZ_ENABLE, KILL_PIX_POST_EARLY_Z = 1, 0x80
COLORCONTROL = 0x87000007  # Direct3D's default: alpha test off, ALWAYS
f2w = gpugen.f2w


def zfunc(func, write=False):
    return Z | (Z_WRITE if write else 0) | (func << 4)


def stencil(func):
    return STENCIL | (func << 8)


def texture(depth):
    return [f2w(v) for y in range(HEIGHT) for x in range(WIDTH)
            for v in (depth(x), 0.25, 0.5, 1.0)]


def build(suite):
    rects = rbdraw.Shapes()
    suite.add_texture('half', texture(lambda x: 0.5))
    suite.add_texture('ramp', texture(lambda x: (x + 0.5) / WIDTH))
    whole = rb.RECT_DRAW
    nothing = (pm4.PRIM_RECTANGLE_LIST, 0, 0)

    def state(samples):
        # RB_COLORCONTROL: Direct3D's default, as a test may change it.
        return rb.registers(0, samples, depth_format=0) + [
            ('RB_COLORCONTROL', [COLORCONTROL])]

    reset = ([('RB_DEPTHCONTROL', [RESET])], rb.fetch('half'), whole)

    def zpd(name, depthcontrol, fetched='ramp', draw=whole, passes=(),
            samples=0, registers=(), control=(2, 2), zero=False):
        """An occlusion query over `passes` (depthcontrol, texture, draw)
        then the test's draw."""
        suite.test(name, gpugen.resolve_vertex_shader(),
                   constants=rb.CONSTANTS,
                   pixel_shader=rb.pixel_shader(depth=True),
                   program_cntl=PROGRAM_CNTL, registers=state(samples),
                   passes=[reset] + [
                       ([('RB_DEPTHCONTROL', [control])], rb.fetch(t), d)
                       for control, t, d in passes],
                   after_passes=[('RB_DEPTHCONTROL', [depthcontrol])] +
                   list(registers),
                   fetch_constants=rb.fetch(fetched), draw=draw, zpd=1,
                   zpd_control=control, zpd_reset=zero)

    # RB_SAMPLE_COUNT_CTL: its copy (bit 1) and reset after it (bit 0). The
    # first test resets the free-running counters, so the counts after it
    # are the suite's own.
    zpd('zpd after reset', zfunc(ALWAYS), zero=True)
    for control in ((3, 2), (2, 3)):
        zpd(f'zpd control {control[0]} then {control[1]}', zfunc(ALWAYS),
            control=control)

    # What a query counts.
    zpd('zpd nothing drawn', zfunc(ALWAYS), draw=nothing)
    zpd('zpd always', zfunc(ALWAYS))
    zpd('zpd never', zfunc(NEVER))
    zpd('zpd less half', zfunc(LESS))
    zpd('zpd greater half', zfunc(GREATER))
    zpd('zpd less half written', zfunc(LESS, write=True))
    zpd('zpd depth test off', 0)
    zpd('zpd stencil never', zfunc(ALWAYS) | stencil(NEVER))
    zpd('zpd stencil never depth never', zfunc(NEVER) | stencil(NEVER))
    zpd('zpd 16x8', zfunc(ALWAYS), draw=rects.rectangles([(8, 8, 24, 16)]))
    zpd('zpd two draws', zfunc(LESS), passes=[(zfunc(ALWAYS), 'ramp', whole)])
    zpd('zpd color mask 0', zfunc(ALWAYS),
        registers=[('RB_COLOR_MASK', [0])])
    zpd('zpd depth only mode', zfunc(LESS),
        registers=[('RB_MODECONTROL', [5])])
    zpd('zpd alpha test never', zfunc(ALWAYS),
        registers=[('RB_COLORCONTROL', [COLORCONTROL & ~0xF | 0x8 | NEVER])])
    for samples in (1, 2):
        zpd(f'zpd {1 << samples}x always', zfunc(ALWAYS), samples=samples)
        zpd(f'zpd {1 << samples}x less half', zfunc(LESS), samples=samples)

    def viz(name, depthcontrol, fetched='ramp', draw=whole, query=1,
            viz_query=None, token=None, counted=2, hiz=0):
        """A VIZ query `query` over a draw under `depthcontrol`, then a draw
        of everything predicated on it (VIZ token `token`), an occlusion
        query over the passes from `counted` (2 the predicated draw alone, 1
        both)."""
        if viz_query is None:
            viz_query = VIZ_ENABLE | (query << 1) | KILL_PIX_POST_EARLY_Z
        hizcontrol = [('RB_HIZCONTROL', [hiz])]
        suite.test(name, gpugen.resolve_vertex_shader(),
                   constants=rb.CONSTANTS,
                   pixel_shader=rb.pixel_shader(depth=True),
                   program_cntl=PROGRAM_CNTL, registers=state(0),
                   passes=[(hizcontrol + reset[0],) + reset[1:],
                           ([('RB_DEPTHCONTROL', [depthcontrol]),
                             ('PA_SC_VIZ_QUERY', [viz_query])],
                            rb.fetch(fetched), draw)],
                   after_passes=[('PA_SC_VIZ_QUERY', [0]),
                                 ('RB_HIZCONTROL', [0]),
                                 ('RB_DEPTHCONTROL', [zfunc(ALWAYS)])],
                   fetch_constants=rb.fetch('half'), draw=whole,
                   zpd=counted, viz_queries={1: query},
                   viz=0x100 | query if token is None else token)

    # Which queries a predicated draw passes.
    viz('viz visible', zfunc(ALWAYS))
    viz('viz visible counted', zfunc(ALWAYS), counted=1)
    viz('viz occluded', zfunc(NEVER))
    viz('viz occluded counted', zfunc(NEVER), counted=1)
    viz('viz nothing drawn', zfunc(ALWAYS), draw=nothing)
    viz('viz less half', zfunc(LESS))
    viz('viz one pixel', zfunc(ALWAYS), draw=rects.rectangles([(5, 3, 6, 4)]))
    viz('viz stencil never', zfunc(ALWAYS) | stencil(NEVER))
    viz('viz depth test off', 0)
    viz('viz visible kill_pix off', zfunc(ALWAYS),
        viz_query=VIZ_ENABLE | (1 << 1))
    viz('viz visible kill_pix off counted', zfunc(ALWAYS),
        viz_query=VIZ_ENABLE | (1 << 1), counted=1)
    viz('viz occluded draw not predicated', zfunc(NEVER), token=1)
    viz('viz occluded id 63', zfunc(NEVER), query=63)
    viz('viz visible id 63', zfunc(ALWAYS), query=63)
    viz('viz visible query id 2 draw on 1', zfunc(ALWAYS), query=2,
        token=0x101)
    # With the hierarchical depth buffer on, as titles query: written by the
    # reset, tested by the query's draw.
    for hiz in (0x3, 0x13):
        viz(f'viz occluded hiz {hiz:#x}', zfunc(NEVER), hiz=hiz)
        viz(f'viz visible hiz {hiz:#x}', zfunc(ALWAYS), hiz=hiz)
        viz(f'viz nothing drawn hiz {hiz:#x}', zfunc(ALWAYS), draw=nothing,
            hiz=hiz)
        viz(f'viz less half hiz {hiz:#x}', zfunc(LESS), hiz=hiz)
        viz(f'viz greater nearer hiz {hiz:#x}', zfunc(GREATER), 'half',
            hiz=hiz)
    suite.set_inputs(rects.inputs, 2)
