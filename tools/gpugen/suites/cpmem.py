"""Resolve destinations and EDRAM surfaces off the layouts titles use: the
shared setup of the destbase, destpitch and surfpitch suites (the
resolverect suite's; see its docstring).

Every test draws the 128x64 k_8_8_8_8 render target (pitch set per test)
whole, pixel (x, y) = bytes (x, y, 0x96, 0xC3) in R, G, B, A order, then
Resolves the whole of it to a k_8_8_8_8 tiled texture at the test's output,
so every destination texel names the pixel it holds. The output starts out
the 0xCDCDCDCD sentinel, which marks the texels a Resolve leaves alone. Every
output has `ROOM` bytes past the texture, for texels a Resolve places
beyond it.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpugen  # noqa: E402
import pm4  # noqa: E402
import resolverect  # noqa: E402
import rb  # noqa: E402

WIDTH, HEIGHT = resolverect.WIDTH, resolverect.HEIGHT
K_8888 = resolverect.K_8888
ROOM = 0x8000
WHOLE = (pm4.PRIM_RECTANGLE_LIST, 0, 3)


class Builder:
    def __init__(self, suite):
        self.suite = suite
        self.rects = resolverect.Rects()
        suite.add_texture('coords', resolverect.coords_texture())
        self.fetch = {rb.SLOT: (gpugen.texture_constant(
            resolverect.F32_32_32_32, WIDTH, HEIGHT), 'coords')}

    def test(self, name, dest=(WIDTH, HEIGHT), surface=None, samples=0,
             sample=None, **resolve):
        """A draw into the render target at EDRAM surface pitch `surface`
        (default resolverect's 160) at 2 ** samples samples, then a Resolve
        of it: converting at 1x, raw of sample `sample` otherwise."""
        pitch = resolverect.SURFACE_PITCH if surface is None else surface
        state = resolverect.registers(False, pitch)
        if samples:
            state += [('RB_SURFACE_INFO', [pitch | (samples << 16)]),
                      ('PA_SC_AA_CONFIG', [gpugen.AA_CONFIG[samples]]),
                      ('PA_SU_SC_MODE_CNTL', [gpugen.MODE_CNTL |
                                              gpugen.MSAA_ENABLE])]
            resolve.setdefault('command', 0)
            resolve['sample_select'] = sample
            resolve['after'] = [('PA_SU_SC_MODE_CNTL', [resolverect.MODE_CNTL])]
        resolve.setdefault('registers', ())
        resolve.setdefault('after', ())
        r = gpugen.resolve_registers(
            WIDTH, HEIGHT, K_8888, 4, dest=dest,
            rect=self.rects.rect(0, 0, WIDTH, HEIGHT), **resolve)
        r['bytes'] += ROOM
        if samples:
            # The put-back of the 1x state follows the Resolve.
            r['after'] = list(r['after']) + [
                ('PA_SC_AA_CONFIG', [gpugen.AA_CONFIG[0]]),
                ('RB_SURFACE_INFO', [resolverect.SURFACE_PITCH])]
        self.suite.test(name, gpugen.resolve_vertex_shader(),
                        constants=rb.CONSTANTS, pixel_shader=rb.pixel_shader(),
                        program_cntl=rb.PROGRAM_CNTL, registers=state,
                        fetch_constants=self.fetch, draw=WHOLE, resolve=r)

    def finish(self):
        self.suite.set_inputs(self.rects.inputs, 2)
