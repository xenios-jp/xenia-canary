"""The EXPAND texture formats with the integer number format (num_format 1)
that games set for 16_16_16_16_EXPAND, over every 16-bit value: see
texholeexp, whose textures and vertices this reuses (texholeexp read them
with the fraction format).
"""

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import gpugen  # noqa: E402
import pm4  # noqa: E402
import xenos as x  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    'texholeexp_suite', os.path.join(HERE, 'texholeexp.py'))
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)


def build(suite):
    suite.set_inputs(base.inputs(), 4)
    move = gpugen.export_copy(3)
    vertex = gpugen.fetch_inputs(4)
    fetch = x.tfetch(3, 1, 'xy', base.SLOT, denormalized=True).serialized()
    shader = gpugen.memexport_shader(4, [], [[move]],
                                     fetches=vertex + [fetch])

    def test(name, texture_name, fmt, sign):
        words = gpugen.texture_constant(
            fmt, base.WIDTH, base.ROWS, sign=sign, pitch=base.WIDTH,
            endian=pm4.ENDIAN_NONE, num_format=1)
        suite.test(name, shader,
                   fetch_constants={base.SLOT: (words, texture_name)})

    for fname, fmt, components in (('16_EXPAND', 27, 1),
                                   ('16_16_EXPAND', 28, 2),
                                   ('16_16_16_16_EXPAND', 29, 4)):
        for q in range(4):
            suite.add_texture(f'{fname} q{q}', base.texture(components, q))
            test(f'{fname} integer signed q{q}', f'{fname} q{q}', fmt,
                 base.SIGNED)
            if components == 1 or q == 0:
                test(f'{fname} integer unsigned q{q}', f'{fname} q{q}', fmt,
                     base.UNSIGNED)
