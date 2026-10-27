"""Cube map fetches with exp_adjust 2 (Halo: Reach reads its 32x32 8_8_8_8
cubes so), from vertex shaders: texcube's inputs, textures and face fetch,
the fetch constant's exp_adjust set to 2 (and 3, -1), point and linear,
unsigned and signed; see texcube.
"""

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import gpugen  # noqa: E402
import xenos as x  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    'texcube_suite', os.path.join(HERE, 'texcube.py'))
texcube = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(texcube)


def build(suite):
    suite.set_inputs(texcube.inputs(), 8)
    suite.add_texture('addr', texcube.addr())
    suite.add_texture('tags', texcube.tags())
    vertex = gpugen.fetch_inputs(8)

    def face_test(name, texture, fmt, filters, adjust, sign=0):
        words = texcube.cube_constant(fmt, filters, 2)
        words[3] |= (adjust & 0x3F) << 13
        words[0] |= sign * 0x55 << 2
        fetch = x.tfetch(5, 1, 'xyz', texcube.SLOT, dimension=3).serialized()
        suite.test(name, gpugen.memexport_shader(
            8, [], [[gpugen.export_copy(5)]], fetches=vertex + [fetch]),
            fetch_constants={texcube.SLOT: (words, texture)})

    for adjust in (2, 3, -1):
        face_test(f'face point exp_adjust {adjust}', 'addr', texcube.F32,
                  texcube.POINT, adjust)
        face_test(f'face linear exp_adjust {adjust}', 'tags', texcube.F8888,
                  texcube.LINEAR, adjust)
    face_test('face point exp_adjust 2 signed', 'tags', texcube.F8888,
              texcube.POINT, 2, sign=1)
    face_test('face linear exp_adjust 2 signed', 'tags', texcube.F8888,
              texcube.LINEAR, 2, sign=1)
