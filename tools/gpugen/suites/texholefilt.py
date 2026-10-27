"""The filter combinations games set that tmips left: mip basemap with
linear texel filters, mip point and basemap with anisotropy, mag/min filters
that differ, LOD biases between -128 and 74, and the fetch instruction's
filter and anisotropy overriding the constant's.

The inputs, textures and fetch are tmips's (the same vertices: r1 the
coordinate and LOD, r2 the gradients; 'addr' and 'tags' 64x64 textures with
levels 0-6 packed, linear), see there.
"""

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import gpugen  # noqa: E402
import xenos as x  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    'tmips_suite', os.path.join(HERE, 'tmips.py'))
tmips = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tmips)

SLOT = 1
POINT, LINEAR, BASEMAP = 0, 1, 2
F32, F8888 = 36, 6
f2w = gpugen.f2w


def build(suite):
    suite.set_inputs(tmips.inputs(), 8)
    suite.add_texture('addr base', [f2w(float(d))
                                    for d in range(tmips.BASE_DWORDS)])
    suite.add_texture('addr mips', [f2w(float(tmips.MIP_TAG + d))
                                    for d in range(tmips.MIP_DWORDS)])
    suite.add_texture('tags base', tmips.tags(tmips.BASE_DWORDS, 0x40))
    suite.add_texture('tags mips', tmips.tags(tmips.MIP_DWORDS, 0xC0))
    vertex = gpugen.fetch_inputs(8)
    move = gpugen.export_copy(3)
    def test(name, texture, fmt, filters, levels=(0, 6), gradients=False,
             fetch=None, **constant):
        words = gpugen.texture_constant(
            fmt, 64, 64, filters=filters, pitch=64, mips=levels,
            packed_mips=True, **constant)
        keywords = dict(use_reg_gradients=True) if gradients else dict(
            use_reg_lod=True, use_comp_lod=False)
        keywords.update(fetch or {})
        if gradients:
            sets = [x.set_texture_state(x.FETCH_SET_GRADIENTS_HORZ, 2,
                                        'xy').serialized(),
                    x.set_texture_state(x.FETCH_SET_GRADIENTS_VERT, 2, 'zw')]
        else:
            sets = [x.set_texture_state(x.FETCH_SET_LOD, 1, 'z').serialized()]
        fetches = vertex + sets + [x.tfetch(3, 1, 'xy', tmips.SLOT,
                                            **keywords)]
        suite.test(name, gpugen.memexport_shader(8, [], [[move]],
                                                 fetches=fetches),
                   fetch_constants={tmips.SLOT: (
                       words, (f'{texture} base', f'{texture} mips'))})

    bilinear_basemap = (LINEAR, LINEAR, BASEMAP)
    # Mip basemap with linear texel filters.
    test('tags basemap bilinear', 'tags', F8888, bilinear_basemap)
    test('tags basemap bilinear levels 2-4', 'tags', F8888,
         bilinear_basemap, levels=(2, 4))
    test('tags basemap bilinear levels 1-5', 'tags', F8888,
         bilinear_basemap, levels=(1, 5))
    test('addr basemap levels 2-4', 'addr', F32, (POINT, POINT, BASEMAP),
         levels=(2, 4))
    test('addr basemap levels 1-5', 'addr', F32, (POINT, POINT, BASEMAP),
         levels=(1, 5))
    test('tags basemap bilinear gradients', 'tags', F8888, bilinear_basemap,
         gradients=True)
    # Mixed mag and min filters with a mip filter.
    test('tags mag linear min point mip linear', 'tags', F8888,
         (LINEAR, POINT, LINEAR))
    test('tags mag point min linear mip linear', 'tags', F8888,
         (POINT, LINEAR, LINEAR))
    test('tags mag linear min point mip linear gradients', 'tags', F8888,
         (LINEAR, POINT, LINEAR), gradients=True)
    test('tags mag point min linear mip linear gradients', 'tags', F8888,
         (POINT, LINEAR, LINEAR), gradients=True)
    # Anisotropy with bilinear and mip point, or mip basemap.
    for aniso in range(6):
        test(f'tags bilinear mip point gradients aniso {aniso}', 'tags',
             F8888, (LINEAR, LINEAR, POINT), gradients=True, aniso=aniso)
        test(f'tags basemap gradients aniso {aniso}', 'tags', F8888,
             bilinear_basemap, gradients=True, aniso=aniso)
        test(f'addr mip point gradients aniso {aniso}', 'addr', F32,
             (POINT, POINT, POINT), gradients=True, aniso=aniso)
    # LOD biases games set (1/32 of a level), trilinear.
    for bias in (10, 16, 20, 26, 32, 48, 74, -21, -48, -128):
        test(f'tags trilinear const lod bias {bias}', 'tags', F8888,
             (LINEAR, LINEAR, LINEAR), lod_bias=bias)
        test(f'addr const lod bias {bias}', 'addr', F32,
             (POINT, POINT, POINT), lod_bias=bias)
    # The fetch instruction's filters and anisotropy over the constant's.
    trilinear = (LINEAR, LINEAR, LINEAR)
    test('tags point constant, fetch bilinear mip point', 'tags', F8888,
         (POINT, POINT, POINT),
         fetch=dict(mag_filter=LINEAR, min_filter=LINEAR, mip_filter=POINT))
    test('tags trilinear constant, fetch mip point', 'tags', F8888,
         trilinear, fetch=dict(mip_filter=POINT))
    test('tags trilinear constant, fetch mip basemap', 'tags', F8888,
         trilinear, fetch=dict(mip_filter=BASEMAP))
    test('tags trilinear constant, fetch point min', 'tags', F8888,
         trilinear, fetch=dict(min_filter=POINT))
    test('tags trilinear constant, fetch point mag', 'tags', F8888,
         trilinear, fetch=dict(mag_filter=POINT))
    for aniso in range(6):
        test(f'tags trilinear gradients constant aniso 0, fetch aniso '
             f'{aniso}', 'tags', F8888, trilinear, gradients=True,
             fetch=dict(aniso_filter=aniso))
        test(f'tags trilinear gradients constant aniso 3, fetch aniso '
             f'{aniso}', 'tags', F8888, trilinear, gradients=True, aniso=3,
             fetch=dict(aniso_filter=aniso))
