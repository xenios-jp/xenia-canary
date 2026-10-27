"""Packed mip tails when the fetch constant's mip address is its base
address, as Halo 3 and Reach set the textures that stream (levels 1-5, 1-8,
1-9, 1-10 packed with no level 0 resident) and tiny textures whose tail is
all there is: which memory a point-sampled fetch at an explicit LOD
(setTexLOD, so no derivatives) reads.

The texture describes its own memory as in tailmips: block (or texel) u of
the one allocation holds the tag u + 2^23. The tag takes 1-3 fetches of the
same layout over differently filled memory (a plane each, slots 1-3): k_8
returns 8 bits of it in R a plane, k_8_8 16 in R and G, DXT1 16 bits in c0
(R5 G6 B5), DXN 16 bits in R and G, k_8_8_8_8 all 32 in RGBA; every format
is read with endian none.

Vertex v (78): r1 = (u, v, lod, 0), lod = -1 .. 11 and the point 0-5 of
(.003, .003), (.997, .003), (.003, .997), (.997, .997), (.4321, .6789) and
(.7123, .2345), normalized. Each test exports one float4 a plane.
"""

import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gpugen  # noqa: E402
import pm4  # noqa: E402
import xenos as x  # noqa: E402

LODS = list(range(-1, 12))
POINTS = ((.003, .003), (.997, .003), (.003, .997), (.997, .997),
          (.4321, .6789), (.7123, .2345))
VERTICES = len(LODS) * len(POINTS)
REGION_BIT = 1 << 23
POINT = (0, 0, 0)
f2w = gpugen.f2w

# name -> (TextureFormat, bytes a block, block width, planes)
FORMATS = {'k_8': (2, 1, 1, 3), 'k_8_8': (10, 2, 1, 2),
           'k_8_8_8_8': (6, 4, 1, 1), 'DXT1': (18, 8, 4, 2),
           'DXN': (49, 16, 4, 2)}


def inputs():
    words = []
    for lod in LODS:
        for u, v in POINTS:
            words += [f2w(u), f2w(v), f2w(float(lod)), 0]
    return words


def align(value, to):
    return (value + to - 1) // to * to


def pot(value):
    return 1 << (value - 1).bit_length()


def region_blocks(fmt, width, height, tiled):
    """(base, mips) blocks to allocate, a little over what the layout takes:
    levels from 1 to the packed one, each 32x32-block aligned (linear rows
    to 256 bytes) and 4 KiB aligned."""
    _, bytes_, bw, _ = FORMATS[fmt]
    row_align = 32 if tiled else max(256 // bytes_, 32)
    pitch = align(align(width, 32) // bw, row_align)
    rows = align(align(height, 32) // bw, 32)
    base = pitch * rows + 32 * 32
    packed = max((min(width - 1, height - 1)).bit_length() - 4, 0)
    total = 0
    for level in range(1, max(packed, 1) + 2):
        w = align(max(pot(width) >> level, 1) // bw or 1, row_align)
        h = align(max(pot(height) >> level, 1) // bw or 1, 32)
        total += align(w * h * bytes_, 4096) // bytes_
    return base, total + 4096 // bytes_


def plane_words(fmt, plane, units, region):
    """The words of `units` blocks tagged `unit | region` in plane `plane`."""
    _, bytes_, bw, _ = FORMATS[fmt]
    data = bytearray()
    for unit in range(units):
        tag = unit | region
        if fmt == 'k_8':
            data.append(tag >> (8 * plane) & 0xFF)
        elif fmt == 'k_8_8':
            data += bytes([tag >> (16 * plane) & 0xFF,
                           tag >> (16 * plane + 8) & 0xFF])
        elif fmt == 'k_8_8_8_8':
            data += struct.pack('<I', tag)
        elif fmt == 'DXT1':
            data += struct.pack('<HHI', tag >> (16 * plane) & 0xFFFF, 0, 0)
        elif fmt == 'DXN':
            low = tag >> (16 * plane) & 0xFF
            high = tag >> (16 * plane + 8) & 0xFF
            data += bytes([low, 0, 0, 0, 0, 0, 0, 0,
                           high, 0, 0, 0, 0, 0, 0, 0])
    data += bytes(-len(data) % 4)
    return list(struct.unpack(f'>{len(data) // 4}I', data))


def constant(fmt, width, height, tiled, levels, mip_address):
    fmt_id = FORMATS[fmt][0]
    words = gpugen.texture_constant(
        fmt_id, width, height, filters=POINT, pitch=width, mips=levels,
        packed_mips=True, endian=pm4.ENDIAN_NONE)
    if tiled:
        words[0] |= 1 << 31
    return words


# (format, width, height, tiled, (min, max) level, mip address)
def configs():
    out = []

    def add(fmt, size, ranges, tiled=True):
        width, height = size
        for levels in ranges:
            out.append((fmt, width, height, tiled, levels, True))
    add('DXN', (128, 128), [(1, 5)])
    add('DXN', (256, 128), [(1, 8)])
    add('DXN', (512, 512), [(1, 9)])
    add('DXT1', (256, 256), [(1, 8), (1, 2)])
    add('DXT1', (32, 64), [(1, 4)])
    add('DXT1', (32, 32), [(1, 3), (1, 1)])
    add('DXT1', (1024, 1024), [(1, 10)])
    add('k_8_8', (128, 128), [(1, 7)])
    add('k_8_8', (32, 32), [(1, 1)])
    add('k_8', (256, 256), [(1, 8)])
    add('k_8_8_8_8', (64, 64), [(1, 6), (2, 4)])
    # Tiny textures: the whole chain is the tail, at the base address.
    add('k_8_8_8_8', (16, 16), [(0, 4), (0, 3)])
    add('DXT1', (4, 4), [(0, 2)])
    add('DXT1', (16, 16), [(0, 3), (0, 4)])
    add('k_8_8_8_8', (8, 8), [(0, 3)])
    add('DXT1', (32, 64), [(1, 4)], tiled=False)
    return out


def build(suite):
    suite.set_inputs(inputs(), 4)
    vertex = gpugen.fetch_inputs(4)
    shaders = {}
    made = set()

    def shader(planes):
        if planes not in shaders:
            fetches = list(vertex)
            for plane in range(planes):
                fetches.append(x.set_texture_state(
                    x.FETCH_SET_LOD, 1, 'z').serialized())
                fetches.append(x.tfetch(
                    3 + plane, 1, 'xy', 1 + plane, use_reg_lod=True,
                    use_comp_lod=False))
            shaders[planes] = gpugen.memexport_shader(
                4, [], [[gpugen.export_copy(3 + plane)]
                        for plane in range(planes)], fetches=fetches)
        return shaders[planes]

    for fmt, width, height, tiled, levels, mip in configs():
        planes = FORMATS[fmt][3]
        layout = 'T' if tiled else 'L'
        shape = f'{fmt} {width}x{height} {layout}'
        base_units, mip_units = region_blocks(fmt, width, height, tiled)
        units = mip_units + (base_units if levels[0] == 0 else 0)
        shape = shape + (' mips only' if levels[0] else '')
        for plane in range(planes):
            if (shape, plane) in made:
                continue
            made.add((shape, plane))
            suite.add_texture(f'{shape} p{plane}',
                              plane_words(fmt, plane, units, REGION_BIT))
        fetch_constants = {}
        for plane in range(planes):
            words = constant(fmt, width, height, tiled, levels, mip)
            fetch_constants[1 + plane] = (
                words, (f'{shape} p{plane}', f'{shape} p{plane}'))
        name = f'{shape} levels {levels[0]}-{levels[1]} packed same address'
        suite.test(name, shader(planes), elements=planes,
                   fetch_constants=fetch_constants)
