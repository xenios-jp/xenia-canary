"""Mip chains without a packed tail (the fetch constant's packed_mips clear),
as games set them: levels 0-3, 1-1, 2-2, 3-3, 4-4 and full chains. Which
memory a point-sampled fetch at an explicit LOD (setTexLOD, so no
derivatives) reads, for tiled and linear 2D textures of 1, 4, 8 bytes a
texel (k_8, k_8_8_8_8, k_16_16_16_16) and 8 a block (DXT1), with a mip
address.

The textures describe their own memory as in tailmips: block (or texel) u of
the base allocation holds the tag u, of the mip allocation u + 2^23, so a
fetch's result says which block of which allocation it read. The tag takes
1-3 fetches of the same layout over differently filled memory (a plane
each, slots 1-3): k_8 returns 8 bits of it in R a plane, k_16_16_16_16 16 in
R and 16 in G, DXT1 16 bits in c0 (R5 G6 B5) with c1 = 0 and the indices 0
(planes 0 and 1), k_8_8_8_8 all 32 in RGBA; every format is read with endian
none.

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
FORMATS = {'k_8': (2, 1, 1, 3), 'k_8_8_8_8': (6, 4, 1, 1),
           'k_16_16_16_16': (26, 8, 1, 1), 'DXT1': (18, 8, 4, 2)}


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
    base = pitch * rows + 32 * 32 + 4096 // bytes_
    packed = max(width - 1, height - 1).bit_length()
    total = 0
    for level in range(1, packed + 1):
        w = align(max(pot(width) >> level, 1) // bw or 1, row_align)
        h = align(max(pot(height) >> level, 1) // bw or 1, 32)
        total += align(w * h * bytes_, 4096) // bytes_
    return base, total + 8192 // bytes_


def plane_words(fmt, plane, units, region):
    """The words of `units` blocks tagged `unit | region` in plane `plane`."""
    _, bytes_, bw, _ = FORMATS[fmt]
    data = bytearray()
    for unit in range(units):
        tag = unit | region
        if fmt == 'k_8':
            data.append(tag >> (8 * plane) & 0xFF)
        elif fmt == 'k_8_8_8_8':
            data += struct.pack('<I', tag)
        elif fmt == 'k_16_16_16_16':
            data += struct.pack('<HHHH', tag & 0xFFFF, tag >> 16, 0, 0)
        elif fmt == 'DXT1':
            data += struct.pack('<HHI', tag >> (16 * plane) & 0xFFFF, 0, 0)
    data += bytes(-len(data) % 4)
    return list(struct.unpack(f'>{len(data) // 4}I', data))


def constant(fmt, width, height, tiled, levels, mip_address):
    fmt_id = FORMATS[fmt][0]
    words = gpugen.texture_constant(
        fmt_id, width, height, filters=POINT, pitch=width, mips=levels,
        packed_mips=False, endian=pm4.ENDIAN_NONE)
    if tiled:
        words[0] |= 1 << 31
    return words


# (format, width, height, tiled, (min, max) level, mip address)
def configs():
    out = []

    def add(fmt, size, ranges, tiled=True, mip=True):
        width, height = size
        for levels in ranges:
            out.append((fmt, width, height, tiled, levels, mip))
    # GTA IV's signed 16_16_16_16 chain (the tags are read unsigned here).
    add('k_16_16_16_16', (512, 256), [(0, 3), (1, 1), (2, 2), (0, 9)])
    add('k_16_16_16_16', (256, 256), [(0, 3), (1, 1), (2, 2)])
    add('k_8_8_8_8', (128, 64), [(0, 5), (1, 1), (2, 2), (3, 3), (4, 4),
                                  (0, 3)])
    add('k_8_8_8_8', (64, 64), [(0, 3), (1, 1), (2, 2), (0, 6)])
    add('k_8_8_8_8', (192, 192), [(0, 3), (1, 1), (2, 2), (0, 7)])
    add('k_8', (32, 32), [(0, 3), (1, 1), (2, 2)])
    add('k_8', (64, 64), [(0, 3), (1, 1), (2, 2)])
    add('k_8', (96, 96), [(0, 3), (1, 1), (2, 2)])
    add('k_8_8_8_8', (256, 256), [(0, 3), (0, 8)])
    add('DXT1', (256, 256), [(0, 3), (0, 8)])
    add('DXT1', (512, 512), [(1, 1), (2, 2)])
    add('k_8_8_8_8', (64, 64), [(0, 3)], tiled=False)
    add('DXT1', (256, 256), [(0, 3)], tiled=False)
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
        for plane in range(planes):
            if (shape, plane) in made:
                continue
            made.add((shape, plane))
            suite.add_texture(f'{shape} p{plane} base',
                              plane_words(fmt, plane, base_units, 0))
            suite.add_texture(f'{shape} p{plane} mips',
                              plane_words(fmt, plane, mip_units, REGION_BIT))
        fetch_constants = {}
        for plane in range(planes):
            words = constant(fmt, width, height, tiled, levels, mip)
            fetch_constants[1 + plane] = (
                words, (f'{shape} p{plane} base',
                        f'{shape} p{plane} mips' if mip else None))
        name = (f'{shape} levels {levels[0]}-{levels[1]}'
                f'{"" if mip else " no mip address"}')
        suite.test(name, shader(planes), elements=planes,
                   fetch_constants=fetch_constants)
