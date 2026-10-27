"""Vertex DXT3A sampling across a large non-power-of-two atlas.

The 891x1023 atlas has a 896-texel pitch, unsigned 8in16 DXT3A data,
identity swizzle, point filtering and base-level LOD, matching the fetch
configuration of a terrain blend-weight atlas. All data is synthetic.
Normalized tiled sampling is paired with unnormalized and linear-layout
controls. Coordinates bracket texel, block, macro-tile and clamp edges;
each invocation exports the fetched float4 to its own memory element.
"""
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gpugen  # noqa: E402
import play  # noqa: E402
import xenos as x  # noqa: E402

WIDTH, HEIGHT, PITCH = 891, 1023, 896
SLOT = 1


def atlas(tiled):
    blocks_x, blocks_y = PITCH // 4, (HEIGHT + 3) // 4
    offsets = [play.tiled2d(bx, by, blocks_x, 3) if tiled else
               (by * blocks_x + bx) * 8
               for by in range(blocks_y) for bx in range(blocks_x)]
    data = bytearray(max(offsets) + 8)
    for by in range(blocks_y):
        for bx in range(blocks_x):
            value = 0
            for y in range(4):
                for x_ in range(4):
                    px, py = 4 * bx + x_, 4 * by + y
                    alpha = (3 * px + 5 * py + 7 * (px >> 7) +
                             11 * (py >> 7)) & 15
                    value |= alpha << (4 * (4 * y + x_))
            block = struct.pack('<Q', value)
            # The fetch swaps bytes within each 16-bit word.
            block = b''.join(block[i:i + 2][::-1] for i in range(0, 8, 2))
            at = offsets[by * blocks_x + bx]
            data[at:at + 8] = block
    return gpugen.texture_words(data)


def build(suite):
    xs = (-1, 0, 3, 4, 127, 128, 255, 256, 511, 512, 767, 768,
          889, 890, 891, 892)
    ys = (-1, 0, 3, 4, 127, 128, 255, 256, 511, 512, 767, 768,
          1021, 1022, 1023, 1024)
    words = []
    for phase in (-1 / 512, 1 / 512, .5, 1 - 1 / 512):
        for y in ys:
            for x_ in xs:
                px, py = x_ + phase, y + phase
                words.extend(gpugen.f2w(v) for v in
                             (px / WIDTH, py / HEIGHT, px, py))
    suite.set_inputs(words, 4)
    for tiled, normalized in ((True, True), (True, False), (False, True)):
        name = ('tiled' if tiled else 'linear') + ' ' + (
            'normalized' if normalized else 'unnormalized')
        texture_name = 'atlas tiled' if tiled else 'atlas linear'
        if normalized:
            suite.add_texture(texture_name, atlas(tiled))
        fetch = x.tfetch(3, gpugen.INPUT_REGISTER,
                         'xy' if normalized else 'zw', SLOT,
                         denormalized=not normalized,
                         use_comp_lod=False).serialized()
        shader = gpugen.memexport_shader(
            4, [], [[gpugen.export_copy(3)]],
            fetches=gpugen.fetch_inputs(4) + [fetch])
        # Keep all non-address bits of the terrain fetch constant explicit.
        constant = [0x87004802 if tiled else 0x07004802, 0x7A,
                    0x007FC37A, 0x01000D10, 0, 0x218]
        suite.test(name, shader,
                   fetch_constants={SLOT: (constant, texture_name)})
