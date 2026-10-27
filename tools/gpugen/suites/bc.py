"""Block-compressed alpha decoding: every endpoint pair and index of a DXT5
(BC3) alpha block, a DXT5A (BC4) block and a DXN (BC5) red block, as 1024x1024
textures of 256x256 blocks. Block (a, z) has endpoints a and z and the index
of pixel p in it is p % 8, so each pair decodes every index twice. Vertex v
point-samples texels 4v to 4v + 3 in rows of 1024 (unnormalized centers) and
exports the decoded channel of each, so a test's results are the decoded
image, row-major. The texture's swizzle puts the channel in X, endian 8in16
as Direct3D's DXT formats; the other channel's blocks are zero. Results are
raw."""

import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gpugen  # noqa: E402
import pm4  # noqa: E402
import xenos as x  # noqa: E402

WIDTH = 1024
SLOT = 1
# (name, TextureFormat, block bytes, swizzle of the channel into X)
FORMATS = [('DXT5 alpha', 20, 16, 3), ('DXT5A red', 59, 8, 0),
           ('DXN red', 49, 16, 0)]
f2w = gpugen.f2w


def texture(block_bytes):
    """Rows of 256 blocks of big-endian 16-bit words, the alpha (or red)
    block first: a | z << 8, then the 48-bit indices."""
    indices = sum((p % 8) << (3 * p) for p in range(16))
    alpha = [indices & 0xFFFF, (indices >> 16) & 0xFFFF, indices >> 32]
    words = []
    for z in range(256):
        for a in range(256):
            block = [a | (z << 8)] + alpha + [0] * (block_bytes // 2 - 4)
            words += block
    data = struct.pack(f'>{len(words)}H', *words)
    return list(struct.unpack(f'>{len(data) // 4}I', data))


def build(suite):
    suite.set_generated(WIDTH * WIDTH // 4)
    suite.raw = True
    k = gpugen.C_FIRST_FREE
    # r1 = (v / 256, 4v); r1.x = y = floor(v / 256); r2 = (x0 = 4v - 1024y,
    # y); r3 + i = (x0 + i + 0.5, y + 0.5), exact.
    body = [x.alu(vector_op=x.V_MUL, vector_dest=1, vector_mask=0x3,
                  src1=x.r(0, 'xxxx'), src2=x.c(k)),
            x.alu(vector_op=x.V_FLOOR, vector_dest=1, vector_mask=0x1,
                  src1=x.r(1), src2=x.r(1)),
            x.alu(vector_op=x.V_MAD, vector_dest=2, vector_mask=0x1,
                  src1=x.r(1, 'xxxx'), src2=x.c(k + 1), src3=x.r(1, 'yyyy')),
            x.alu(vector_op=x.V_MAX, vector_dest=2, vector_mask=0x2,
                  src1=x.r(1, 'xxxx'), src2=x.r(1, 'xxxx'))]
    body += [x.alu(vector_op=x.V_ADD, vector_dest=3 + i, vector_mask=0x3,
                   src1=x.r(2), src2=x.c(k + 2 + i)) for i in range(4)]
    constants = {k: [f2w(1 / 256), f2w(4.0), 0, 0],
                 k + 1: [f2w(-1024.0), 0, 0, 0]}
    for i in range(4):
        constants[k + 2 + i] = [f2w(i + 0.5), f2w(0.5), 0, 0]
    # r7 lane i = channel of texel i.
    fetches = [x.tfetch(7, 3 + i, 'xy', SLOT, denormalized=True,
                        dst_swizzle='____'[:i] + 'x' + '____'[i + 1:])
               for i in range(4)]
    fetches[0] = fetches[0].serialized()
    for name, fmt, block_bytes, channel in FORMATS:
        suite.add_texture(name, texture(block_bytes))
        swizzle = channel | (channel << 3) | (channel << 6) | (5 << 9)
        suite.test(name, gpugen.memexport_shader(
            0, [], [[gpugen.export_copy(7)]], fetches=fetches, prologue=body),
            constants=constants,
            fetch_constants={SLOT: (gpugen.texture_constant(
                fmt, WIDTH, WIDTH, endian=pm4.ENDIAN_8IN16,
                swizzle=swizzle), name)})
