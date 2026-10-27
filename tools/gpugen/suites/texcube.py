"""Cube map fetches from vertex shaders: which face and texel tfetchCube
reads for given (s, t, face) and for directions through Direct3D's cube
sequence, and how it clamps and filters at face edges.

The textures describe their own memory: an 8x8 cube, pitch 32 texels, over
8 blocks of 4 KiB (Xenia's face stride; the console may use another), where
dword d holds
- 'addr' (k_32_FLOAT, point sampled): d;
- 'tags' (k_8_8_8_8): bytes (8 (d % 32), 8 (d // 32 % 32), 32 (d // 1024),
  0xFF) in x, y, z, w order: column, row and 4 KiB block, so a filtered x
  or y is a fraction between texels and z a blend between blocks.

Vertex v (1024):
- r1 = (s, t, face, 0) for tfetchCube directly: v < 768: face v // 128,
  s = 0.8 + 0.09 (v % 16), t = 0.85 + 0.18 (v // 16 % 8); v < 896: texel
  edges on faces 0 and 3, s (then t) = 1 + n / 8 + (-2^-21, 0, 2^-21,
  2^-12)[v % 4], n = v // 4 % 16 - 4, the other 1.4375; v >= 896: face
  FACES[(v - 896) // 8] at s = 1.1 + 0.1 (v % 8), t = 1.6.
- r2 = a direction (x, y, z, 0) for Direct3D's sequence: cube r4,
  r2.zzxy, r2.yxzz; rcp r6.z, |r4.z|; mad r6.xy, r4, r6.z, 1.5; r6.w = r4.w;
  tfetchCube r6.yxw. v < 512 random in [-1, 1]^3; v < 768 ties between
  axes and corners; else one axis 2 with the others swept across [-2, 2].
The 'dir' tests export r4 (cube's result), r6 (the fetch coordinates) and
the fetch.
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gpugen  # noqa: E402
import xenos as x  # noqa: E402

SLOT = 1
VERTICES = 1024
SIDE = 8
PITCH = 32
BLOCK_DWORDS = 1024  # 4 KiB
BLOCKS = 8
F32, F8888 = 36, 6
POINT, LINEAR = (0, 0, 0), (1, 1, 0)
DIMENSION_CUBE = 3 << 9
FACES = (0.25, 0.5, 0.75, 1.5, 2.49, 2.5, 2.51, 3.999, 4.5, 5.25, 5.5, 5.99,
         6.0, 6.5, 7.0, 7.5)
EDGE = (-2.0 ** -21, 0.0, 2.0 ** -21, 2.0 ** -12)
K = gpugen.C_FIRST_FREE
f2w = gpugen.f2w


def directions(rng):
    out = [(rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1))
           for _ in range(512)]
    ties = []
    for a in (1.0, -1.0):
        for b in (1.0, -1.0):
            for c in (0.0, 0.25, -0.5, 0.999, 1.0, -1.0, 1.001, 0.75):
                ties += [(a, b, c), (a, c, b), (c, a, b)]
    for k in range(256 - len(ties)):
        s = 1.0 + (k % 8 - 4) * 2.0 ** -20
        ties.append(((1.0, -1.0)[k // 8 % 2], (1.0, -1.0)[k // 16 % 2] * s,
                     (k // 32 - 4) / 4))
    out += ties[:256]
    for k in range(256):
        axis, sign = k // 64 % 3, (1.0, -1.0)[k // 192] if k >= 192 else 1.0
        sweep = -2.0 + 4.0 * (k % 64) / 63
        d = [0.3, sweep, 2.0 * sign]
        d = d[-axis:] + d[:-axis] if axis else d
        out.append(tuple(d))
    return out


def inputs():
    rng = random.Random(0x0C0BE)
    dirs = directions(rng)
    words = []
    for v in range(VERTICES):
        if v < 768:
            s = 0.8 + 0.09 * (v % 16)
            t = 0.85 + 0.18 * (v // 16 % 8)
            face = float(v // 128)
        elif v < 896:
            k = v - 768
            edge = 1.0 + (k // 4 % 16 - 4) / SIDE + EDGE[k % 4]
            face = (0.0, 3.0)[k // 64]
            s, t = (edge, 1.4375) if k // 32 % 2 == 0 else (1.4375, edge)
        else:
            k = v - 896
            face = FACES[k // 8]
            s, t = 1.1 + 0.1 * (k % 8), 1.6
        words += [f2w(s), f2w(t), f2w(face), 0]
        words += [f2w(c) for c in dirs[v]] + [0]
    return words


def addr():
    return [f2w(float(d)) for d in range(BLOCKS * BLOCK_DWORDS)]


def tags():
    return [8 * (d % 32) | 8 * (d // 32 % 32) << 8 | 32 * (d // 1024) << 16 |
            0xFF << 24 for d in range(BLOCKS * BLOCK_DWORDS)]


def cube_constant(fmt, filters, clamp):
    words = gpugen.texture_constant(fmt, SIDE, SIDE, filters=filters,
                                    clamp=clamp, pitch=PITCH)
    words[2] |= 5 << 26  # stack depth 6 faces
    words[5] = (words[5] & ~(3 << 9)) | DIMENSION_CUBE
    return words


def build(suite):
    suite.set_inputs(inputs(), 8)
    suite.add_texture('addr', addr())
    suite.add_texture('tags', tags())
    vertex = gpugen.fetch_inputs(8)

    def face_test(name, texture, fmt, filters, clamp):
        fetch = x.tfetch(5, 1, 'xyz', SLOT, dimension=3).serialized()
        suite.test(name, gpugen.memexport_shader(
            8, [], [[gpugen.export_copy(5)]], fetches=vertex + [fetch]),
            fetch_constants={SLOT: (cube_constant(fmt, filters, clamp),
                                    texture)})

    def dir_test(name, texture, fmt, filters, clamp):
        body = [x.alu(vector_op=x.V_CUBE, vector_dest=4, vector_mask=0xF,
                      src1=x.r(2, 'zzxy'), src2=x.r(2, 'yxzz')),
                x.alu(vector_op=x.V_MAX, vector_dest=6, vector_mask=0x8,
                      src1=x.r(4), src2=x.r(4), scalar_op=x.S_RCP,
                      scalar_dest=6, scalar_mask=0x4,
                      src3=x.r(4, 'xxxz', absolute=True)),
                x.alu(vector_op=x.V_MAD, vector_dest=6, vector_mask=0x3,
                      src1=x.r(4), src2=x.r(6, 'zzzz'), src3=x.c(K, 'xxxx'))]

        def flow(s):
            s.exec_([x.tfetch(5, 6, 'yxw', SLOT, dimension=3).serialized()])
            s.exec_([x.alu(vector_op=x.V_MAX, vector_dest=7, vector_mask=0xF,
                           src1=x.r(5), src2=x.r(5), serialize=True)])

        suite.test(name, gpugen.memexport_shader(
            8, body, [[gpugen.export_copy(4)], [gpugen.export_copy(6)],
                      [gpugen.export_copy(7)]], fetches=vertex, flow=flow),
            constants={K: [f2w(1.5)] * 4}, elements=3,
            fetch_constants={SLOT: (cube_constant(fmt, filters, clamp),
                                    texture)})

    face_test('face point', 'addr', F32, POINT, 2)
    face_test('face point wrap', 'addr', F32, POINT, 0)
    face_test('face point border', 'addr', F32, POINT, 6)
    face_test('face linear', 'tags', F8888, LINEAR, 2)
    dir_test('dir point', 'addr', F32, POINT, 2)
    dir_test('dir linear', 'tags', F8888, LINEAR, 2)
