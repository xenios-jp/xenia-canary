"""Memory export endian transforms and red/blue swap on finite lane patterns.

Each supported endian width no larger than the element is tested, with
and without stream swap_rb. Float and integer packed formats use distinct
components so byte, halfword, word and red/blue swaps are distinguishable.
All four eM0 components are initialized; vertices have disjoint addresses.
Modes wider than the element are excluded because padding semantics have
not been established. Existing mexport covers conversion boundary values.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gpugen  # noqa: E402

FORMATS = (('rgba8 uint', 6, 2, 4), ('rgba16 uint', 26, 2, 8),
           ('half4', 32, 7, 8), ('float', 36, 7, 4),
           ('float2', 37, 7, 8), ('float4', 38, 7, 16))


def stream_mode(endian, swap_rb):
    def change(words):
        words = list(words)
        words[2] = (words[2] & ~7) | endian | (int(swap_rb) << 19)
        return words
    return change


def build(suite):
    words = []
    for i in range(16):
        values = (3 + i, 17 + 2 * i, 65 + 3 * i, 129 + 5 * i)
        words += [gpugen.f2w(v) for v in values]
    suite.set_inputs(words, 4)
    shader = gpugen.memexport_shader(
        4, [], [[gpugen.export_copy(gpugen.INPUT_REGISTER)]])
    for name, fmt, num, size in FORMATS:
        modes = range(6 if size == 16 else 5 if size == 8 else 4)
        for endian in modes:
            for swap in (False, True):
                suite.test(f'{name} endian {endian} swap_rb {int(swap)}',
                           shader, export=(fmt, num, size),
                           stream=stream_mode(endian, swap))
