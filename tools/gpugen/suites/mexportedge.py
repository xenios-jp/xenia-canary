"""Memory-export conversion thresholds and special-value classification.

Finite inputs bracket integer half steps and normalized half steps by
adjacent float32 values. Signed 16-bit values also bracket half steps by
1/512 and 1/256, testing precision loss observed near negative halves.
Both signs of quiet/signaling NaNs, infinities, finite extremes and signed
zeros are separate inputs. Every eM0 component is written and every vertex
has a distinct address. Raw results establish behavior rather than assuming
host rounding or NaN conventions.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gpugen  # noqa: E402


def build(suite):
    values = {0, 0x80000000, 1, 0x80000001, 0x7F7FFFFF, 0xFF7FFFFF,
              0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00000,
              0x7F800001, 0xFF800001, 0x7FFFFFFF, 0xFFFFFFFF}

    def neighbors(value):
        word = gpugen.f2w(value)
        values.update((word + delta) & 0xFFFFFFFF
                      for delta in (-2, -1, 0, 1, 2))

    for integer in (0, 1, 2, 127, 128, 255, 1023, 32767, 32768, 65535):
        for sign in (-1, 1):
            half = sign * (integer + .5)
            neighbors(half)
            for offset in (-1 / 256, -1 / 512, 1 / 512, 1 / 256):
                neighbors(half + offset)
    for maximum in (127, 255, 511, 1023, 32767, 65535):
        for integer in (0, 1, maximum // 2, maximum - 1):
            for sign in (-1, 1):
                neighbors(sign * (integer + .5) / maximum)
    words = sorted(values)
    words += [0] * (-len(words) % 4)
    suite.set_inputs(words, 4)
    shader = gpugen.memexport_shader(
        4, [], [[gpugen.export_copy(gpugen.INPUT_REGISTER)]])
    for name, fmt, size in (('rgba8', 6, 4), ('rgb10a2', 7, 4),
                            ('rgba16', 26, 8)):
        for number, kind in enumerate(('unorm', 'snorm', 'uint', 'sint')):
            suite.test(f'{name} {kind} thresholds', shader,
                       export=(fmt, number, size))
