"""Compare D24S8 slope units with absolute and D24-scaled constant offsets.

The planes have binary-exact slopes at the 34-pixel viewport extent. Each
slope draw is paired with two constant-offset draws. Xbox references record
whether they agree; no expected output is calculated from the candidate rule.
Both slope signs, two magnitudes, two scales and X/Y orientations are covered.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import depthcases as dc


def cases():
    out = []
    for exponent in (-18, -22):
        step = 2.0**exponent
        for axis in ('x', 'y'):
            delta = 34 * step
            corners = ([0.5, 0.5 + delta, 0.5, 0.5 + delta] if axis == 'x'
                       else [0.5, 0.5, 0.5 - delta, 0.5 - delta])
            for scale in (0.5, -0.5, 2.0, -2.0):
                scaled = dc.f32(dc.f32(step * dc.from_bits(0x3F7FFFFF)) * scale)
                for kind, slope, bias in (
                        ('slope', scale, 0),
                        ('absolute', 0, step * scale),
                        ('d24-scaled', 0, scaled)):
                    out.append((len(out),
                                f'e{exponent}-{axis}-scale{scale:+g}-{kind}',
                                dc.case(dc.D24S8, 0x555555A5, [dc.draw(
                                    corners=corners, slope=slope, bias=bias)],
                                        full=1)))
    return out


def build(suite):
    dc.build(suite, cases(), samples=1)
