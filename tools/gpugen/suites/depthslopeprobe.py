"""Signed planar slope bias versus explicit depth, with every sample read raw.

Both depth formats use identical +X and -Y planes. Explicit oDepth remains
constant 0.5 while the vertex plane varies, separating polygon slope bias
from depth exported by the shader. References are measured on Xbox 360.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import depthcases as dc


def cases():
    out = []
    for fmt in (dc.D24S8, dc.D24FS8):
        for path in (0, 1):
            for axis in ('x', 'y'):
                corners = ([0.5, 0.5 + 2**-8, 0.5, 0.5 + 2**-8]
                           if axis == 'x' else
                           [0.5, 0.5, 0.5 - 2**-8, 0.5 - 2**-8])
                for slope in (-1.0, 0.0, 1.0):
                    out.append((len(out),
                                f'{dc.PATHS[path]} {axis}-slope{slope:+g}',
                                dc.case(fmt, 0x555555A5, [dc.draw(
                                    path, corners=corners, exported=0.5,
                                    slope=slope)], full=1)))
    return out


def build_samples(suite, samples):
    dc.build(suite, cases(), samples=samples)


def build(suite):
    build_samples(suite, 2)
