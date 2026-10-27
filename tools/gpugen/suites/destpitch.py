"""Resolve destination pitches that aren't a multiple of 32 texels, which
Direct3D never programs (it aligns to 32): how many macro tiles a row the
console tiles the destination with, and what the rest of a 128x64 render
target's macro tiles do when the destination is narrower. The whole target
(cpmem.py) is resolved to each pitch.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cpmem  # noqa: E402

PITCHES = (128, 96, 160, 80, 100, 112, 127, 129, 144, 192, 200, 224, 256, 64,
           48)


def build(suite):
    b = cpmem.Builder(suite)
    for pitch in PITCHES:
        b.test(f'dest pitch {pitch}', dest=(pitch, cpmem.HEIGHT))
    b.finish()
