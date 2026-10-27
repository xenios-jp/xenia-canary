"""Render target (EDRAM surface) pitches that aren't a whole number of 80-pixel
tiles, which titles never program: how many EDRAM tiles a row the console
lays the surface out with, at 1x, 2x and 4x, and which pixel wins where tiles
alias. The 128x64 render target (cpmem.py) is drawn whole, then resolved
(1x; raw a sample at a time at 2x and 4x).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cpmem  # noqa: E402

PITCHES = {0: (160, 128, 120, 150, 159, 161, 200, 240, 320),
           1: (160, 128, 200, 240, 320),
           2: (320, 160, 200, 240, 256, 400)}


def build(suite):
    b = cpmem.Builder(suite)
    for samples, pitches in PITCHES.items():
        for pitch in pitches:
            if samples == 0:
                b.test(f'1x surface pitch {pitch}', surface=pitch)
                continue
            for sample in range(1 << samples):
                b.test(f'{1 << samples}x surface pitch {pitch} sample {sample}',
                       surface=pitch, samples=samples, sample=sample)
    b.finish()
