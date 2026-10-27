"""Resolve destination bases off 4 KiB: where the console puts texels when
RB_COPY_DEST_BASE isn't aligned, which titles never resolve to. The resolverect
suite saw 0x20, 0x100 and 0x800 move texels; this one steps the base through
every bit and combinations of them, with the whole 128x64 render target
(cpmem.py) resolved to a 128x64 tiled destination.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cpmem  # noqa: E402

# Titles' own bases first, then each bit alone, then sums of them.
OFFSETS = (0x1000, 0x20, 0x100, 0x800, 0x40, 0x80, 0x200, 0x400, 0x10, 0x8,
           0x4, 0x60, 0xC0, 0x120, 0x180, 0x1C0, 0x300, 0x600, 0xA00, 0xC00,
           0xE00, 0x3E0, 0x7C0, 0xF80, 0xFFC, 0x1100, 0x1800, 0x2020, 0x2900,
           0x3400, 0x5A0, 0x6C4, 0x9E8, 0xB70, 0xD50)


def build(suite):
    b = cpmem.Builder(suite)
    for offset in OFFSETS:
        b.test(f'base {offset:#x}', offset=offset)
    b.finish()
