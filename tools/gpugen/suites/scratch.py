"""Scratch register writeback: the command processor writing SCRATCH_REG0 to
SCRATCH_ADDR when SCRATCH_UMSK enables it, in each of the four byte orders
SCRATCH_UMSK's swap field (bits 16-17) selects: 0x12345678, 0x01FE2374 and
0x80FF0001 in each, then one write with writeback off, which leaves its word as
it was. Each write has a 32-byte slot of its own, as the console drops
SCRATCH_ADDR's low 5 bits; three more writes aim 4, 16 and 28 bytes into their
slots.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gpugen  # noqa: E402

VALUES = (0x12345678, 0x01FE2374, 0x80FF0001)
REG0 = 1  # SCRATCH_UMSK's writeback enable bit for SCRATCH_REG0
SLOT = 32
MISALIGNED = (4, 16, 28)


def writes():
    """[(SCRATCH_ADDR's offset, SCRATCH_UMSK, SCRATCH_REG0)]: each swap
    mode's values, one with writeback off, then the misaligned ones."""
    out = [(mode << 16 | REG0, value) for mode in range(4) for value in VALUES]
    out.append((0, VALUES[0]))
    out = [(SLOT * n, mask, value) for n, (mask, value) in enumerate(out)]
    out += [(SLOT * (len(out) + n) + offset, REG0, 0xA0B0C0D0 + n)
            for n, offset in enumerate(MISALIGNED)]
    return out


def build(suite):
    suite.test('writeback', None, scratch=writes())
