"""zresolve1 without its wedge: zresolve1 stopped the console 3 of 3 times at
its first Resolve to a destination format no title or clean suite resolves
to (k_24_8; its clearing Resolve to the same format went first). Here every
test clears through a k_8_8_8_8 destination, and the depth buffer (the same
2048 flat pixels of known Z in D24S8 and D24FS8) is resolved raw (copy
command 0) and converting (1) only to the destination formats the targets
suites resolve to on the console: k_8_8_8_8 (number formats 0 and 2, the
two titles resolve depth with), k_2_10_10_10, k_16_16, k_16_16_16_16,
k_16_16_16_16_FLOAT and k_32_FLOAT (float formats with number format 7 too).
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpugen  # noqa: E402
import zplane  # noqa: E402
import zresolve1  # noqa: E402

# (name, texture format, bytes, number formats)
DESTS = [('8_8_8_8', 6, 4, (0, 2)), ('2_10_10_10', 7, 4, (0,)),
         ('16_16', 25, 4, (0,)),
         ('16_16_16_16', 26, 8, (0,)), ('16_16_16_16_FLOAT', 32, 8, (0, 7)),
         ('32_FLOAT', 36, 4, (0, 7))]


def build(suite):
    rng = random.Random(0x24F8)
    b = zplane.Builder(suite)
    quads = zplane.pixel_quads([(z, 1.0) for z in zresolve1.zs(rng)])
    for fmt, fname in enumerate(zplane.FORMATS):
        for command in (0, 1):
            for dname, dest, bpp, numbers in DESTS:
                for number in numbers:
                    resolve = gpugen.resolve_registers(
                        zplane.WIDTH, zplane.HEIGHT, dest, bpp, number,
                        command=command, source=4)
                    b.test(f'{fname} command {command} {dname} number '
                           f'{number}', fmt, [zplane.draw(quads)],
                           resolve=resolve)
    b.finish()
