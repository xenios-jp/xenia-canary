"""Per-draw state as titles combine it: for each state tuple (sample count,
each written target's format, blending and write mask, depth format, write
and compare, stencil, alpha test, alpha to mask and polygon offset) that no
other suite draws, at 0.1% or more of the draws in 59 traces of titles, the
output merger registers of the commonest draw titles make with it, exactly
(TUPLES).

A tuple's first test sets up a 64x32 window at its sample count: every target
its draw writes stored with random words (drawn raw through a k_8_8_8_8 view,
as the merger suite stores its clears) and, when it tests depth or stencil,
the depth buffer drawn: depth by the tuple's own geometry (as titles' EQUAL
passes repeat a draw), then rising 0.02 to 0.98 left to right over rows
0-15, and stencil in bands of 16 columns, the tuple's reference r, r + l,
r - l and r ^ m for its read mask m and m's lowest bit l. Then the tuple's
draw in its registers, culling off: a quad over the window, depth rising
0.02 to 0.98 top to bottom, and a triangle at 0.5 over it, the pixel shader
exporting oC0..oC3 from four textures of random sixty-fourths (the merger
suite's). Each written target's
samples raw, at 2x and 4x their average converted, then each depth sample
raw, are resolved, a test each, the tests after the first drawing nothing; a
tuple writing nothing reads target 0's stored words back. The tests fill as
many parts as the ring takes (tuples1.py, ...).
"""

import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpugen  # noqa: E402
import merger  # noqa: E402
import pm4  # noqa: E402
import rb  # noqa: E402

WIDTH, HEIGHT = rb.WIDTH, rb.HEIGHT
# Each tuple's registers, in this order.
FIELDS = ('RB_COLOR_MASK', 'RB_BLENDCONTROL0', 'RB_BLENDCONTROL1',
          'RB_BLENDCONTROL2', 'RB_BLENDCONTROL3', 'RB_DEPTHCONTROL',
          'RB_STENCILREFMASK', 'RB_STENCILREFMASK_BF', 'RB_COLORCONTROL',
          'RB_ALPHA_REF', 'PA_SC_AA_MASK', 'PA_SU_SC_MODE_CNTL',
          'PA_SU_POLY_OFFSET_FRONT_SCALE', 'PA_SU_POLY_OFFSET_FRONT_OFFSET',
          'PA_SU_POLY_OFFSET_BACK_SCALE', 'PA_SU_POLY_OFFSET_BACK_OFFSET',
          'RB_BLEND_RED', 'RB_BLEND_GREEN', 'RB_BLEND_BLUE', 'RB_BLEND_ALPHA',
          'RB_MODECONTROL')
TUPLES = (
    ('4x rt[1w 2w] z[d1wf6 s]',
     (0x000000FF, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00708667,
      0x00E01F1F, 0x00FFFF00, 0x87000006, 0x3F008081, 0x0000FFFF, 0x00218002,
      0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x00000000, 0x00000000,
      0x00000000, 0x00000000, 0x00000004)),
    ('4x rt[1b 0w] z[d1f6 s]',
     (0x000000FF, 0x000E0001, 0x00010001, 0x00010001, 0x00010001, 0x00708663,
      0x00E01F1F, 0x00FFFF00, 0x87000006, 0x3F008081, 0x0000FFFF, 0x00218002,
      0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x00000000, 0x00000000,
      0x00000000, 0x00000000, 0x00000004)),
    ('4x rt[1b 10b] z[d1f6 s] po',
     (0x000000FF, 0x070E0706, 0x00010706, 0x070E0706, 0x070E0706, 0x00700563,
      0x00002020, 0x00FFFF00, 0x87000006, 0x3F008081, 0x0000FFFF, 0x00219802,
      0x41000000, 0x36A7C5AC, 0x41000000, 0x36A7C5AC, 0x00000000, 0x00000000,
      0x00000000, 0x00000000, 0x00000004)),
    ('4x rt[0w] z[d0wf3]',
     (0x0000000F, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00700736,
      0x00FFFF00, 0x00FFFF00, 0x87000004, 0x00000000, 0x0000FFFF, 0x00018002,
      0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x3F800000, 0x3F800000,
      0x3F800000, 0x3F800000, 0x00000004)),
    ('2x rt[3w] z[d1wf6 s] at4 a2c',
     (0x0000000F, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00708767,
      0x00FFFF00, 0x00FFFF00, 0x8700001C, 0x00000000, 0x0000FFFF, 0x00018000,
      0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
      0x00000000, 0x00000000, 0x00000004)),
    ('4x rt[1b 2w] z[d1wf6 s] at6',
     (0x000000FF, 0x000E0001, 0x00010001, 0x00010001, 0x00010001, 0x00708767,
      0x00200020, 0x00FFFF00, 0x8700000E, 0x3F008081, 0x0000FFFF, 0x00218000,
      0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x00000000, 0x00000000,
      0x00000000, 0x00000000, 0x00000004)),
    ('4x rt[] z[d0wf3 s] po',
     (0x00000000, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00708737,
      0x00FFFF00, 0x00FFFF00, 0xAA000004, 0x00000000, 0x0000FFFF, 0x00019806,
      0x42333333, 0x00000000, 0x42333333, 0x00000000, 0x00000000, 0x00000000,
      0x00000000, 0x00000000, 0x00000005)),
    ('2x rt[3w] z[d1wf6 s]',
     (0x0000000F, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00708767,
      0x00FFFF00, 0x00FFFF00, 0x87000004, 0x00000000, 0x0000FFFF, 0x00018002,
      0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
      0x00000000, 0x00000000, 0x00000004)),
    ('1x rt[] z[d0wf3]',
     (0x00000000, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00700736,
      0x00FFFF00, 0x00FFFF00, 0x87000004, 0x00000000, 0x0000FFFF, 0x00010000,
      0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x3F800000, 0x3F800000,
      0x3F800000, 0x3F800000, 0x00000004)),
    ('4x rt[] z[-] at6',
     (0x00000000, 0x000E0001, 0x00010001, 0x00010001, 0x00010001, 0x00000000,
      0x00200020, 0x00FFFF00, 0x8700000E, 0x3F008081, 0x0000FFFF, 0x00010000,
      0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x00000000, 0x00000000,
      0x00000000, 0x00000000, 0x00000004)),
    ('2x rt[0wm] z[d1wf6] a2c',
     (0x00000007, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00700766,
      0x00FFFF00, 0x00FFFF00, 0x87000016, 0x3B808081, 0x0000FFFF, 0x00018002,
      0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x3F800000, 0x3F800000,
      0x3F800000, 0x3F800000, 0x00000004)),
    ('1x rt[1w 2w] z[d1wf6 s]',
     (0x000000FF, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00708667,
      0x00E01F1F, 0x00FFFF00, 0x87000006, 0x3F008081, 0x0000FFFF, 0x00218002,
      0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x00000000, 0x00000000,
      0x00000000, 0x00000000, 0x00000004)),
    ('2x rt[3w] z[d1f2]',
     (0x0000000F, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00708722,
      0x00FFFF07, 0x00FFFF00, 0xAA000005, 0x00000000, 0x0000FFFF, 0x00018002,
      0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x3F800000, 0x3F800000,
      0x3F800000, 0x3F800000, 0x00000004)),
    ('1x rt[0w 0w] z[d0wf3 s]',
     (0x000000FF, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00708737,
      0x00FFFF01, 0x00FFFF00, 0x87000007, 0x00000000, 0x0000FFFF, 0x00018006,
      0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x3F800000, 0x3F800000,
      0x3F800000, 0x3F800000, 0x00000004)),
    ('4x rt[0wm] z[d0wf3]',
     (0x00000007, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00700736,
      0x00FFFF00, 0x00FFFF00, 0x87000004, 0x00000000, 0x0000FFFF, 0x00018006,
      0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x3F800000, 0x3F800000,
      0x3F800000, 0x3F800000, 0x00000004)),
    ('1x rt[3w 2w] z[d1wf6 s]',
     (0x000000FF, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00708467,
      0x00F807A4, 0x00FFFF00, 0x87000007, 0x00000000, 0x0000FFFF, 0x00018002,
      0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x00000000, 0x00000000,
      0x00000000, 0x00000000, 0x00000004)),
    ('1x rt[] z[d0wf3] po',
     (0x00000000, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00700736,
      0x00FFFF00, 0x00FFFF00, 0xAA000006, 0x3F008081, 0x0000FFFF, 0x00019802,
      0x41800000, 0x3851B717, 0x41800000, 0x3851B717, 0x3F800000, 0x3F800000,
      0x3F800000, 0x3F800000, 0x00000005)),
    ('4x rt[] z[d1wf7]',
     (0x000000FF, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00000076,
      0x00E01F3F, 0x00FFFF00, 0x87000006, 0x3F008081, 0x0000FFFF, 0x00010000,
      0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x3F800000, 0x3F800000,
      0x3F800000, 0x3F800000, 0x00000005)),
    ('1x rt[1bm 1b] z[d1f6 s]',
     (0x000000F7, 0x07060706, 0x07060706, 0x07060706, 0x07060706, 0x00708663,
      0x00E01F3F, 0x00FFFF00, 0x87000006, 0x3F008081, 0x0000FFFF, 0x00218002,
      0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x3F800000, 0x3F800000,
      0x3F800000, 0x3F800000, 0x00000004)),
    ('1x rt[1b 0w] z[d1f6 s]',
     (0x000000FF, 0x000E0001, 0x00010001, 0x00010001, 0x00010001, 0x00708663,
      0x00E01F1F, 0x00FFFF00, 0x87000006, 0x3F008081, 0x0000FFFF, 0x00218002,
      0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x00000000, 0x00000000,
      0x00000000, 0x00000000, 0x00000004)),
    ('1x rt[3w] z[d1f6 s]',
     (0x0000000F, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00708463,
      0x00F80704, 0x00FFFF00, 0x87000007, 0x00000000, 0x0000FFFF, 0x00018002,
      0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x00000000, 0x00000000,
      0x00000000, 0x00000000, 0x00000004)),
    ('2x rt[0w 0w 0w] z[d1wf6 s]',
     (0x00000FFF, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00708767,
      0x00FFFF00, 0x00FFFF00, 0x87000005, 0x00000000, 0x0000FFFF, 0x00018002,
      0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
      0x00000000, 0x00000000, 0x00000004)),
    ('2x rt[3w] z[d1wf6]',
     (0x0000000F, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00700766,
      0x00FFFF00, 0x00FFFF00, 0x87000007, 0x3F800000, 0x0000FFFF, 0x00018002,
      0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x3F800000, 0x3F800000,
      0x3F800000, 0x3F800000, 0x00000004)),
    ('1x rt[1bm 1b] z[d1f6]',
     (0x000000F7, 0x07060706, 0x07060706, 0x07060706, 0x07060706, 0x00708662,
      0x00E01F3F, 0x00FFFF00, 0x87000006, 0x3F008081, 0x0000FFFF, 0x00218002,
      0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x3F800000, 0x3F800000,
      0x3F800000, 0x3F800000, 0x00000004)),
    ('4x rt[2w 2w 0w 0w] z[-]',
     (0x0000FFFF, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00000000,
      0x00FF0002, 0x00FFFF00, 0x00000000, 0x3F008081, 0x0000FFFF, 0x00010000,
      0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x3F800000, 0x3F800000,
      0x3F800000, 0x3F800000, 0x00000004)),
    ('4x rt[0w 0w 0w 0w] z[-]',
     (0x0000FFFF, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00000000,
      0x00FF0000, 0x00FFFF00, 0x00000000, 0x3F008081, 0x0000FFFF, 0x00010000,
      0x41000000, 0x36A7C5AC, 0x41000000, 0x36A7C5AC, 0x00000000, 0x00000000,
      0x00000000, 0x00000000, 0x00000004)),
    ('1x rt[] z[d0wf3 s] po',
     (0x00000000, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00708737,
      0x00770000, 0x00FFFF00, 0x87000007, 0x00000000, 0x0000FFFF, 0x00219802,
      0x41B4F4AC, 0x3B03126F, 0x41B4F4AC, 0x3B03126F, 0x00000000, 0x00000000,
      0x00000000, 0x00000000, 0x00000005)),
    ('4x rt[3wm] z[d1wf6 s]',
     (0x00000007, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00708767,
      0x00FFFF00, 0x00FFFF00, 0xAA000004, 0x00000000, 0x0000FFFF, 0x00018002,
      0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
      0x00000000, 0x00000000, 0x00000004)),
    ('2x rt[0wm] z[d1wf6]',
     (0x00000007, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00700766,
      0x00FFFF00, 0x00FFFF00, 0xAA000005, 0x00000000, 0x0000FFFF, 0x00018002,
      0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x3F800000, 0x3F800000,
      0x3F800000, 0x3F800000, 0x00000004)),
    ('4x rt[0b] z[d0f3]',
     (0x0000000F, 0x07060706, 0x07060706, 0x07060706, 0x07060706, 0x00700732,
      0x00FFFF00, 0x00FFFF00, 0x87000004, 0x00000000, 0x0000FFFF, 0x00018000,
      0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x3F800000, 0x3F800000,
      0x3F800000, 0x3F800000, 0x00000004)),
    ('2x rt[12b] z[d1wf6 s] at4',
     (0x0000000F, 0x07060706, 0x07060706, 0x07060706, 0x07060706, 0x00708767,
      0x00FFFF00, 0x00FFFF00, 0x8700000C, 0x00000000, 0x0000FFFF, 0x00018002,
      0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
      0x00000000, 0x00000000, 0x00000004)),
    ('4x rt[0w] z[d0wf3] at4',
     (0x0000000F, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00700736,
      0x00FFFF00, 0x00FFFF00, 0x8700000C, 0x3E48C8CA, 0x0000FFFF, 0x00018000,
      0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x3F800000, 0x3F800000,
      0x3F800000, 0x3F800000, 0x00000004)),
    ('2x rt[3w] z[d1f2 s]',
     (0x0000000F, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00708723,
      0x00FFFF07, 0x00FFFF00, 0xAA000005, 0x00000000, 0x0000FFFF, 0x00018002,
      0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x3F800000, 0x3F800000,
      0x3F800000, 0x3F800000, 0x00000004)),
    ('2x rt[] z[d1wf6]',
     (0x00000000, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00700766,
      0x00FFFF00, 0x00FFFF00, 0x87000007, 0x3F800000, 0x0000FFFF, 0x00018002,
      0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x3F800000, 0x3F800000,
      0x3F800000, 0x3F800000, 0x00000005)),
    ('4x rt[] z[d0wf3 s]',
     (0x00000000, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00708737,
      0x00FFFF00, 0x00FFFF00, 0xAA000004, 0x00000000, 0x0000FFFF, 0x00018006,
      0x428F851F, 0x33D6BF95, 0x428F851F, 0x33D6BF95, 0x00000000, 0x00000000,
      0x00000000, 0x00000000, 0x00000005)),
    ('2x rt[12wm] z[d1wf6 s] at4',
     (0x00000007, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00708767,
      0x00FFFF05, 0x00FFFF00, 0x8700000C, 0x3C808081, 0x0000FFFF, 0x00018002,
      0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
      0x00000000, 0x00000000, 0x00000004)),
    ('2x rt[] z[d1f6]',
     (0x00000000, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00700762,
      0x00FFFF00, 0x00FFFF00, 0x87000006, 0x3B808081, 0x0000FFFF, 0x00018006,
      0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x3F800000, 0x3F800000,
      0x3F800000, 0x3F800000, 0x00000005)),
    ('1x rt[12bm] z[d1f6]',
     (0x00000007, 0x07060706, 0x07060706, 0x07060706, 0x07060706, 0x00700762,
      0x00000000, 0x00FFFF00, 0x87000007, 0x00000000, 0x0000FFFF, 0x00018002,
      0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x00000000, 0x00000000,
      0x00000000, 0x00000000, 0x00000004)),
    ('4x rt[0w] z[d0wf3] at6',
     (0x0000000F, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00700736,
      0x00FFFF00, 0x00FFFF00, 0x8700000E, 0x3F34B4B5, 0x0000FFFF, 0x00018000,
      0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x3F800000, 0x3F800000,
      0x3F800000, 0x3F800000, 0x00000004)),
    ('4x rt[0b] z[d0f3] at1',
     (0x0000000F, 0x07060706, 0x07060706, 0x07060706, 0x07060706, 0x00700732,
      0x00FFFF00, 0x00FFFF00, 0x87000009, 0x3F34B4B5, 0x0000FFFF, 0x00018000,
      0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x3F800000, 0x3F800000,
      0x3F800000, 0x3F800000, 0x00000004)),
    ('4x rt[] z[d1f6 s]',
     (0x00000077, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00744F63,
      0x007F0040, 0x00FFFF00, 0x87000006, 0x3F008081, 0x0000FFFF, 0x00218006,
      0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x00000000, 0x00000000,
      0x00000000, 0x00000000, 0x00000005)),
    ('4x rt[1bm 10b] z[d1f6 s] po',
     (0x000000F7, 0x070E0706, 0x00010706, 0x070E0706, 0x070E0706, 0x00700563,
      0x00002020, 0x00FFFF00, 0x87000006, 0x3F008081, 0x0000FFFF, 0x00219802,
      0x41000000, 0x36A7C5AC, 0x41000000, 0x36A7C5AC, 0x00000000, 0x00000000,
      0x00000000, 0x00000000, 0x00000004)),
    ('1x rt[1w 2w] z[d1wf6 s] at6',
     (0x000000FF, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00708667,
      0x00E01F1F, 0x00FFFF00, 0x8700000E, 0x3F008081, 0x0000FFFF, 0x00218002,
      0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x00000000, 0x00000000,
      0x00000000, 0x00000000, 0x00000004)),
    ('1x rt[1b 0w] z[d1f6 s] at6',
     (0x000000FF, 0x000E0001, 0x00010001, 0x00010001, 0x00010001, 0x00708663,
      0x00E01F1F, 0x00FFFF00, 0x8700000E, 0x3F008081, 0x0000FFFF, 0x00218002,
      0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x00000000, 0x00000000,
      0x00000000, 0x00000000, 0x00000004)),
    ('2x rt[12b] z[d1f6]',
     (0x0000000F, 0x07060706, 0x07060706, 0x07060706, 0x07060706, 0x00708762,
      0x00FFFF07, 0x00FFFF00, 0xAA000006, 0x3B808081, 0x0000FFFF, 0x00018002,
      0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x3F800000, 0x3F800000,
      0x3F800000, 0x3F800000, 0x00000004)),
    ('2x rt[3wm] z[d1wf6 s] a2c',
     (0x00000007, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00708767,
      0x00FFFF87, 0x00FFFF00, 0xAA000015, 0x00000000, 0x0000FFFF, 0x00018000,
      0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x3F800000, 0x3F800000,
      0x3F800000, 0x3F800000, 0x00000004)),
    ('2x rt[12wm] z[d1wf6 s]',
     (0x00000007, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00708767,
      0x00FFFF04, 0x00FFFF00, 0x87000004, 0x00000000, 0x0000FFFF, 0x00018002,
      0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
      0x00000000, 0x00000000, 0x00000004)),
    ('2x rt[3wm] z[d1f6] a2c',
     (0x00000007, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00708762,
      0x00FFFF87, 0x00FFFF00, 0x87000015, 0x3EC0C0C2, 0x0000FFFF, 0x00018000,
      0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x3F800000, 0x3F800000,
      0x3F800000, 0x3F800000, 0x00000004)),
    ('4x rt[3w] z[d1wf6] a2c',
     (0x0000000F, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00700766,
      0x00FFFF00, 0x00FFFF00, 0x87000015, 0x00000000, 0x0000FFFF, 0x00018006,
      0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x3F800000, 0x3F800000,
      0x3F800000, 0x3F800000, 0x00000004)),
    ('4x rt[] z[d0f3 s]',
     (0x00000000, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00740733,
      0x00FFFF01, 0x00FFFF00, 0x87000007, 0x00000000, 0x0000FFFF, 0x00018006,
      0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x3F800000, 0x3F800000,
      0x3F800000, 0x3F800000, 0x00000005)),
    ('2x rt[0w] z[d0wf3]',
     (0x0000000F, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00700736,
      0x00FFFF00, 0x00FFFF00, 0x87000004, 0x00000000, 0x0000FFFF, 0x00018002,
      0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x3F800000, 0x3F800000,
      0x3F800000, 0x3F800000, 0x00000004)),
    ('4x rt[] z[d1f6]',
     (0x00000000, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00700762,
      0x00000000, 0x00FFFF00, 0x87000007, 0x00000000, 0x0000FFFF, 0x00018000,
      0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x00000000, 0x00000000,
      0x00000000, 0x00000000, 0x00000005)),
    ('1x rt[] z[d1f6]',
     (0x00000000, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00700762,
      0x00000000, 0x00FFFF00, 0x87000007, 0x00000000, 0x0000FFFF, 0x00018000,
      0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x00000000, 0x00000000,
      0x00000000, 0x00000000, 0x00000005)),
    ('2x rt[12bm] z[d1f6 s]',
     (0x00000007, 0x07060706, 0x07060706, 0x07060706, 0x07060706, 0x00700563,
      0x00000F07, 0x00FFFF00, 0xAA000005, 0x00000000, 0x0000FFFF, 0x00018006,
      0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x3F800000, 0x3F800000,
      0x3F800000, 0x3F800000, 0x00000004)),
    ('1x rt[] z[d0wf7]',
     (0x00000000, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00000076,
      0x00002020, 0x00FFFF00, 0x87000006, 0x3F008081, 0x0000FFFF, 0x00010000,
      0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x00000000, 0x00000000,
      0x00000000, 0x00000000, 0x00000005)),
    ('2x rt[3wm] z[d1wf6 s]',
     (0x00000007, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00708767,
      0x00FFFF05, 0x00FFFF00, 0x87000004, 0x00000000, 0x0000FFFF, 0x00018002,
      0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
      0x00000000, 0x00000000, 0x00000004)),
    ('1x rt[12b] z[-] at5',
     (0x0000000F, 0x07060706, 0x07060706, 0x07060706, 0x07060706, 0x00700760,
      0x00FFFF00, 0x00FFFF00, 0xAA00000D, 0x00000000, 0x0000FFFF, 0x00018000,
      0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
      0x00000000, 0x00000000, 0x00000004)),
    ('4x rt[] z[d0wf7]',
     (0x00000000, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00000076,
      0x00002020, 0x00FFFF00, 0x87000006, 0x3F008081, 0x0000FFFF, 0x00010000,
      0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x00000000, 0x00000000,
      0x00000000, 0x00000000, 0x00000005)),
    ('1x rt[0bm] z[d1wf6]',
     (0x00000007, 0x07060706, 0x07060706, 0x07060706, 0x07060706, 0x00700766,
      0x00FFFF00, 0x00FFFF00, 0x87000007, 0x00000000, 0x0000FFFF, 0x00018006,
      0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x3F800000, 0x3F800000,
      0x3F800000, 0x3F800000, 0x00000004)),
    ('4x rt[1bm 1bm] z[- s]',
     (0x00000077, 0x06070607, 0x06070607, 0x06070607, 0x06070607, 0x00704261,
      0x007FFF40, 0x00FFFF00, 0x87000006, 0x3F008081, 0x0000FFFF, 0x00218006,
      0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x00000000, 0x00000000,
      0x00000000, 0x00000000, 0x00000004)),
    ('1x rt[] z[d0wf3 s]',
     (0x00000000, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00708637,
      0x00E01F1F, 0x00FFFF00, 0x87000006, 0x3F008081, 0x0000FFFF, 0x00218000,
      0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
      0x00000000, 0x00000000, 0x00000005)),
    ('1x rt[0bm] z[d1f6]',
     (0x00000007, 0x07060706, 0x07060706, 0x07060706, 0x07060706, 0x00700762,
      0x00FFFF00, 0x00FFFF00, 0x87000007, 0x00000000, 0x0000FFFF, 0x00018000,
      0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x3F800000, 0x3F800000,
      0x3F800000, 0x3F800000, 0x00000004)),
    ('1x rt[] z[d0f3]',
     (0x00000000, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00700232,
      0x0000FF00, 0x00FFFF00, 0x87000007, 0x00000000, 0x0000FFFF, 0x00018006,
      0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x3F800000, 0x3F800000,
      0x3F800000, 0x3F800000, 0x00000005)),
    ('2x rt[12bm] z[d1wf6 s]',
     (0x00000007, 0x07060706, 0x07060706, 0x07060706, 0x07060706, 0x00708767,
      0x00FFFF04, 0x00FFFF00, 0x87000004, 0x00000000, 0x0000FFFF, 0x00018002,
      0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
      0x00000000, 0x00000000, 0x00000004)),
    ('1x rt[] z[d1f6 s]',
     (0x00000007, 0x01010101, 0x01010101, 0x01010101, 0x01010101, 0x00744F63,
      0x00F70080, 0x00FFFF00, 0x87000007, 0x00000000, 0x0000FFFF, 0x00018006,
      0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x00000000, 0x00000000,
      0x00000000, 0x00000000, 0x00000005)),
    ('4x rt[12b] z[d1wf6] at5',
     (0x0000000F, 0x07060706, 0x07060706, 0x07060706, 0x07060706, 0x00700766,
      0x00FFFF00, 0x00FFFF00, 0xAA00000D, 0x00000000, 0x0000FFFF, 0x00018002,
      0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
      0x00000000, 0x00000000, 0x00000004)),
    ('1x rt[1b 2w] z[d1wf6 s] at6',
     (0x000000FF, 0x000E0001, 0x00010001, 0x00010001, 0x00010001, 0x00708767,
      0x00200020, 0x00FFFF00, 0x8700000E, 0x3F008081, 0x0000FFFF, 0x00218000,
      0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x00000000, 0x00000000,
      0x00000000, 0x00000000, 0x00000004)),
)

FORMATS = {f[1]: f for f in rb.FORMATS}
STENCIL, Z, Z_WRITE = 1, 2, 4
ALWAYS, REPLACE = 7, 2
# The depth buffer's draws: depth written; stencil written, REPLACE.
DEPTH_PASS = Z | Z_WRITE | ALWAYS << 4
STENCIL_PASS = (STENCIL | ALWAYS << 8 | REPLACE << 11 | REPLACE << 14 |
                REPLACE << 17)
CULL = 3  # PA_SU_SC_MODE_CNTL's cull_front and cull_back
f2w = gpugen.f2w


def parse(name, values):
    """(samples, [(slot, format)], depth format, whether depth or stencil is
    tested, {register: value})."""
    state = dict(zip(FIELDS, values))
    samples = int(name[0])
    formats = [int(re.match(r'\d+', t).group())
               for t in name[name.index('rt[') + 3:name.index(']')].split()]
    # Only RB_MODECONTROL 4 writes color.
    slots = [t for t in range(4) if state['RB_COLOR_MASK'] >> 4 * t & 0xF
             and state['RB_MODECONTROL'] == 4]
    assert len(slots) == len(formats), name
    depth = re.search(r'z\[d(\d)', name)
    return (samples, list(zip(slots, formats)),
            int(depth.group(1)) if depth else 0,
            bool(state['RB_DEPTHCONTROL'] & (STENCIL | Z)), state)


class Geometry:
    """Inputs of 8 dwords a vertex (merger.Quads'): the Resolve's rectangle
    first, then quads and triangles in clip space at a depth per vertex,
    each added once."""

    def __init__(self):
        self.inputs = rb.RECT + [0, 0]
        self.draws = {}

    def triangles(self, *vertices):
        """A draw of triangles of (x, y, z) pixel-space vertices."""
        if vertices not in self.draws:
            start = len(self.inputs) // 8
            for px, py, z in vertices:
                self.inputs += [f2w(v) for v in (
                    (px - 0.5) / (WIDTH / 2) - 1,
                    1 - (py - 0.5) / (HEIGHT / 2), z, 1.0, 0.0, 0.0, 0.0,
                    1.0)]
            self.draws[vertices] = (pm4.PRIM_TRIANGLE_LIST, start,
                                    len(vertices))
        return self.draws[vertices]

    def quad(self, x0, y0, x1, y1, top_left, top_right, bottom_left,
             bottom_right, more=()):
        """A rectangle over pixels x0 .. x1 - 1, y0 .. y1 - 1 (edges half a
        pixel before their centers) at a depth per corner, and `more`."""
        tl, tr = (x0, y0, top_left), (x1, y0, top_right)
        bl, br = (x0, y1, bottom_left), (x1, y1, bottom_right)
        return self.triangles(tl, tr, bl, bl, tr, br, *more)


def textures(suite):
    """'source t': random sixty-fourths; 'store t': random words, as
    (byte + 1/4) / 255 a channel."""
    rng = random.Random(0x7AB1E5)
    for t in range(4):
        suite.add_texture(f'source {t}', [
            f2w(rng.randrange(65) / 64) for _ in range(4 * WIDTH * HEIGHT)])
        suite.add_texture(f'store {t}', [
            f2w((rng.randrange(256) + 0.25) / 255)
            for _ in range(4 * WIDTH * HEIGHT)])


def fetch(name):
    return merger.fetch(name)


def target_registers(samples, targets, depth_format):
    """merger.target_registers', with the depth buffer's format."""
    code = merger.SAMPLES[samples][0]
    infos = [t * merger.TARGET_TILES for t in range(4)]
    for slot, fmt in targets:
        infos[slot] |= fmt << 16
    return [
        ('SQ_CONTEXT_MISC', [1 << 2 | merger.PARAM_GEN << 8 |
                             gpugen.SCREEN_XY]),
        ('SQ_INTERPOLATOR_CNTL', [0xFFFFFFFF]),
        ('RB_SURFACE_INFO', [80 | code << 16]),
        ('RB_COLOR_INFO', [infos[0], merger.DEPTH_BASE | depth_format << 16,
                           *infos[1:]]),
        ('PA_SC_WINDOW_SCISSOR_BR', [HEIGHT << 16 | WIDTH]),
        ('PA_SC_AA_CONFIG', [gpugen.AA_CONFIG[code]]),
        ('PA_CL_CLIP_CNTL', [1 << 19, gpugen.MODE_CNTL | (
            gpugen.MSAA_ENABLE if samples > 1 else 0), 0x43F]),
        ('PA_CL_VPORT_XSCALE', [f2w(v) for v in (
            WIDTH / 2, WIDTH / 2, -HEIGHT / 2, HEIGHT / 2, 1.0, 0.0)])], infos


def add_tests(suite, geometry, name, values):
    samples, targets, depth_format, depth, state = parse(name, values)
    stored = targets or [(0, 0)]
    registers, infos = target_registers(samples, targets, depth_format)
    shader = gpugen.interpolating_vertex_shader(8)
    pixel = merger.pixel_shader()
    constants = {256 + merger.K: [f2w(0.5), 0, 0, 0]}
    sources = {1 + t: fetch(f'source {t}') for t in range(4)}
    whole = geometry.quad(0, 0, WIDTH, HEIGHT, 0.0, 0.0, 0.0, 0.0)
    passes = []
    for slot, _ in stored:
        passes.append((
            [('RB_COLOR_INFO', [slot * merger.TARGET_TILES]),
             ('RB_DEPTHCONTROL', [0, merger.NO_BLEND_CONTROL,
                                  merger.COLORCONTROL]),
             ('RB_COLOR_MASK', [0xF])],
            {**sources, 1: fetch(f'store {slot}')} if not passes else
            {1: fetch(f'store {slot}')}, whole))
    draw = geometry.quad(0, 0, WIDTH, HEIGHT, 0.02, 0.02, 0.98, 0.98,
                         ((6, 3, 0.5), (58, 10, 0.5), (24, 29, 0.5)))
    if depth:
        passes.append(([('RB_COLOR_MASK', [0]),
                        ('RB_DEPTHCONTROL', [DEPTH_PASS])], {}, draw))
        passes.append(([], {}, geometry.quad(0, 0, WIDTH, 16, 0.02, 0.98,
                                              0.02, 0.98)))
        reference = state['RB_STENCILREFMASK'] & 0xFF
        mask = state['RB_STENCILREFMASK'] >> 8 & 0xFF
        low = mask & -mask or 1
        for band, value in enumerate((reference, reference + low,
                                      reference - low,
                                      reference ^ (mask or 0xFF))):
            passes.append((
                ([('RB_DEPTHCONTROL', [STENCIL_PASS])] if band == 0 else []) +
                [('RB_STENCILREFMASK', [0xFFFF00 | value & 0xFF])], {},
                geometry.quad(16 * band, 0, 16 * band + 16, HEIGHT,
                              0.5, 0.5, 0.5, 0.5)))
    after = [
        ('RB_COLOR_INFO', [infos[0]]),
        ('RB_COLOR_MASK', [state['RB_COLOR_MASK']]),
        ('RB_BLEND_RED', [state[f'RB_BLEND_{c}']
                          for c in ('RED', 'GREEN', 'BLUE', 'ALPHA')]),
        ('RB_STENCILREFMASK_BF', [state['RB_STENCILREFMASK_BF'],
                                  state['RB_STENCILREFMASK'],
                                  state['RB_ALPHA_REF']]),
        ('RB_DEPTHCONTROL', [state['RB_DEPTHCONTROL'],
                             state['RB_BLENDCONTROL0'],
                             state['RB_COLORCONTROL']]),
        ('RB_MODECONTROL', [state['RB_MODECONTROL'],
                            state['RB_BLENDCONTROL1'],
                            state['RB_BLENDCONTROL2'],
                            state['RB_BLENDCONTROL3']]),
        ('PA_SC_AA_MASK', [state['PA_SC_AA_MASK']]),
        ('PA_SU_SC_MODE_CNTL', [state['PA_SU_SC_MODE_CNTL'] & ~CULL]),
        ('PA_SU_POLY_OFFSET_FRONT_SCALE', [
            state[f'PA_SU_POLY_OFFSET_{f}']
            for f in ('FRONT_SCALE', 'FRONT_OFFSET', 'BACK_SCALE',
                      'BACK_OFFSET')])]
    readbacks = []
    for slot, fmt in stored:
        view = [('RB_COLOR_INFO', [infos[slot] if targets else 0])]
        for s in range(samples):
            readbacks.append((f't{slot} s{s}', gpugen.resolve_registers(
                WIDTH, HEIGHT, 6, 4, command=0, sample_select=s,
                registers=view)))
        if samples > 1:
            dest = FORMATS[fmt][3]
            readbacks.append((f't{slot} average', gpugen.resolve_registers(
                WIDTH, HEIGHT, dest, rb.TEXEL_BYTES[dest],
                sample_select=merger.SAMPLES[samples][1], registers=view)))
    if depth:
        for s in range(samples):
            readbacks.append((f'depth s{s}', gpugen.resolve_registers(
                WIDTH, HEIGHT, 6, 4, command=0, source=4, sample_select=s)))
    nothing = geometry.triangles((0, 0, 0.5), (1, 0, 0.5), (0, 1, 0.5))
    for i, (readback, resolve) in enumerate(readbacks):
        if i == 0:
            suite.test(f'{name} {readback}', shader, constants=constants,
                       pixel_shader=pixel, program_cntl=merger.PROGRAM_CNTL,
                       registers=registers, passes=passes,
                       after_passes=after, fetch_constants={1: sources[1]},
                       draw=draw, resolve=resolve)
        else:
            # Nothing drawn: RB_COLOR_MASK and RB_DEPTHCONTROL are 0.
            suite.test(f'{name} {readback}', shader, constants=constants,
                       pixel_shader=pixel, program_cntl=merger.PROGRAM_CNTL,
                       registers=registers, draw=nothing, resolve=resolve)


# The ring's words for the tests, less the stream's start and end.
RING_WORDS = gpugen.RING_SIZE // 4 - 1024


def parts():
    """TUPLES split into parts fitting the ring, in order."""
    out, words = [[]], 0
    for name, values in TUPLES:
        suite = gpugen.Suite('size')
        geometry = Geometry()
        textures(suite)
        add_tests(suite, geometry, name, values)
        suite.set_inputs(geometry.inputs, 8)
        layout = {'input': 0, 'dummy': 0, 'textures': {
            n: 0 for n in suite.textures}, 'outputs': {
                t.name: 0 for t in suite.tests}}
        empty = gpugen.Suite('empty')
        size = (len(gpugen.build_stream(suite, layout).words) -
                len(gpugen.build_stream(empty, layout).words))
        if words + size > RING_WORDS:
            out.append([])
            words = 0
        out[-1].append((name, values))
        words += size
    return out


def build(suite, part):
    """Part `part` (from 1) of the tuples (tuples1.py, ...)."""
    geometry = Geometry()
    textures(suite)
    for name, values in parts()[part - 1]:
        add_tests(suite, geometry, name, values)
    suite.set_inputs(geometry.inputs, 8)
