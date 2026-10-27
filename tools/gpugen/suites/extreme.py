"""Products and sums at the edges of the float range, and the ops whose
special cases the specials suite can't separate.

Every vertex carries A, B and C (three float4s). The vertices hold: products
landing on either side of 2^-126 (does the flush see the exact or the
rounded product?); mad products whose exponent leaves the float range below
or above, with an addend that brings the sum back; dp4, dp3 and dp2add terms
beyond FLT_MAX or below 2^-126 beside terms that cancel them; cube on ties,
signed zeros, denormals, infinities and NaNs; NaNs and infinities in each
lane of max4, dp3 and dp2add; and mulsprev2 with the previous scalar result
and its b operand set independently.
"""

import math
import os
import random
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gpugen  # noqa: E402
import xenos as x  # noqa: E402

f2w = gpugen.f2w
FLT_MAX, INF, NINF = 0x7F7FFFFF, 0x7F800000, 0xFF800000
QNAN, NQNAN, PAYLOAD, SNAN = 0x7FC00000, 0xFFC00000, 0x7FC12345, 0x7F800001
NEG = 0x80000000
ONE = f2w(1.0)


def w(value):
    """The word of a finite double that is exactly a float."""
    word = f2w(value)
    if struct.unpack('>f', struct.pack('>I', word))[0] != value:
        raise ValueError(f'{value!r} is not a float')
    return word


def mant(rng, bits=23):
    return 1 + rng.getrandbits(bits) / 2.0 ** bits


def sign(rng):
    return rng.choice((-1, 1))


def near_tiny(rng):
    """Lanes whose product is within a few ulps of +-2^-126: a random
    mantissa times the float nearest 2^-126 / a, stepped by -2..2 ulps;
    c = 0 (mad shows the flush) or +-2^-126 (shows the product's bits)."""
    a, b, c = [], [], []
    for _ in range(4):
        ea = rng.randint(-100, -26)
        av = sign(rng) * mant(rng) * 2.0 ** ea
        bv = 2.0 ** -126 / abs(av)
        bw = f2w(bv) + rng.randint(-2, 2)
        a.append(f2w(av))
        b.append(bw ^ (NEG if rng.random() < 0.5 else 0))
        c.append(rng.choice((0, NEG, w(2.0 ** -126), w(-2.0 ** -126),
                             w(1.5 * 2.0 ** -126))))
    return a, b, c


def low_mad(rng):
    """Products 2^-252..2^-127 (exact exponent), addend within 2^30 of the
    product but at least 2^-126 in magnitude, either sign."""
    a, b, c = [], [], []
    for _ in range(4):
        ep = rng.randint(-252, -127)
        ea = rng.randint(-126, ep + 126)
        eb = ep - ea
        av = sign(rng) * mant(rng) * 2.0 ** ea
        bv = sign(rng) * mant(rng) * 2.0 ** eb
        ec = max(-126, ep + rng.randint(0, 30))
        cv = sign(rng) * mant(rng, rng.choice((1, 4, 23))) * 2.0 ** ec
        a.append(w(av))
        b.append(w(bv))
        c.append(w(cv))
    return a, b, c


def high_mad(rng):
    """Products between FLT_MAX and 2^130 with an addend of the other sign
    that brings the sum back below FLT_MAX (or nearly)."""
    a, b, c = [], [], []
    for _ in range(4):
        s = sign(rng)
        ep = rng.choice((127, 128, 128, 129))
        ea = rng.randint(ep - 127, 127)
        av = mant(rng) * 2.0 ** ea
        bv = mant(rng) * 2.0 ** (ep - ea)
        cw = rng.choice((FLT_MAX, w(2.0 ** 127), w(mant(rng) * 2.0 ** 127),
                         w(mant(rng) * 2.0 ** 126)))
        a.append(w(s * av))
        b.append(w(bv))
        c.append(cw ^ (0 if s < 0 else NEG))
    return a, b, c


def dot_edges(rng):
    """dp terms: a product above FLT_MAX against one that cancels it, a
    product below 2^-126 beside a 2^-126 term, both with filler."""
    kind = rng.randrange(4)
    m = mant(rng)
    if kind == 0:
        # x = 1.5 * 2^128 (unbounded), y = -FLT_MAX: 2^127 + 2^104 if the
        # product keeps its exponent, 0 if it clamps.
        a = [w(1.5 * 2.0 ** 127), w(-1.0), w(m * 2.0 ** rng.randint(80, 126)),
             0]
        b = [w(2.0), FLT_MAX, w(sign(rng)), w(1.0)]
    elif kind == 1:
        # Two products above FLT_MAX that cancel, then a small term.
        e = rng.randint(64, 127)
        a = [w(m * 2.0 ** e), w(-m * 2.0 ** e), w(mant(rng)), w(mant(rng))]
        b = [w(2.0 ** (130 - e)), w(2.0 ** (130 - e)), w(1.0),
             w(2.0 ** rng.randint(-30, 30))]
    elif kind == 2:
        # A product below 2^-126 beside +-2^-126 terms.
        e = rng.randint(-160, -127)
        a = [w(m * 2.0 ** (e + 63)), w(2.0 ** -126), w(sign(rng) * 2.0 ** -126),
             w(mant(rng) * 2.0 ** -100)]
        b = [w(sign(rng) * 2.0 ** -63), w(1.5), w(1.0),
             w(sign(rng) * 2.0 ** rng.randint(-60, -20))]
    else:
        # Products far below 2^-126 only: does the sum flush per term or at
        # the end?
        a = [w(sign(rng) * mant(rng) * 2.0 ** rng.randint(-100, -64))
             for _ in range(4)]
        b = [w(mant(rng) * 2.0 ** rng.randint(-60, -27)) for _ in range(4)]
    c = [w(mant(rng) * 2.0 ** rng.randint(-130 + 4, -100)), 0, 0, 0]
    c[0] = rng.choice((c[0], 0, NEG, w(2.0 ** -126), w(-2.0 ** -126)))
    return a, b, c


CUBE_VALUES = [0, NEG, 1, NEG | 1, 0x007FFFFF, w(2.0 ** -126), ONE,
               w(-1.0), w(2.0), w(-2.0), INF, NINF, QNAN, PAYLOAD | NEG,
               FLT_MAX, w(-0.5)]


def cube_vertex(rng, i):
    """Ties first (two or three equal magnitudes, every sign), then random
    picks from CUBE_VALUES."""
    if i < 64:
        signs = [(i >> k) & 1 for k in range(3)]
        tie = (i >> 3) & 7
        big, small = w(1.5), w(0.75)
        mags = {0: (big, big, small), 1: (big, small, big),
                2: (small, big, big), 3: (big, big, big),
                4: (0, 0, 1), 5: (1, 0, 0), 6: (0, 1, 1),
                7: (0x00400000, 0x00400000, 1)}[tie]
        v = [m | (NEG if s else 0) for m, s in zip(mags, signs)]
    else:
        v = [rng.choice(CUBE_VALUES) for _ in range(3)]
    return v + [ONE]


LANE_SPECIALS = [QNAN, NQNAN, PAYLOAD, SNAN, INF, NINF, 0, NEG]


def lanes_vertex(rng, i):
    """max4 / dp3 / dp2add operands with specials in chosen lanes: A has one
    or two specials among ordinary values, B ordinary or special."""
    a = [w(sign(rng) * mant(rng, 4) * 2.0 ** rng.randint(-2, 2))
         for _ in range(4)]
    b = [w(sign(rng) * mant(rng, 4)) for _ in range(4)]
    a[i % 4] = LANE_SPECIALS[(i // 4) % 8]
    if i >= 32:
        a[(i + 1 + i // 32) % 4] = LANE_SPECIALS[(i // 3) % 8]
    if i >= 64:
        b[rng.randrange(4)] = rng.choice(LANE_SPECIALS)
    c = [rng.choice(LANE_SPECIALS + [ONE, w(-3.0)]), ONE, ONE, ONE]
    return a, b, c


PREV2_VALUES = [0xFF7FFFFF, NINF, INF, QNAN, PAYLOAD, 0, NEG, ONE, w(-1.0),
                1, w(2.0 ** -126), FLT_MAX]


def prev2_vertices():
    """mulsprev2 lanes: a = A, b = B, ps = C, over every (b, ps) pair of
    PREV2_VALUES with a cycling through ordinary and special values."""
    avals = [w(3.0), w(-0.5), INF, QNAN, 0, w(2.0 ** 100)]
    lanes = [(avals[(i + j) % len(avals)], b, ps)
             for i, b in enumerate(PREV2_VALUES)
             for j, ps in enumerate(PREV2_VALUES)]
    out = []
    for v in range(0, len(lanes), 4):
        group = lanes[v:v + 4]
        while len(group) < 4:
            group.append((ONE, ONE, ONE))
        out.append(([g[0] for g in group], [g[1] for g in group],
                    [g[2] for g in group]))
    return out


GROUPS = [('near tiny', near_tiny, 64), ('low mad', low_mad, 96),
          ('high mad', high_mad, 48), ('dot edges', dot_edges, 96)]


def vertices():
    """[(label, A, B, C)]"""
    rng = random.Random(0xE47E)
    out = []
    for label, make, count in GROUPS:
        for i in range(count):
            out.append((f'{label} {i}', *make(rng)))
    for i in range(96):
        a = cube_vertex(rng, i)
        out.append((f'cube {i}', a, [ONE] * 4, [ONE] * 4))
    for i in range(96):
        out.append((f'lanes {i}', *lanes_vertex(rng, i)))
    for i, (a, b, c) in enumerate(prev2_vertices()):
        out.append((f'prev2 {i}', a, b, c))
    return out


def inputs():
    words = []
    for _, a, b, c in vertices():
        words += a + b + c
    return words


def build(suite):
    suite.set_inputs(inputs(), 12)
    a, b, c = (gpugen.INPUT_REGISTER + i for i in range(3))

    def test(name, body, writes):
        suite.test(name, gpugen.memexport_shader(12, body, [writes]))

    vector = gpugen.export_vector
    test('mul', [], vector(x.V_MUL, x.r(a), x.r(b)))
    test('add', [], vector(x.V_ADD, x.r(a), x.r(c)))
    test('mad', [], vector(x.V_MAD, x.r(a), x.r(b), x.r(c)))
    test('dp4', [], vector(x.V_DP4, x.r(a), x.r(b)))
    test('dp3', [], vector(x.V_DP3, x.r(a), x.r(b)))
    test('dp2add', [], vector(x.V_DP2ADD, x.r(a), x.r(b), x.r(c)))
    test('max4', [], vector(x.V_MAX4, x.r(a)))
    test('cube', [], vector(x.V_CUBE, x.r(a, 'zzxy'), x.r(a, 'yxzz')))
    # Scalar ops on (A.i, B.i): r4 + i = (A.i, B.i, -, -), then the op reads
    # a from src3's W slot and b from its X slot.
    pairs = []
    for i in range(4):
        lane = 'xyzw'[i]
        pairs.append(x.alu(vector_op=x.V_MAX, vector_dest=4 + i,
                           vector_mask=0b0001, src1=x.r(a, lane),
                           src2=x.r(a, lane)))
        pairs.append(x.alu(vector_op=x.V_MAX, vector_dest=4 + i,
                           vector_mask=0b0010, src1=x.r(b, lane),
                           src2=x.r(b, lane)))
    test('muls', pairs, gpugen.export_scalar_lanes(
        x.S_MULS, lambda i: x.r(4 + i, 'yxxx')))
    for name, op in (('muls_prev2', x.S_MULS_PREV2),
                     ('muls_prev', x.S_MULS_PREV)):
        writes = []
        for i in range(4):
            lane = 'xyzw'[i]
            # ps = C.i, then the op on (a = A.i, b = B.i).
            writes.append(x.alu(scalar_op=x.S_MAXS, scalar_dest=8,
                                scalar_mask=1, src3=x.r(c, lane * 4)))
            writes.append(x.alu(vector_dest=x.EXPORT_DATA0, export=True,
                                scalar_op=op, scalar_mask=1 << i,
                                src3=x.r(4 + i, 'yxxx')))
        test(name, pairs, writes)
