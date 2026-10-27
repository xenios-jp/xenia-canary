"""Control flow and predication in pixel shaders, the flow suite's
counterpart, in the forms games' pixel shaders use them (feature census:
setp, ALU and fetches predicated inside cond_exec_pred, setp_*_push and
setp_pop nested ifs, jmp over a region on the predicate, cond_exec on bool
constants, loops over loop constant 16 reading c[aL + n], kill_gt, a0
relative constants and Direct3D's scalar_dest_rel on exports).

One rectangle over the 64x32 window (or the other draws below) through
Direct3D's Resolve vertex shader; PsParamGen is r0. The pixel shader fetches
three 64x32 textures at its pixel (x, y), p = x + 64y:
- r1 = (x, y, p, q), q = (x & 1) + 2 (y & 1) its place in a 2x2 quad;
- r6 = predicate patterns, +1 or -1: x checker ((x + y) odd, two pixels of
  each quad), y single (the pixels SINGLE), z half (x >= 32), w quad lane 0;
- r7 = (x - 16, t, a, k): t = p % 5 a loop trip count, a = 7p % 13 an a0
  index, k = p % 3 - 1 a kill value.
Every test starts r2-r5 at c8-c11 and memory exports r2-r5 at element 4p
after its control flow (setp_gt on a pattern sets the predicate where it is
+1), so a register a branch skipped keeps its start value and a pixel not
drawn, or not exporting, keeps the sentinel. Tests named 'rt ...' instead
write oC0.x = p + 1 to a 32_FLOAT render target cleared to 0 and resolve it
raw, so a killed pixel stays 0.

'vector' tests find which pixels shade together: per bit of p whether any
pixel of the vector has it set (a cond_exec_pred of an unpredicated write)
and whether all have (a jmp on the predicate over one), and for 16 probe
pixels each which pixels ran its exec.
Constants: c12 = 1s, c13 = (10, 20, 30, 40), c14 = 0s, c15 = (0.5, 0.25,
-0.25, 64), c16 + i = (i, i * i, 100 + i, -i).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gpugen  # noqa: E402
import pm4  # noqa: E402
import xenos as x  # noqa: E402

WIDTH, HEIGHT = 64, 32
PIXELS = WIDTH * HEIGHT
C_START, C_ONES, C_TENS, C_ZEROS, C_MISC, C_TABLE = 8, 12, 13, 14, 15, 16
TABLE = 48
C_BITS = 64      # 2^-b for bits 0-11 of p
C_PROBES = 68    # -P for the probe pixels, 4 vec4s a set
TEX_IDS, TEX_PAT, TEX_PAT2 = 1, 2, 3
LOOP = 16        # the loop constant Direct3D's pixel shaders use
F32_32_32_32 = 38
f2w = gpugen.f2w
SINGLE = (37, 300, 1100, 2000)
PROBES = [
    [0, 1, 7 + 7 * 64, 8, 15 + 3 * 64, 16, 31 + 15 * 64, 32 + 16 * 64,
     63 + 31 * 64, 5 + 20 * 64, 40 + 9 * 64, 17 + 30 * 64, 48 + 24 * 64,
     2 + 9 * 64, 60 + 2 * 64, 33 + 64],
    [3 + 64, 9 + 2 * 64, 23 + 5 * 64, 24 + 8 * 64, 4 + 12 * 64, 12 + 16 * 64,
     20 + 19 * 64, 28 + 22 * 64, 36 + 25 * 64, 44 + 28 * 64, 52 + 31 * 64,
     56 + 4 * 64, 47 + 13 * 64, 39 + 17 * 64, 62 + 11 * 64, 11 + 27 * 64]]
# 3 VS and 16 PS registers, vertex and pixel fetches, PsParamGen, Direct3D's
# PS export mode.
PROGRAM_CNTL = 2 | (15 << 8) | (1 << 16) | (1 << 17) | (1 << 18) | (2 << 27)
C_32_FLOAT = 14


# ALU helpers.

def mov(dest, source, mask=0xF, **keywords):
    return x.alu(vector_op=x.V_MAX, vector_dest=dest, vector_mask=mask,
                 src1=source, src2=source, **keywords)


def add(dest, a, b, mask=0xF, **keywords):
    return x.alu(vector_op=x.V_ADD, vector_dest=dest, vector_mask=mask,
                 src1=a, src2=b, **keywords)


def vop(op, dest, a, b=None, mask=0xF, **keywords):
    return x.alu(vector_op=op, vector_dest=dest, vector_mask=mask,
                 src1=a, src2=b or a, **keywords)


def scalar(op, dest, register, component, mask=1, **keywords):
    """Scalar `op` of r`register`.`component` (src3's W slot) into r`dest`."""
    return x.alu(scalar_op=op, scalar_dest=dest, scalar_mask=mask,
                 src3=x.r(register, 'xxx' + component), **keywords)


def setp(op, register, component):
    """A setp of one component into r9.x (a scratch register)."""
    return scalar(op, 9, register, component)


def pattern(name):
    """(register, component) of a +1/-1 predicate pattern."""
    return (6, {'checker': 'x', 'single': 'y', 'half': 'z', 'lane0': 'w'}
            [name])


def on(name):
    """A setp_gt making the predicate the pattern (or a constant: 'all'
    sets it everywhere, 'none' nowhere)."""
    if name == 'all':
        return x.alu(scalar_op=x.S_SETP_GT, scalar_dest=9, scalar_mask=1,
                     src3=x.c(C_ONES))
    if name == 'none':
        return x.alu(scalar_op=x.S_SETP_GT, scalar_dest=9, scalar_mask=1,
                     src3=x.c(C_ZEROS))
    return setp(x.S_SETP_GT, *pattern(name))


NOP = x.alu(vector_op=x.V_MAX)
TENS = x.c(C_TENS)
ONES = x.c(C_ONES)


def kill(op, dest=9, mask=0, **keywords):
    """Vector kill `op` (24 eq, 25 gt, 26 ge, 27 ne) of k against 0, as
    games' kill_gt with no write mask."""
    return x.alu(vector_op=op, vector_dest=dest, vector_mask=mask,
                 src1=x.r(7, 'wwww'), src2=x.c(C_ZEROS), **keywords)


def kills(op, dest=9, mask=0, **keywords):
    """Scalar kill `op` (35 eq, 36 gt, 37 ge, 38 ne, 39 one) of k."""
    return x.alu(scalar_op=op, scalar_dest=dest, scalar_mask=mask,
                 src3=x.r(7, 'wwww'), **keywords)


KILL_EQ, KILL_GT, KILL_GE, KILL_NE = 24, 25, 26, 27
KILLS_EQ, KILLS_GT, KILLS_GE, KILLS_NE, KILLS_ONE = 35, 36, 37, 38, 39


# Textures.

def textures():
    ids, pat, pat2 = [], [], []
    for p in range(PIXELS):
        px, py = p % WIDTH, p // WIDTH
        q = (px & 1) + 2 * (py & 1)
        ids += [f2w(v) for v in (px, py, p, q)]
        sign = lambda c: 1.0 if c else -1.0
        pat += [f2w(sign(c)) for c in ((px + py) & 1, p in SINGLE,
                                       px >= 32, q == 0)]
        pat2 += [f2w(v) for v in (px - 16.0, p % 5, 7 * p % 13,
                                  p % 3 - 1.0)]
    return {'ids': ids, 'pat': pat, 'pat2': pat2}


def fetch_constants():
    tc = gpugen.texture_constant(F32_32_32_32, WIDTH, HEIGHT)
    return {TEX_IDS: (tc, 'ids'), TEX_PAT: (tc, 'pat'),
            TEX_PAT2: (tc, 'pat2')}


# Shaders.

def prologue(s):
    s.exec_([x.alu(vector_op=x.V_ADD, vector_dest=1, vector_mask=0x3,
                   src1=x.r(0, absolute=True), src2=x.c(C_MISC, 'xxxx'))])
    s.exec_([x.tfetch(6, 1, 'xy', TEX_PAT, denormalized=True).serialized(),
             x.tfetch(7, 1, 'xy', TEX_PAT2, denormalized=True),
             x.tfetch(1, 1, 'xy', TEX_IDS, denormalized=True)])
    starts = [mov(2 + i, x.c(C_START + i)) for i in range(4)]
    starts[0] = starts[0].serialized()
    s.exec_(starts)


def export_shader(flow, exports=None, rt=False):
    """The pixel shader: prologue, `flow`, then the memory exports (each
    entry the ALU instructions of one element) or with `rt` oC0.x = p + 1."""
    s = x.Shader()
    prologue(s)
    if flow:
        flow(s)
    if not rt:
        if exports is None:
            exports = [[gpugen.export_copy(2 + i)] for i in range(4)]
        gpugen.emit_exports(s, exports, x.r(1, 'zzzz'), False)
    s.alloc(x.ALLOC_COLORS, 0)
    s.exec_([x.alu(vector_op=x.V_ADD, vector_dest=0, vector_mask=0xF,
                   export=True, src1=x.r(1, 'zzzz'), src2=ONES)], end=True)
    return s


# Draws, of 2-float vertices.

def rect(l, t, r, b):
    return [(l, t), (r, t), (r, b)]


RECT = rect(-0.5, -0.5, WIDTH - 0.5, HEIGHT - 0.5)
TRIANGLES = [(-0.5, -0.5), (WIDTH - 0.5, -0.5), (-0.5, HEIGHT - 0.5),
             (WIDTH - 0.5, -0.5), (WIDTH - 0.5, HEIGHT - 0.5),
             (-0.5, HEIGHT - 0.5)]
TRIANGLE = [(-0.5, -0.5), (WIDTH - 0.5, -0.5), (-0.5, HEIGHT - 0.5)]
POINTS = [(p % WIDTH, p // WIDTH) for p in range(PIXELS)]
BLOCKS = [('rect', pm4.PRIM_RECTANGLE_LIST, RECT),
          ('triangles', pm4.PRIM_TRIANGLE_LIST, TRIANGLES),
          ('triangle', pm4.PRIM_TRIANGLE_LIST, TRIANGLE),
          ('points', pm4.PRIM_POINT_LIST, POINTS)]


def registers(samples=0):
    """The 64x32 32_FLOAT render target at `samples` (0, 1, 2 for 1x, 2x,
    4x)."""
    state = [('SQ_CONTEXT_MISC', [gpugen.SCREEN_XY]),
             ('RB_SURFACE_INFO', [80 | (samples << 16)]),
             ('RB_COLOR_INFO', [C_32_FLOAT << 16]),
             ('RB_COLOR_MASK', [0xF]),
             ('PA_SC_WINDOW_SCISSOR_BR', [(HEIGHT << 16) | WIDTH]),
             ('PA_SC_AA_CONFIG', [gpugen.AA_CONFIG[samples]])]
    # PA_SU_SC_MODE_CNTL stays the default's, msaa_enable set.
    return state


def loop_constant(count, start=0, step=1):
    return count | (start & 0xFF) << 8 | (step & 0xFF) << 16


def build(suite, keep=None, extra=None):
    """`keep` filters the tests by name, `extra` is called with the test
    function to add more (psflow1)."""
    words, draws = [], {}
    for name, primitive, vertices in BLOCKS:
        draws[name] = (primitive, len(words) // 2, len(vertices))
        words += [f2w(v) for vertex in vertices for v in vertex]
    suite.set_inputs(words, 2)
    suite.export_slots = PIXELS
    for name, data in textures().items():
        suite.add_texture(name, data)

    constants = {256 + C_START + i: [f2w(-1000.0 - 10 * i - j)
                                     for j in range(4)] for i in range(4)}
    constants[256 + C_ONES] = [f2w(1.0)] * 4
    constants[256 + C_TENS] = [f2w(v) for v in (10.0, 20.0, 30.0, 40.0)]
    constants[256 + C_ZEROS] = [0] * 4
    constants[256 + C_MISC] = [f2w(v) for v in (0.5, 0.25, -0.25, 64.0)]
    for i in range(TABLE):
        constants[256 + C_TABLE + i] = [f2w(v)
                                        for v in (i, i * i, 100 + i, -i)]
    for i in range(3):
        constants[256 + C_BITS + i] = [f2w(2.0 ** -(4 * i + j))
                                       for j in range(4)]
    for k, probes in enumerate(PROBES):
        for i in range(4):
            constants[256 + C_PROBES + 4 * k + i] = [
                f2w(-float(p)) for p in probes[4 * i:4 * i + 4]]
    fetches = fetch_constants()
    raw = gpugen.resolve_registers(WIDTH, HEIGHT, 6, 4, command=0)
    clear = gpugen.resolve_registers(WIDTH, HEIGHT, 6, 4, command=0,
                                     clear=(0, 0), clear_enable=1)
    # b128 set, b129 clear, as the bool constants of Direct3D's pixel
    # shaders.
    bools = [0, 0, 0, 0, 1, 0, 0, 0]

    def test(name, flow, block='rect', exports=None, elements=4, rt=False,
             loops=None, samples=0):
        sets = [(pm4.CONSTANT_BOOL, 0, bools)]
        if loops is not None:
            sets.append((pm4.CONSTANT_LOOP, 0, loops))
        shader = export_shader(flow, exports, rt)
        keywords = {}
        if rt:
            keywords = {'resolve': raw, 'clear': clear}
            elements = 1
        suite.test(name, gpugen.resolve_vertex_shader(), constants=constants,
                   elements=elements, pixel_shader=shader,
                   program_cntl=PROGRAM_CNTL, registers=registers(samples),
                   draw=draws[block], fetch_constants=fetches,
                   set_constants=sets, **keywords)

    def write(index, value=ONES):
        """r(2 + index // 4).component(index % 4) = value."""
        return mov(2 + index // 4, value, mask=1 << (index % 4))

    # Vector shape. Bit b of p into r8.x as 0 or 0.5, less 0.25, setp_gt.
    def bit_setp(b):
        k = x.c(C_BITS + b // 4, 'xyzw'[b % 4] * 4)
        return [x.alu(vector_op=x.V_MUL, vector_dest=8, vector_mask=1,
                      src1=x.r(1, 'zzzz'), src2=k),
                vop(x.V_FLOOR, 8, x.r(8, 'xxxx'), mask=1),
                x.alu(vector_op=x.V_MUL, vector_dest=8, vector_mask=1,
                      src1=x.r(8, 'xxxx'), src2=x.c(C_MISC, 'xxxx')),
                vop(x.V_FRC, 8, x.r(8, 'xxxx'), mask=1),
                add(8, x.r(8, 'xxxx'), x.c(C_MISC, 'zzzz'), mask=1),
                setp(x.S_SETP_GT, 8, 'x')]

    def any_bits(s):
        for b in range(11):
            s.exec_(bit_setp(b), predicate_clean=False)
            s.exec_([write(b)], condition=('pred', True))

    def all_bits(value):
        def flow(s):
            for b in range(11):
                s.exec_(bit_setp(b), predicate_clean=False)
                s.jmp(f'bit {b}', ('pred', value))
                s.exec_([write(b)])
                s.label(f'bit {b}')
            s.exec_([NOP])
        return flow

    def probes(k):
        def flow(s):
            for i in range(16):
                s.exec_([add(8, x.r(1, 'zzzz'),
                             x.c(C_PROBES + 4 * k + i // 4,
                                 'xyzw'[i % 4] * 4), mask=1),
                         setp(x.S_SETP_EQ, 8, 'x')], predicate_clean=False)
                s.exec_([write(i)], condition=('pred', True))
        return flow

    for block in ('rect', 'triangles', 'triangle', 'points'):
        test(f'vector any bits {block}', any_bits, block)
        test(f'vector all bits {block}', all_bits(True), block)
        test(f'vector probes 0 {block}', probes(0), block)
        if block == 'rect':
            test(f'vector none bits {block}', all_bits(False), block)
            test(f'vector probes 1 {block}', probes(1), block)
    # At 4x, as most of games' draws.
    test('vector any bits rect 4x', any_bits, samples=2)
    test('vector probes 0 rect 4x', probes(0), samples=2)
    test('vector probes 0 triangles 4x', probes(0), 'triangles', samples=2)

    # Predicate setting ops on x - 16 per pixel, then ALU instructions
    # predicated each way: r5.x the op's result, r3 written where the
    # predicate is set, r4 where it is clear.
    def predicated(*ops):
        return [*ops, mov(3, TENS, predicate=True),
                mov(4, TENS, predicate=False)]

    test('predicate initial', lambda s: s.exec_(predicated()))
    for name, op in (('eq', x.S_SETP_EQ), ('ne', x.S_SETP_NE),
                     ('gt', x.S_SETP_GT), ('ge', x.S_SETP_GE),
                     ('inv', x.S_SETP_INV), ('pop', x.S_SETP_POP),
                     ('clr', x.S_SETP_CLR), ('rstr', x.S_SETP_RSTR)):
        test(f'setp_{name}', lambda s, op=op: s.exec_(
            predicated(scalar(op, 5, 7, 'x')), predicate_clean=False))
    for name, op in (('eq', 20), ('ne', 21), ('gt', 22), ('ge', 23)):
        test(f'setp_{name}_push', lambda s, op=op: s.exec_(predicated(
            x.alu(vector_op=op, vector_dest=5, vector_mask=0xF,
                  src1=x.r(7), src2=x.r(7, 'yxwz'))), predicate_clean=False))

    # Execs on the predicate, with an unpredicated write (the whole vector
    # if any pixel passes?), and as Direct3D's, predicated too.
    def on_predicate(name, value=True, alu_predicated=False, clean=False):
        def flow(s):
            s.exec_([on(name)], predicate_clean=clean)
            s.exec_([mov(3, TENS, predicate=value if alu_predicated
                         else None)], condition=('pred', value))
            s.exec_([mov(4, TENS, predicate=(not value) if alu_predicated
                         else None)], condition=('pred', not value))
        return flow
    for name in ('checker', 'lane0', 'half', 'single', 'none', 'all'):
        test(f'exec pred 1 {name}', on_predicate(name))
    test('exec pred 0 checker', on_predicate('checker', False))
    test('exec pred 1 alu predicated checker',
         on_predicate('checker', True, True))
    test('exec pred 1 clean setp checker',
         on_predicate('checker', clean=True))
    # A dirty setp of half, then a clean one of the checker: which the
    # cond_exec sees (r3) and which the ALU (r4).
    test('exec pred clean after dirty', lambda s: (
        s.exec_([on('half')], predicate_clean=False),
        s.exec_([on('checker')], predicate_clean=True),
        s.exec_([mov(3, TENS)], condition=('pred', True)),
        s.exec_([mov(4, TENS, predicate=True)])))
    # Direct3D's cond_exec on a bool constant (predicate clean form) with
    # its ALU predicated from an earlier setp.
    test('exec bool clean predicated alu', lambda s: (
        s.exec_([on('checker')], predicate_clean=False),
        s.exec_([mov(3, TENS, predicate=True)],
                condition=('bool clean', 128, True)),
        s.exec_([mov(4, TENS, predicate=True)],
                condition=('bool clean', 128, False)),
        s.exec_([mov(5, TENS, predicate=True)],
                condition=('bool clean', 129, False))))

    # Texture fetches inside a cond_exec_pred, predicated as Direct3D's and
    # not: r3 = ids texel where they run.
    def predicated_fetch(instruction, value):
        w0, w1, w2 = instruction.words
        w1 |= 1 << 31
        w2 = (w2 & 0x7FFFFFFF) | (int(value) << 31)
        return x.Instruction((w0, w1, w2), True)

    def fetch_in_exec(predicate):
        def flow(s):
            s.exec_([on('checker'), add(8, x.r(1), x.c(C_MISC, 'xxxx'),
                                       mask=0x3)], predicate_clean=False)
            fetch = x.tfetch(3, 8, 'xy', TEX_IDS, denormalized=True)
            if predicate:
                fetch = predicated_fetch(fetch, True)
            s.exec_([fetch.serialized()], condition=('pred', True))
            s.exec_([mov(4, x.r(3), predicate=True).serialized()])
        return flow
    test('tfetch predicated in exec pred checker', fetch_in_exec(True))
    test('tfetch in exec pred checker', fetch_in_exec(False))

    # Jumps over an exec writing r2 to one writing r3: on the predicate (the
    # region unpredicated, as Halo 3's), always and on bool constants.
    def jump(condition, first=None, clean=False):
        def flow(s):
            if first:
                s.exec_([on(first)], predicate_clean=clean)
            s.jmp('target', condition)
            s.exec_([mov(2, TENS)])
            s.label('target')
            s.exec_([mov(3, TENS)])
        return flow
    for name in ('checker', 'lane0', 'half', 'single', 'none', 'all'):
        test(f'jmp pred 0 {name}', jump(('pred', False), name))
    test('jmp pred 1 checker', jump(('pred', True), 'checker'))
    test('jmp pred 1 all', jump(('pred', True), 'all'))
    test('jmp pred 1 all clean setp', jump(('pred', True), 'all', True))
    test('jmp', jump(None))
    test('jmp b128 1', jump(('bool', 128, True)))
    test('jmp b128 0', jump(('bool', 128, False)))

    # Direct3D's nested if (4B4E0823's form): setp_gt_push of the checker
    # with a counter in r9, a jmp on the predicate past the if, the outer
    # body (r2) predicated in a cond_exec_pred, the inner if (half, r3)
    # pushed and popped in one exec on b128, then the rest of the outer
    # body (r4); r5 = the counter.
    def nested(predicate_bodies=True):
        p = True if predicate_bodies else None

        def flow(s):
            s.exec_([mov(9, x.c(C_ZEROS)),
                     x.alu(vector_op=22, vector_dest=9, vector_mask=0xF,
                           src1=x.r(9), src2=x.r(6, 'xxxx'))],
                    predicate_clean=False)
            s.jmp('endif', ('pred', False))
            s.exec_([mov(2, TENS, predicate=p)], condition=('pred', True))
            s.exec_([x.alu(vector_op=22, vector_dest=9, vector_mask=0xF,
                           src1=x.r(9), src2=x.r(6, 'zzzz')),
                     mov(3, TENS, predicate=p),
                     scalar(x.S_SETP_POP, 9, 9, 'x', mask=0x9)],
                    condition=('bool', 128, True))
            s.exec_([mov(4, TENS, predicate=p)], condition=('pred', True))
            s.label('endif')
            s.exec_([mov(5, x.r(9))])
        return flow
    test('nested if', nested())
    test('nested if unpredicated', nested(False))

    # Calls: a subroutine adding 10 to r2, then r3 = r2 + 1 after it.
    def calls(condition=None, first=None):
        def flow(s):
            if first:
                s.exec_([on(first)], predicate_clean=False)
            s.call('sub', condition)
            s.exec_([add(3, x.r(2), ONES)])
            s.jmp('done')
            s.label('sub')
            s.exec_([add(2, x.r(2), TENS)])
            s.ret()
            s.label('done')
            s.exec_([NOP])
        return flow
    test('call', calls())
    test('call pred 1 checker', calls(('pred', True), 'checker'))
    test('call pred 1 single', calls(('pred', True), 'single'))

    # Loops over loop constant 16: r2 counts the iterations, r3 sums
    # c[aL + 16], r4 is the last read.
    def loop(loop_index=LOOP):
        def flow(s):
            s.loop_start(loop_index, 'exit')
            s.label('body')
            s.exec_([add(2, x.r(2), ONES),
                     add(3, x.r(3), x.c(C_TABLE), loop_relative=True),
                     mov(4, x.c(C_TABLE), loop_relative=True)])
            s.loop_end(loop_index, 'body')
            s.label('exit')
        return flow

    def loops(**by_index):
        words = [0] * 32
        for k, v in by_index.items():
            words[int(k[1:])] = v
        return words
    for count, start, step in ((5, 0, 1), (3, 2, 3), (0, 0, 1), (4, 40, -7)):
        test(f'loop {count} from {start} step {step}', loop(),
             loops=loops(i16=loop_constant(count, start, step)))
    # Which loop constant a pixel shader's loop 16 and loop 0 read.
    both = loops(i0=loop_constant(2, 0, 1), i16=loop_constant(4, 1, 2))
    test('loop constant 16', loop(16), loops=both)
    test('loop constant 0', loop(0), loops=both)

    # Halo 3's light loop: in each iteration a dirty exec sets the
    # predicate where aL == t and accumulates predicated, a cond_exec_pred
    # and a plain exec of predicated ALU follow; r5 counts unpredicated.
    def light_loop(s):
        s.loop_start(LOOP, 'exit')
        s.label('body')
        s.exec_([add(8, x.r(7, 'yyyy'), x.c(C_TABLE, 'xxxx', negate=True),
                     mask=1, loop_relative=True),
                 setp(x.S_SETP_EQ, 8, 'x'),
                 add(2, x.r(2), x.c(C_TABLE), predicate=True,
                     loop_relative=True),
                 add(5, x.r(5), ONES)], predicate_clean=False)
        s.exec_([add(3, x.r(3), TENS, predicate=True)],
                condition=('pred', True))
        s.exec_([add(4, x.r(4), ONES, predicate=True)])
        s.loop_end(LOOP, 'body')
        s.label('exit')
    test('loop setp per iteration', light_loop,
         loops=loops(i16=loop_constant(6)))

    # Halo 3's predicated loop: the predicate set before it (checker), a
    # cond_exec_pred body, break where it is clear; r3 counts the
    # iterations unpredicated.
    def break_before(s):
        s.exec_([on('checker')], predicate_clean=False)
        s.loop_start(LOOP, 'exit')
        s.label('body')
        s.exec_([add(2, x.r(2), ONES, predicate=True)],
                condition=('pred', True))
        s.exec_([add(3, x.r(3), ONES)])
        s.loop_end(LOOP, 'body', predicated_break=False)
        s.label('exit')
    test('loop break pred 0 checker', break_before,
         loops=loops(i16=loop_constant(4)))

    # A break once aL >= t, set in the body (aL - t >= 0 for a break on 1,
    # t - aL > 0 on 0): r2 counts the iterations, t + 1 a pixel.
    def break_per_pixel(value, clean=False):
        def flow(s):
            s.loop_start(LOOP, 'exit')
            s.label('body')
            s.exec_([add(2, x.r(2), ONES),
                     add(8, x.c(C_TABLE, 'xxxx', negate=not value),
                         x.r(7, 'yyyy', negate=value), mask=1,
                         loop_relative=True),
                     setp(x.S_SETP_GE if value else x.S_SETP_GT, 8, 'x')],
                    predicate_clean=clean)
            s.loop_end(LOOP, 'body', predicated_break=value)
            s.label('exit')
        return flow
    test('loop break pred 1 at t', break_per_pixel(True),
         loops=loops(i16=loop_constant(8)))
    test('loop break pred 0 at t', break_per_pixel(False),
         loops=loops(i16=loop_constant(8)))

    # aL-relative temporaries: r[aL + 2] written for aL 0-3; read back.
    test('loop relative dest', lambda s: (
        s.loop_start(LOOP, 'exit'), s.label('body'),
        s.exec_([add(2, TENS, x.c(C_TABLE), loop_relative=True,
                     vector_dest_relative=True)]),
        s.loop_end(LOOP, 'body'), s.label('exit')),
        loops=loops(i16=loop_constant(4)))
    test('loop relative source', lambda s: (
        s.exec_([mov(3, x.r(1)), mov(4, TENS)]),
        s.loop_start(LOOP, 'exit'), s.label('body'),
        s.exec_([add(5, x.r(5), x.r(2, relative=True))]),
        s.loop_end(LOOP, 'body'), s.label('exit')),
        loops=loops(i16=loop_constant(3)))
    # Nested loops: aL is the inner one's inside it, the outer's after.
    test('loop nested', lambda s: (
        s.loop_start(LOOP, 'exit'), s.label('body'),
        s.exec_([add(2, x.r(2), ONES)]),
        s.loop_start(LOOP + 1, 'inner exit'), s.label('inner body'),
        s.exec_([add(5, x.r(5), x.c(C_TABLE), loop_relative=True)]),
        s.loop_end(LOOP + 1, 'inner body'), s.label('inner exit'),
        s.exec_([mov(4, x.c(C_TABLE), loop_relative=True)]),
        s.loop_end(LOOP, 'body'), s.label('exit')),
        loops=loops(i16=loop_constant(3, 1, 4), i17=loop_constant(2, 10, 1)))

    # a0 per pixel (7p % 13), c[a0 + 16] read an instruction later; and as
    # Direct3D's, in a cond_exec_pred, predicated.
    def a0_read(predicate=None):
        return [x.alu(vector_op=x.V_MAXA, vector_dest=8, vector_mask=0xF,
                      src1=x.r(7, 'zzzz'), src2=x.r(7, 'zzzz'),
                      predicate=predicate),
                x.alu(vector_op=x.V_MAX, predicate=predicate),
                x.alu(vector_op=x.V_MAX, vector_dest=2, vector_mask=0xF,
                      src1=x.c(C_TABLE), src2=x.c(C_TABLE), a0_relative=True,
                      predicate=predicate)]
    test('a0 per pixel', lambda s: s.exec_(a0_read()))
    test('a0 per pixel in exec pred checker', lambda s: (
        s.exec_([on('checker')], predicate_clean=False),
        s.exec_(a0_read(True), condition=('pred', True))))

    # Kills of k against 0 and what follows them: do killed pixels still
    # export to memory (kill first), what a kill writes (r5), a kill in a
    # cond_exec_pred predicated as Direct3D's and not, and a predicated kill
    # where the predicate is clear.
    test('kill_gt then export', lambda s: s.exec_([kill(KILL_GT)]))
    test('kill_gt result', lambda s: s.exec_([kill(KILL_GT, 5, 0xF)]))
    test('kills_gt result', lambda s: s.exec_([kills(KILLS_GT, 5, 0xF)]))
    test('export then kill_gt', lambda s: None, exports=[
        [gpugen.export_copy(2 + i)] + ([kill(KILL_GT)] if i == 3 else [])
        for i in range(4)])

    def kill_in_exec(predicate, name='checker'):
        def flow(s):
            s.exec_([on(name)], predicate_clean=False)
            s.exec_([kill(KILL_GT, predicate=predicate),
                     mov(3, TENS, predicate=predicate)],
                    condition=('pred', True))
        return flow
    # After a kill of the checker's set pixels: whether a jmp and a
    # cond_exec on a predicate set only on killed pixels still count them.
    def after_kill(s):
        s.exec_([x.alu(vector_op=KILL_GT, vector_dest=9, vector_mask=0,
                       src1=x.r(6, 'xxxx'), src2=x.c(C_ZEROS))])
        s.exec_([on('checker')], predicate_clean=False)
        s.jmp('target', ('pred', False))
        s.exec_([mov(2, TENS)])
        s.label('target')
        s.exec_([mov(3, TENS)], condition=('pred', True))
    test('jmp and exec pred after kill', after_kill)

    for rt in (False, True):
        prefix = 'rt ' if rt else ''
        test(f'{prefix}kill in exec pred predicated checker',
             kill_in_exec(True), rt=rt)
        test(f'{prefix}kill in exec pred checker', kill_in_exec(None),
             rt=rt)
        test(f'{prefix}kill predicated none', lambda s: (
            s.exec_([on('none')], predicate_clean=False),
            s.exec_([kill(KILL_GT, predicate=True)])), rt=rt)
    for name, op in (('kill_eq', KILL_EQ), ('kill_gt', KILL_GT),
                     ('kill_ge', KILL_GE), ('kill_ne', KILL_NE)):
        test(f'rt {name}', lambda s, op=op: s.exec_([kill(op)]), rt=True)
    for name, op in (('kills_eq', KILLS_EQ), ('kills_gt', KILLS_GT),
                     ('kills_ge', KILLS_GE), ('kills_ne', KILLS_NE),
                     ('kills_one', KILLS_ONE)):
        test(f'rt {name}', lambda s, op=op: s.exec_([kills(op)]), rt=True)
    test('rt jmp pred 0 over kill checker', lambda s: (
        s.exec_([on('checker')], predicate_clean=False),
        s.jmp('target', ('pred', False)),
        s.exec_([kill(KILL_GT)]),
        s.label('target'), s.exec_([NOP])), rt=True)

    # Direct3D's exports set scalar_dest_rel: element 0 exported as eM0
    # with it (or vector_dest_rel), outside a loop, inside one with aL = 1
    # and after it. With aL added it would land in element 1.
    def relative_export(scalar_rel=True):
        return [[x.alu(vector_op=x.V_MAX, vector_dest=x.EXPORT_DATA0,
                       vector_mask=0xF, export=True, src1=x.r(2),
                       src2=x.r(2), scalar_dest_relative=scalar_rel,
                       vector_dest_relative=not scalar_rel)], []]

    def in_loop(exports):
        def flow(s):
            s.loop_start(LOOP, 'exit')
            s.label('body')
            gpugen.emit_exports(s, exports, x.r(1, 'zzzz'), False)
            s.loop_end(LOOP, 'body')
            s.label('exit')
        return flow
    aL1 = loops(i16=loop_constant(1, 1, 0))
    test('export scalar dest relative', None, exports=relative_export(),
         elements=2)
    test('export scalar dest relative in loop aL 1',
         in_loop(relative_export()), exports=[[], []], elements=2, loops=aL1)
    test('export vector dest relative in loop aL 1',
         in_loop(relative_export(False)), exports=[[], []], elements=2,
         loops=aL1)
    test('export scalar dest relative after loop aL 1', lambda s: (
        s.loop_start(LOOP, 'exit'), s.label('body'), s.exec_([NOP]),
        s.loop_end(LOOP, 'body'), s.label('exit')),
        exports=relative_export(), elements=2, loops=aL1)
    if extra:
        extra(test)
    if keep:
        suite.tests = [t for t in suite.tests if keep(t.name)]
