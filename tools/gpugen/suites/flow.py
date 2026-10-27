"""Shader control flow and predication in vertex shaders, as games' shaders
use them: setp ops and predicated ALU instructions, execs conditional on
the predicate or a bool constant, jumps, calls, loops over loop constants
with aL-relative constants and temporaries, and predicated loop breaks.

64 vertices, vertex k fetched into r1 = (k - 32, k % 3 - 1, k / 64, k % 2),
so predicates differ between vertices (and between those a vertex vector
shades together). An exec setting the predicate for control flow is
predicate-dirty, as Direct3D's; the 'clean setp' tests set it in a clean
exec, which control flow doesn't see. Every test starts r2-r5 at c8-c11 and
exports r2-r5 after its control flow, so a register a branch skipped keeps
its start value.
Constants: c12 = 1s, c13 = (10, 20, 30, 40), c14 = 0s, c16 + i =
(i, i * i, 100 + i, -i) for aL-relative reads.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gpugen  # noqa: E402
import pm4  # noqa: E402
import xenos as x  # noqa: E402

VERTICES = 64
C_START, C_ONES, C_TENS, C_ZEROS, C_TABLE = 8, 12, 13, 14, 16
TABLE = 64
f2w = gpugen.f2w
IN = gpugen.INPUT_REGISTER


def mov(dest, source, **keywords):
    return x.alu(vector_op=x.V_MAX, vector_dest=dest, vector_mask=0xF,
                 src1=source, src2=source, **keywords)


def add(dest, a, b, **keywords):
    return x.alu(vector_op=x.V_ADD, vector_dest=dest, vector_mask=0xF,
                 src1=a, src2=b, **keywords)


def scalar(op, dest, operand, **keywords):
    """Scalar `op` of one operand (src3's W slot) into r`dest`.x."""
    return x.alu(scalar_op=op, scalar_dest=dest, scalar_mask=1,
                 src3=operand, **keywords)


def operand(register, component):
    return x.r(register, 'xxx' + component)


def setp(op, register=IN, component='x'):
    return scalar(op, 5, operand(register, component))


def loop_constant(count, start=0, step=1):
    return count | (start & 0xFF) << 8 | (step & 0xFF) << 16


def inputs():
    return [f2w(v) for k in range(VERTICES)
            for v in (k - 32.0, k % 3 - 1.0, k / 64.0, float(k % 2))]


def build(suite):
    suite.set_inputs(inputs(), 4)
    constants = {C_START + i: [f2w(-1000.0 - 10 * i - j) for j in range(4)]
                 for i in range(4)}
    constants[C_ONES] = [f2w(1.0)] * 4
    constants[C_TENS] = [f2w(v) for v in (10.0, 20.0, 30.0, 40.0)]
    constants[C_ZEROS] = [0] * 4
    for i in range(TABLE):
        constants[C_TABLE + i] = [f2w(v) for v in (i, i * i, 100 + i, -i)]
    starts = [mov(2 + i, x.c(C_START + i)) for i in range(4)]
    exports = [[gpugen.export_copy(2 + i)] for i in range(4)]

    def test(name, flow, bools=None, loops=None, export=exports):
        sets = []
        if bools is not None:
            sets.append((pm4.CONSTANT_BOOL, 0, bools))
        if loops is not None:
            sets.append((pm4.CONSTANT_LOOP, 0, loops))
        suite.test(name, gpugen.memexport_shader(4, starts, export,
                                                 flow=flow),
                   constants=constants, elements=4, set_constants=sets)

    tens = x.c(C_TENS)

    # Predicate setting ops on each vertex's x and y, then ALU instructions
    # predicated each way: r5.x the op's result, r3 written where the
    # predicate is set, r4 where it is clear.
    def predicated(*ops):
        return [*ops, mov(3, tens, predicate=True),
                mov(4, tens, predicate=False)]

    test('predicate initial', lambda s: s.exec_(predicated()))
    for name, op in (('eq', x.S_SETP_EQ), ('ne', x.S_SETP_NE),
                     ('gt', x.S_SETP_GT), ('ge', x.S_SETP_GE),
                     ('inv', x.S_SETP_INV), ('pop', x.S_SETP_POP),
                     ('clr', x.S_SETP_CLR), ('rstr', x.S_SETP_RSTR)):
        for component in 'xy':
            test(f'setp_{name} {component}', lambda s, op=op, c=component:
                 s.exec_(predicated(setp(op, IN, c))))
        # After the predicate is set from y, and on x.
        test(f'setp_{name} x after setp_gt y', lambda s, op=op:
             s.exec_(predicated(setp(x.S_SETP_GT, IN, 'y'), setp(op))))
    for name, op in (('eq', 20), ('ne', 21), ('gt', 22), ('ge', 23)):
        test(f'setp_{name}_push', lambda s, op=op: s.exec_(predicated(
            x.alu(vector_op=op, vector_dest=5, vector_mask=0xF,
                  src1=x.r(IN), src2=x.r(IN, 'yxwz')))))
    # A predicated instruction writing only some components, and one whose
    # vector and scalar ops write different registers.
    test('predicated scalar', lambda s: s.exec_([
        setp(x.S_SETP_GT),
        scalar(x.S_MAXS, 3, operand(IN, 'y'), predicate=True),
        x.alu(vector_op=x.V_ADD, vector_dest=4, vector_mask=0x5,
              src1=tens, src2=x.r(IN), scalar_op=x.S_MAXS, scalar_dest=2,
              scalar_mask=0x2, src3=operand(IN, 'x'), predicate=False)]))

    # Execs on the predicate: whether a vertex vector runs one when any or
    # all of its vertices pass, with and without the ALU predicated too.
    def on_predicate(s, value, alu_predicated, clean=False):
        s.exec_([setp(x.S_SETP_GT)], predicate_clean=clean)
        s.exec_([mov(3, tens, predicate=value if alu_predicated else None)],
                condition=('pred', value))
        s.exec_([mov(4, tens)], condition=('pred', not value))
    for value in (True, False):
        for alu_predicated in (False, True):
            test(f'exec pred {int(value)}' +
                 (' alu predicated' if alu_predicated else ''),
                 lambda s, v=value, a=alu_predicated: on_predicate(s, v, a))
    test('exec pred 1 clean setp', lambda s: on_predicate(s, True, False,
                                                          True))
    test('exec pred all set', lambda s: (
        s.exec_([setp(x.S_SETP_GE, IN, 'z')], predicate_clean=False),
        s.exec_([mov(3, tens)], condition=('pred', True)),
        s.exec_([mov(4, tens)], condition=('pred', False))))

    # Execs on bool constants: b0 and b2 set, b1 and b3 clear, b100 set.
    bools = [0x5, 0, 0, 1 << 4, 0, 0, 0, 0]

    def on_bools(form):
        def flow(s):
            for i, (constant, value) in enumerate(
                    ((0, True), (1, True), (0, False), (100, True))):
                s.exec_([mov(2 + i, tens)],
                        condition=(form, constant, value))
        return flow
    test('exec bool', on_bools('bool'), bools)
    test('exec bool clean', on_bools('bool clean'), bools)
    test('exec bool all clear', on_bools('bool'), [0] * 8)

    # Jumps: always, on a bool constant, and on the predicate (over an exec
    # writing r2, then r3 at the target).
    def jump(condition, first=(), clean=False):
        def flow(s):
            if first:
                s.exec_(list(first), predicate_clean=clean)
            s.jmp('target', condition)
            s.exec_([mov(2, tens)])
            s.label('target')
            s.exec_([mov(3, tens)])
        return flow
    test('jmp', jump(None))
    for constant, value in ((0, True), (0, False), (1, True), (1, False)):
        test(f'jmp b{constant} {int(value)}',
             jump(('bool', constant, value)), bools)
    for value in (True, False):
        test(f'jmp pred {int(value)}',
             jump(('pred', value), [setp(x.S_SETP_GT)]))
    test('jmp pred 1 all set',
         jump(('pred', True), [setp(x.S_SETP_GE, IN, 'z')]))
    test('jmp pred 1 all set clean setp',
         jump(('pred', True), [setp(x.S_SETP_GE, IN, 'z')], True))

    # Calls: a subroutine writing r2, then r3 after the return; a call not
    # taken; nested calls.
    def calls(condition=None, nested=False, first=()):
        def flow(s):
            if first:
                s.exec_(list(first), predicate_clean=False)
            s.call('sub', condition)
            s.exec_([add(3, x.r(2), x.c(C_ONES))])
            s.jmp('done')
            s.label('sub')
            s.exec_([add(2, x.r(2), tens)])
            if nested:
                s.call('inner')
            s.ret()
            if nested:
                s.label('inner')
                s.exec_([add(4, x.r(2), tens)])
                s.ret()
            s.label('done')
            s.exec_([add(5, x.r(5), x.c(C_ONES))])
        return flow
    test('call', calls())
    test('call nested', calls(nested=True))
    test('call b1 1', calls(('bool', 1, True)), bools)
    test('call b0 1', calls(('bool', 0, True)), bools)
    test('call pred 1', calls(('pred', True), first=[setp(x.S_SETP_GT)]))

    # Loops: r2 counts the iterations, r3 sums c[aL + 16], r4 is the last
    # read; then loop constant starts, steps and a count of zero.
    def loop(body_extra=(), repeat=False, nested=None, before=()):
        def flow(s):
            if before:
                s.exec_(list(before))
            s.loop_start(0, 'exit', repeat)
            s.label('body')
            s.exec_([add(2, x.r(2), x.c(C_ONES)),
                     add(3, x.r(3), x.c(C_TABLE), loop_relative=True),
                     mov(4, x.c(C_TABLE), loop_relative=True),
                     *body_extra])
            if nested:
                nested(s)
            s.loop_end(0, 'body')
            s.label('exit')
        return flow
    for count, start, step in ((5, 0, 1), (3, 2, 3), (0, 0, 1),
                               (4, 40, -7), (1, 0, 0), (255, 0, 0)):
        test(f'loop {count} from {start} step {step}', loop(),
             loops=[loop_constant(count, start, step)])
    # The loop constant a loop names, not the first.
    test('loop constant 5', lambda s: (
        s.loop_start(5, 'exit'), s.label('body'),
        s.exec_([add(2, x.r(2), x.c(C_ONES)),
                 mov(4, x.c(C_TABLE), loop_relative=True)]),
        s.loop_end(5, 'body'), s.label('exit')),
        loops=[loop_constant(2)] + [0] * 4 + [loop_constant(6, 3, 2)])

    # Nested loops: aL is the inner loop's inside it and the outer's after.
    def inner(s):
        s.loop_start(1, 'inner exit')
        s.label('inner body')
        s.exec_([add(5, x.r(5), x.c(C_TABLE), loop_relative=True)])
        s.loop_end(1, 'inner body')
        s.label('inner exit')
        s.exec_([mov(4, x.c(C_TABLE), loop_relative=True)])
    test('loop nested', loop(nested=inner),
         loops=[loop_constant(3, 1, 4), loop_constant(2, 10, 1)])

    def repeat(s):
        s.loop_start(1, 'inner exit', repeat=True)
        s.label('inner body')
        s.exec_([add(5, x.r(5), x.c(C_TABLE), loop_relative=True)])
        s.loop_end(1, 'inner body')
        s.label('inner exit')
    test('loop nested repeat', loop(nested=repeat),
         loops=[loop_constant(3, 1, 4), loop_constant(2, 10, 1)])

    # A break where the predicate is set: each vertex leaves once
    # r1.x + r2.x >= 0, r2 starting at 0.
    def breaking(value, clean=False):
        def flow(s):
            s.exec_([mov(2, x.c(C_ZEROS))])
            s.loop_start(0, 'exit')
            s.label('body')
            s.exec_([add(2, x.r(2), x.c(C_ONES)),
                     add(3, x.r(IN), x.r(2)),
                     setp(x.S_SETP_GE, 3, 'x')], predicate_clean=clean)
            s.loop_end(0, 'body', predicated_break=value)
            s.label('exit')
        return flow
    for value in (True, False):
        test(f'loop break pred {int(value)}', breaking(value),
             loops=[loop_constant(40)])
    test('loop break pred 1 clean setp', breaking(True, True),
         loops=[loop_constant(40)])

    # aL-relative temporaries: r[aL + 2] written for aL 0-3, then read back.
    test('loop relative dest', lambda s: (
        s.loop_start(0, 'exit'), s.label('body'),
        s.exec_([add(2, tens, x.c(C_TABLE), loop_relative=True,
                     vector_dest_relative=True)]),
        s.loop_end(0, 'body'), s.label('exit')),
        loops=[loop_constant(4)])
    test('loop relative source', lambda s: (
        s.exec_([mov(3, x.r(IN)), mov(4, tens)]),
        s.loop_start(0, 'exit'), s.label('body'),
        s.exec_([add(5, x.r(5), x.r(2, relative=True))]),
        s.loop_end(0, 'body'), s.label('exit')),
        loops=[loop_constant(3)])

    # Direct3D's exports set scalar_dest_rel.
    test('export scalar dest relative', None, export=[[x.alu(
        vector_op=x.V_MAX, vector_dest=x.EXPORT_DATA0, vector_mask=0xF,
        export=True, src1=x.r(2 + i), src2=x.r(2 + i),
        scalar_dest_relative=True)] for i in range(4)])
