"""The psflow suite's tests whose console results repeat (not the vector
shapes of triangles and points, which the console packs differently from
run to run, nor the a0 read in a cond_exec_pred, which once left one
vector's pixels unwritten), and when control flow reads the predicate:
'lazy' tests set it in clean and dirty execs (setp_gt of a pattern) and
read it with cond_execs (r3, r4), jumps and loop breaks.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import psflow  # noqa: E402
import xenos as x  # noqa: E402

on, mov, add, NOP, TENS, ONES = (psflow.on, psflow.mov, psflow.add,
                                 psflow.NOP, psflow.TENS, psflow.ONES)


def keep(name):
    if name == 'a0 per pixel in exec pred checker':
        return False
    return not (name.startswith('vector ') and
                ('triangle' in name or 'points' in name))


def extra(test):
    # A cond_exec reads it after the dirty exec, the clean setp of none
    # after that read: does the second cond_exec see it?
    test('lazy read twice', lambda s: (
        s.exec_([on('half')], predicate_clean=False),
        s.exec_([on('checker')]),
        s.exec_([mov(3, TENS)], condition=('pred', True)),
        s.exec_([on('none')]),
        s.exec_([mov(4, TENS)], condition=('pred', True))))
    # A dirty exec with no setp after a clean setp.
    test('lazy dirty without setp', lambda s: (
        s.exec_([on('checker')]),
        s.exec_([NOP], predicate_clean=False),
        s.exec_([mov(3, TENS)], condition=('pred', True)),
        s.exec_([mov(4, TENS)], condition=('pred', False))))
    # A jmp on 0 after a read of half and a clean setp of the checker:
    # taken in the left half's vectors if it still sees half.
    test('lazy jmp after read', lambda s: (
        s.exec_([on('half')], predicate_clean=False),
        s.exec_([mov(2, TENS)], condition=('pred', True)),
        s.exec_([on('checker')]),
        s.jmp('target', ('pred', False)),
        s.exec_([mov(3, TENS)]),
        s.label('target'), s.exec_([NOP])))
    # A setp in a cond_exec on a bool constant (no clean bit), in the clean
    # form of it, and in a clean cond_exec_pred.
    test('lazy setp in exec bool', lambda s: (
        s.exec_([on('checker')], condition=('bool', 128, True)),
        s.exec_([mov(3, TENS)], condition=('pred', True)),
        s.exec_([mov(4, TENS)], condition=('pred', False))))
    test('lazy setp in exec bool clean', lambda s: (
        s.exec_([on('checker')], condition=('bool clean', 128, True)),
        s.exec_([mov(3, TENS)], condition=('pred', True)),
        s.exec_([mov(4, TENS)], condition=('pred', False))))
    test('lazy setp in clean exec pred', lambda s: (
        s.exec_([on('all')], predicate_clean=False),
        s.exec_([on('none')], condition=('pred', True)),
        s.exec_([mov(3, TENS)], condition=('pred', True)),
        s.exec_([mov(4, TENS)], condition=('pred', False))))

    # A loop break on 1 after a dirty setp of none, the body setting it
    # everywhere in a clean exec: r2 counts the iterations.
    def loop(s):
        s.exec_([on('none')], predicate_clean=False)
        s.loop_start(psflow.LOOP, 'exit')
        s.label('body')
        s.exec_([on('all'), add(2, x.r(2), ONES)])
        s.loop_end(psflow.LOOP, 'body', predicated_break=True)
        s.label('exit')
    test('lazy loop break after clean setp', loop,
         loops=[0] * 16 + [psflow.loop_constant(4)] + [0] * 15)


def build(suite):
    psflow.build(suite, keep, extra)
