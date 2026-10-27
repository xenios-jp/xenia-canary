"""When a dirty exec (no predicate clean bit) that doesn't run makes control
flow take a new copy of the predicate: psflow1's 'lazy' tests run every
dirty exec, these skip one, on a bool constant, on the predicate itself or
by a jmp over it.

Each test sets the predicate in a dirty exec and reads it with a cond_exec
writing r2, which takes control flow's copy, then sets it differently in a
clean exec, then has the dirty exec that may not run, and then reads it
with cond_execs writing r3 (on 1) and r4 (on 0). Where the skipped exec
counts as dirty, r3 and r4 show the second predicate; where it doesn't,
the first. The predicates are 'all', 'none' and 'half' (the left 32
columns), the same over each 8x8 vector, so the any and all rules of the
pixel vectors don't come in.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import psflow  # noqa: E402

on, mov, NOP, TENS = psflow.on, psflow.mov, psflow.NOP, psflow.TENS

NAMES = set()


def extra(test):
    def lazy(name, first, second, between):
        """`between` adds the control flow between the clean setp of
        `second` and the reads."""
        def flow(s):
            s.exec_([on(first)], predicate_clean=False)
            s.exec_([mov(2, TENS)], condition=('pred', True))
            s.exec_([on(second)])
            between(s)
            s.exec_([mov(3, TENS)], condition=('pred', True))
            s.exec_([mov(4, TENS)], condition=('pred', False))
        NAMES.add(name)
        test(name, flow)

    # b128 is set, b129 clear.
    lazy('dirty exec bool run', 'all', 'none', lambda s: s.exec_(
        [NOP], condition=('bool', 128, True), predicate_clean=False))
    lazy('dirty exec bool skipped', 'all', 'none', lambda s: s.exec_(
        [NOP], condition=('bool', 128, False), predicate_clean=False))
    lazy('dirty exec bool skipped b129', 'none', 'all', lambda s: s.exec_(
        [NOP], condition=('bool', 129, True), predicate_clean=False))
    lazy('dirty exec pred run', 'all', 'none', lambda s: s.exec_(
        [NOP], condition=('pred', True), predicate_clean=False))
    lazy('dirty exec pred skipped', 'all', 'none', lambda s: s.exec_(
        [NOP], condition=('pred', False), predicate_clean=False))
    # The dirty exec runs in the left half's vectors only.
    lazy('dirty exec pred half', 'half', 'all', lambda s: s.exec_(
        [NOP], condition=('pred', True), predicate_clean=False))
    lazy('dirty exec pred half 0', 'half', 'none', lambda s: s.exec_(
        [NOP], condition=('pred', False), predicate_clean=False))

    def jump_over(s):
        s.jmp('over', ('bool', 128, True))
        s.exec_([NOP], predicate_clean=False)
        s.label('over')
    lazy('dirty exec jumped over', 'all', 'none', jump_over)

    def jump_not_taken(s):
        s.jmp('over', ('bool', 129, True))
        s.exec_([NOP], predicate_clean=False)
        s.label('over')
    lazy('dirty exec after jmp not taken', 'all', 'none', jump_not_taken)

    # A jmp on the predicate between reads: does the jmp itself, which
    # reads the copy, count as a read that clears a pending refresh?
    def jump_pred(s):
        s.exec_([NOP], predicate_clean=False)
        s.jmp('over', ('pred', False))
        s.label('over')
    lazy('dirty exec then jmp pred', 'all', 'none', jump_pred)


def build(suite):
    psflow.build(suite, lambda name: name in NAMES, extra)
