"""Predicated draws, vizpred.py's tests with the sample counters reset: the
shared build of vizdet1 and vizdet2.

vizpred's three console runs gave three different results for every test,
because its sample count reports are of counters that run freely: they start
from whatever the console's earlier work left, which differs from run to run
(queries.py resets them in its first test; vizpred had no such test). Each
report's end minus its start was the same in every run, 1024 samples in each
half of Total and ZPass and nothing else: the predicated draw was never
skipped. Here every test resets the counters after its VIZ query and before the
predicated draw's start report (RB_SAMPLE_COUNT_CTL 3, a copy then a reset, as
queries.py's 'zpd after reset'), so the start report is all zero and the end
report is the draw's own counts, the same on every run.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vizpred  # noqa: E402


class ResetCounters:
    """The suite, with every test's counters reset before its start report."""

    def __init__(self, suite):
        self._suite = suite

    def test(self, *args, **kwargs):
        kwargs.setdefault('zpd_reset', True)
        return self._suite.test(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._suite, name)


def build(suite, part):
    vizpred.build(ResetCounters(suite), part)
