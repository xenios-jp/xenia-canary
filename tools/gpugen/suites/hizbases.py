"""Hierarchical depth tests at additional observed buffer bases.

Exercises hidden/shown geometry, edge blocks, ALWAYS, stale near/far depth
and disabled depth writes, with a freshly initialized hierarchical buffer.
The controls cover one LESS and two GREATER configurations seen in traces.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hiz  # noqa: E402


def build(suite):
    hiz.build(suite, 3, values=(0x167C0003, 0x18B00012, 0x13B00012))
