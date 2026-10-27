"""4x per-sample depth slope and explicit-depth reference."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import depthslopeprobe


def build(suite):
    depthslopeprobe.build_samples(suite, 4)
