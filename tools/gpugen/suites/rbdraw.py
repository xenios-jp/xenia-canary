"""Draw helpers the render backend suites share over rb's render target:
cases of equal state grouped into one draw of rectangles."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gpugen  # noqa: E402
import pm4  # noqa: E402
import rb  # noqa: E402

f2w = gpugen.f2w


def grouped(states):
    """An order of indices into `states` grouping equal states, in order of
    first appearance."""
    groups = {}
    for i, state in enumerate(states):
        groups.setdefault(state, []).append(i)
    return [i for indices in groups.values() for i in indices]


def runs(states):
    """[state, first, count] of each run of equal consecutive states."""
    out = []
    for i, state in enumerate(states):
        if out and out[-1][0] == state:
            out[-1][2] += 1
        else:
            out.append([state, i, 1])
    return out


class Shapes:
    """Inputs of 2 dwords a vertex, rb.RECT first, then rectangle lists,
    each added once."""

    def __init__(self):
        self.inputs = list(rb.RECT)
        self.draws = {}

    def rectangles(self, rects):
        """A draw of rectangles (x0, y0, x1, y1) over pixels x0 .. x1 - 1,
        y0 .. y1 - 1: their edges half a pixel before those pixel centers."""
        key = tuple(rects)
        if key not in self.draws:
            start = len(self.inputs) // 2
            for x0, y0, x1, y1 in rects:
                self.inputs += [f2w(v - 0.5)
                                for v in (x0, y0, x1, y0, x1, y1)]
            self.draws[key] = (pm4.PRIM_RECTANGLE_LIST, start,
                               3 * len(rects))
        return self.draws[key]

    def pixels(self, first, count, repeat=1):
        """A draw of pixels (index x + 64 y) first .. first + count - 1, a
        rectangle per row, all of them `repeat` times."""
        rows = []
        i = first
        while i < first + count:
            x, y = i % rb.WIDTH, i // rb.WIDTH
            n = min(rb.WIDTH - x, first + count - i)
            rows.append((x, y, x + n, y + 1))
            i += n
        return self.rectangles(rows * repeat)


def test(suite, name, texture, state, resolve, draw=rb.RECT_DRAW, passes=(),
         after_passes=(), shader=None, pixel=None,
         program_cntl=rb.PROGRAM_CNTL, clear=None):
    """rb.test with `clear`, a clearing Resolve before the draws."""
    suite.test(name, shader or gpugen.resolve_vertex_shader(),
               constants=rb.CONSTANTS, pixel_shader=pixel or rb.pixel_shader(),
               program_cntl=program_cntl, registers=state, passes=passes,
               after_passes=after_passes, fetch_constants=rb.fetch(texture),
               draw=draw, resolve=resolve, clear=clear)
