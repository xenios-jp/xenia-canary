"""Builds Xenia's src/xenia/gpu/transcendentals/xenos_transcendentals.bin.zst, the
console's scalar transcendental mantissas, from the sweep runs (every mantissa
of one binade, gpugen sweep suite, its console runs identical).

Seven functions, each 2^23 integer results V(k) of the mantissa k:
  rcp    2^24 / m          units 2^-24 (V(0) = 2^24)
  rsq    2^24 / sqrt(m)    units 2^-24
  rsq2   2^24 / sqrt(2m)   units 2^-24
  sqrt   2^23 * sqrt(m)    units 2^-23
  sqrt2  2^23 * sqrt(2m)   units 2^-23
  log    2^25 * log2(m)    25-bit fixed point L(m)
  exp    2^22 * 2^(1+f)    units 2^-22, f = k / 2^23
with m = 1 + k / 2^23.

Each 2^18-mantissa segment s has an integer base B_s and slope a_s with F
fraction bits; each 16-mantissa group g, dg = g mod 2^14, a residual r_g and
d0_g, and, in int32,
  V(16g + l) = B_s + ((r_g + a_s * (16 dg + l)) >> F)       for l = 1..15
  V(16g)     = B_s + ((r_g + a_s * 16 dg) >> F) + d0_g.
Every word is decoded back and checked before writing.

File (little-endian, zstd): "XTR1", then per function u32 F and 32 x (i32 B_s,
i32 a_s, i32 r of the segment's first group); then per function 2^19 i16
r_g - r_(g-1) (0 at a segment's first group); then per function 2^19 u8
d0_g + 3.

Usage: transcendental_tables.py RUNS OUT.bin.zst, RUNS holding the console
runs of SWEEPS (needs numpy and the zstd CLI).
"""

import os
import struct
import subprocess
import sys

import numpy as np

SWEEPS = {
    'rcp': 'chain-sweep-rcp-2/sweep', 'rsq': 'chain-sweep-rsq-1/sweep',
    'rsq2': 'chain-sweep-rsq2-1/sweep', 'sqrt': 'chain-sweep-sqrt-1/sweep',
    'sqrt2': 'chain-sweep-sqrt2-1/sweep', 'log': 'chain-sweep-log-1/sweep',
    'logh': 'q1001/sweeplogh', 'exp': 'chain-sweep-exp-1/sweep',
}
# (name, units as a power of two, fraction bits of the slope).
FUNCTIONS = [('rcp', 24, 10), ('rsq', 24, 10), ('rsq2', 24, 10),
             ('sqrt', 23, 10), ('sqrt2', 23, 10), ('log', 25, 9),
             ('exp', 22, 10)]


def sweep(name):
    with open(os.path.join(sys.argv[1], SWEEPS[name], 'console_results.bin'),
              'rb') as f:
        data = f.read()
    words = np.frombuffer(data, dtype='>u4', count=1 << 23, offset=0x20)
    return words.astype(np.uint32).view(np.float32).astype(np.float64)


def values(name, units):
    if name != 'log':
        return np.rint(sweep(name) * 2.0**units).astype(np.int64)
    # log [1, 2) where L < 2^24, its 2^-25 bit lost above; there sweeplogh,
    # log2(m) - 1 for m in [1, 2), keeps it.
    low = np.rint(sweep('log') * 2.0**25).astype(np.int64)
    high = np.rint(sweep('logh') * 2.0**25).astype(np.int64) + (1 << 25) - 1
    return np.where(low < (1 << 24), low, high)


def encode(y, f):
    """(segments, residuals, d0) of one function's 2^23 values."""
    n = 1 << 18
    l = np.arange(16, dtype=np.int64)
    segments, residuals, d0s = [], [], []
    for s in range(32):
        g = y[s * n:(s + 1) * n].reshape(-1, 16)
        mid = np.median((g[:, 15] - g[:, 1]) / 14.0)
        a = None
        for da in range(4096):
            for sign in (1, -1):
                t = int(round(mid * 2**f)) + sign * da
                lo = (g[:, 1:] * 2**f - t * l[1:]).max(1)
                hi = ((g[:, 1:] + 1) * 2**f - t * l[1:]).min(1)
                if (lo < hi).all():
                    a = t
                    break
            if a is not None:
                break
        if a is None:
            sys.exit(f'segment {s}: no slope with {f} fraction bits')
        lo = (g[:, 1:] * 2**f - a * l[1:]).max(1)
        hi = ((g[:, 1:] + 1) * 2**f - a * l[1:]).min(1)
        base = int(g[0, 0])
        dg = np.arange(g.shape[0], dtype=np.int64)
        prediction = base * 2**f + a * 16 * dg
        r = np.clip(prediction, lo, hi - 1) - base * 2**f - a * 16 * dg
        decoded = base + ((r[:, None] + a * (16 * dg[:, None] + l)) >> f)
        d0 = g[:, 0] - decoded[:, 0]
        decoded[:, 0] += d0
        assert (decoded == g).all()
        assert np.abs(r[:, None] + a * (16 * dg[:, None] + l)).max() < 2**31
        segments.append((base, a, int(r[0])))
        residuals.append(r)
        d0s.append(d0)
    return segments, np.concatenate(residuals), np.concatenate(d0s)


def main():
    out = sys.argv[2]
    header = [b'XTR1']
    deltas, d0_planes = [], []
    for name, units, f in FUNCTIONS:
        segments, r, d0 = encode(values(name, units), f)
        header.append(struct.pack('<I', f))
        for base, a, r0 in segments:
            header.append(struct.pack('<iii', base, a, r0))
        delta = np.diff(r, prepend=0).reshape(32, -1)
        delta[:, 0] = 0
        assert np.abs(delta).max() < 2**15
        assert d0.min() >= -3 and d0.max() <= 4
        deltas.append(delta.reshape(-1).astype('<i2').tobytes())
        d0_planes.append((d0 + 3).astype(np.uint8).tobytes())
        print(f'{name}: F {f}, d0 {d0.min()}..{d0.max()}, delta '
              f'{np.abs(delta).max()}', flush=True)
    raw = b''.join(header + deltas + d0_planes)
    packed = subprocess.run(['zstd', '-19', '-q', '-c'], input=raw,
                            capture_output=True, check=True).stdout
    with open(out, 'wb') as f:
        f.write(packed)
    print(f'{len(raw)} bytes -> {len(packed)} in {out}')


if __name__ == '__main__':
    main()
