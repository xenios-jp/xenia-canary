"""Colour write rounding at the conversion boundaries: the rb suite's draw
(oC0 = an exact float4 per pixel) into each colour target format, read back
raw (copy command 0, the EDRAM bits), with textures of values around each
format's rounding boundaries. Pixel i = x + 64 y, d = i // 256 - 4 (-4 to 3)
an offset in units of the float's last bit (toward +inf), k = i % 256:

- 'u8' (8-bit unorm): R (k + 0.5) / 255 + d; G k / 255 + d; B the boundary
  of the 23-bit rule, (ceil((k - 0.5) 2^23 / 255) + d) / 2^23; A B less one
  float bit for even d, plus one for odd d.
- 'u10' (10-bit unorm, 2-bit alpha): R, G and B as u8's with 1023 and
  k' = k * 4 + i % 4; A the 2-bit boundaries (j + 0.5) / 3 and the 23-bit
  rule's for j = k % 3, + d.
- 'fx' (16-bit fixed point, 1024 = 1): q from -32772 to -32741, -64 to 63,
  32740 to 32771 and 64 random; R (q + 0.5) / 1024 + d; G q / 1024 + d;
  B 31 to 40 and -40 to -31 in steps, + d; A SPECIALS then random floats of
  every exponent.
- 'h' (float16): f16 code h = (k * 0x7C00) // 256 + (i // 256); R the
  midpoint between h and h + 1, + (d % 3 - 1); G the same negated; B around
  65504 to 65536 and the denormal midpoints; A SPECIALS then random.
- '7e3' (2_10_10_10_FLOAT): code c = (i * 1023) // 2048; R the midpoint of c
  and c + 1 (+ i % 3 - 1); G c's value + (i % 3 - 1); B 31.5 to 40 in steps,
  negatives and tiny values; A the 2-bit boundaries.

Each test: '<format> <texture>[ exp <bias>] raw'.
"""

import math
import os
import random
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpugen  # noqa: E402
import rb  # noqa: E402

N = rb.WIDTH * rb.HEIGHT


def bits(v):
    return struct.unpack('>I', struct.pack('>f', v))[0]


def value(w):
    return struct.unpack('>f', struct.pack('>I', w))[0]


def step(w, d):
    """w moved d float bits toward +inf (for finite w)."""
    v = value(w)
    for _ in range(abs(d)):
        v = math.nextafter(v, math.inf if d > 0 else -math.inf)
    return bits(v)


def q23_boundary(m, scale):
    """The smallest 23-bit fraction rounding to code m under
    (Q23 * scale + 2^22) >> 23."""
    return max(0, -(-((2 * m - 1) << 22) // scale))


def unorm_texture(scale, alpha_scale=None):
    out = []
    for i in range(N):
        d, k = i // 256 - 4, i % 256
        if scale == 1023:
            k = k * 4 + i % 4
        r = step(bits((k + 0.5) / scale), d)
        g = step(bits(k / scale), d)
        b0 = bits(q23_boundary(k, scale) / 2 ** 23)
        b = step(b0, d)
        if alpha_scale is None:
            a = step(b0, d - 1 if d % 2 == 0 else d + 1)
        else:
            j = k % 3
            a0 = (bits((j + 0.5) / 3) if i % 2 else
                  bits(q23_boundary(j + 1, 3) / 2 ** 23))
            a = step(a0, d)
        out += [r, g, b, a]
    return out


def fixed_texture(rng):
    qs = (list(range(-32772, -32740)) + list(range(-64, 64)) +
          list(range(32740, 32772)) +
          [rng.randrange(-32768, 32768) for _ in range(64)])
    big = [31 + j * 9 / 128 for j in range(128)]
    big = big + [-v for v in big]
    out = []
    for i in range(N):
        d, k = i // 256 - 4, i % 256
        q = qs[k]
        a = (rb.SPECIALS[i] if i < len(rb.SPECIALS) else
             bits(rng.choice((-1, 1)) * rng.uniform(1, 2) *
                  2.0 ** rng.randint(-30, 30)))
        out += [step(bits((q + 0.5) / 1024), d), step(bits(q / 1024), d),
                step(bits(big[k]), d), a]
    return out


def half_value(h):
    e, m = (h >> 10) & 31, h & 1023
    return (m / 2 ** 24) if e == 0 else (1024 + m) / 1024 * 2.0 ** (e - 15)


def half_texture(rng):
    out = []
    for i in range(N):
        k = i % 256
        h = min(0x7BFE, (k * 0x7C00) // 256 + i // 256)
        mid = bits((half_value(h) + half_value(h + 1)) / 2)
        dd = i % 3 - 1
        if i % 256 < 128:
            b = step(bits(65504 + (i % 128) / 4), dd)
        else:
            b = step(bits((i % 128 + 0.5) / 2 ** 24), dd)
        a = (rb.SPECIALS[i] if i < len(rb.SPECIALS) else
             bits(rng.choice((-1, 1)) * rng.uniform(1, 2) *
                  2.0 ** rng.randint(-30, 30)))
        out += [step(mid, dd), step(mid, dd) | 0x80000000, b, a]
    return out


def value_7e3(c):
    e, m = c >> 7, c & 127
    return m / 512 if e == 0 else (128 + m) / 128 * 2.0 ** (e - 3)


def texture_7e3():
    out = []
    for i in range(N):
        c = min(1022, (i * 1023) // 2048)
        dd = i % 3 - 1
        mid = bits((value_7e3(c) + value_7e3(c + 1)) / 2)
        j = i % 64
        b = (bits(31.5 + j / 8) if i % 128 < 64 else
             bits(-(2.0 ** -(j % 32))) if j < 32 else bits(2.0 ** -(j - 22)))
        a = step(bits(q23_boundary(i % 3 + 1, 3) / 2 ** 23) if i % 2 else
                 bits((i % 3 + 0.5) / 3), dd)
        out += [step(mid, dd), step(bits(value_7e3(c)), dd), step(b, dd), a]
    return out


# (format name, ColorRenderTargetFormat, bytes per pixel, texture, bias)
TESTS = [('8_8_8_8', 0, 4, 'u8', 0), ('8_8_8_8', 0, 4, 'u8', 1),
         ('8_8_8_8_GAMMA', 1, 4, 'u10', 0), ('2_10_10_10', 2, 4, 'u10', 0),
         ('2_10_10_10_AS_10_10_10_10', 10, 4, 'u10', 0),
         ('16_16', 4, 4, 'fx', 0), ('16_16_16_16', 5, 8, 'fx', 0),
         ('16_16_16_16', 5, 8, 'fx', 2), ('16_16_16_16', 5, 8, 'fx', 5),
         ('16_16_16_16', 5, 8, 'fx', -1), ('16_16_FLOAT', 6, 4, 'h', 0),
         ('16_16_16_16_FLOAT', 7, 8, 'h', 0),
         ('2_10_10_10_FLOAT', 3, 4, '7e3', 0),
         ('2_10_10_10_FLOAT_AS_16_16_16_16', 12, 4, '7e3', 0),
         ('2_10_10_10_FLOAT_AS_16_16_16_16', 12, 4, 'fx', 0)]


def textures():
    rng = random.Random(0xC3D)
    return {'u8': unorm_texture(255), 'u10': unorm_texture(1023, 3),
            'fx': fixed_texture(rng), 'h': half_texture(rng),
            '7e3': texture_7e3()}


def build(suite):
    suite.set_inputs(rb.RECT, 2)
    for name, words in textures().items():
        suite.add_texture(name, words)
    for name, fmt, bpp, texture, bias in TESTS:
        state = rb.registers(fmt)
        if bias:
            state = [(r, [fmt << 16 | (bias & 0x3F) << 20])
                     if r == 'RB_COLOR_INFO' else (r, v) for r, v in state]
        rb.test(suite, f'{name} {texture}' + (f' exp {bias}' if bias else '') +
                ' raw', texture, state,
                gpugen.resolve_registers(rb.WIDTH, rb.HEIGHT,
                                         6 if bpp == 4 else 26, bpp,
                                         command=0))
