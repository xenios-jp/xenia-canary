# gpugen

Records what the Xenos GPU computes on a console and checks Xenia against it,
the way `tools/xamgen` does for xam exports.

A suite in `suites/` lists tests. `gpugen.py build` turns it into one XEX,
built with `tools/xex`, with a fixed executor that brings up the command
processor through the kernel the way Direct3D does and submits the suite's
PM4 stream one test at a time. A test is a draw whose shaders, written with
the microcode encoder in `xenos.py`, store their results with memory export,
or a draw into EDRAM followed by a Resolve. `run.py` runs that XEX on an RGH
console over XBDM (XeCLI `rgh.exe`), on any console running Aurora through
its FTP, or in the Release build of Xenia, and compares runs. Each build also
writes `gpugen.xtr`, the same stream as a trace, which `run.py replay` plays
through a backend's trace dump with no window or emulated CPU.

```
python3 tools/gpugen/gpugen.py build tools/gpugen/suites/smoke.py OUT \
    --microcode IMAGE
python3 tools/gpugen/run.py console OUT
python3 tools/gpugen/run.py xenia OUT
python3 tools/gpugen/run.py xenia OUT \
    --xenia-arg=--render_target_path=accuracy
python3 tools/gpugen/run.py replay OUT
python3 tools/gpugen/run.py diff OUT/console_results.json \
    OUT/xenia_results.json
python3 tools/gpugen/run.py expect OUT \
    src/xenia/gpu/testing/gpu__gen_smoke.json --source "..."
python3 tools/gpugen/run.py check \
    src/xenia/gpu/testing/gpu__gen_smoke.json OUT
```

The expected results checked in under
`src/xenia/gpu/testing/gpu__gen_<suite>.json` come from a console: the run's
source and the SHA-1 of the `gpugen.xtr` it ran, with its results file beside
it, xz-compressed, as `gpu__gen_<suite>.bin.xz`. Each is a run that three
or more console runs of the same build reproduced exactly. `check` compares
a Xenia run or a replay with them and fails on any word that differs or test
not run; it refuses a build whose stream differs, so a changed suite needs
recording again. Checking a backend is a build, `replay` and `check` per
suite, with no console. `sweep` has none, as its tests depend on the
environment.

A console run needs the command processor microcode Direct3D passes to
`VdInitializeEngines`: the kernel's own doesn't run these streams. It isn't
distributed; `--microcode` takes the decrypted, decompressed image of an XEX
that links Direct3D and finds it there by SHA-1. Without it the build only
runs in Xenia. Like xamgen's, `run.py xenia` runs the Release build or the one
`XENIA_EXE` names, `run.py console` needs XeCLI's `rgh` on `PATH` or named by
`RGH`, and a console run starts from Aurora.

With `GPUGEN_AURORA=ftp://USER:PASSWORD@HOST/FOLDER` in the environment,
`run.py console` needs no XBDM: it puts the XEX in `FOLDER` over Aurora's
FTP, has Aurora launch it and fetches the results once Aurora is back. Given
several OUTDIRs it runs them as one chain from a single launch, each built
with `--results NAME` and the next one's name as `--next`, the last usually
with `--reboot`:

```
python3 tools/gpugen/gpugen.py build tools/gpugen/suites/smoke.py OUT/smoke \
    --microcode IMAGE --results smoke --next alpha
python3 tools/gpugen/gpugen.py build tools/gpugen/suites/alpha.py OUT/alpha \
    --microcode IMAGE --results alpha --reboot
python3 tools/gpugen/run.py console OUT/smoke OUT/alpha
```

A run that stopped short rebooted the console, so a chain ends there.

Each test is one segment ending in a fence, so a hang names its test. A GPU
left busy keeps the title from exiting, so a title whose fences stop landing
writes its results and reboots the console (HalReturnToFirmware), as does a
watchdog thread if the run overruns; `--reboot` reboots at the end of any
run. Things the console needs that Xenia doesn't:

- Float constants go out in whole groups of four aligned registers, never
  across the vertex and pixel banks; fetch constants as whole 6-dword slots.
- A fetch reading the result of an earlier fetch in the same clause needs the
  serialize bit, and a mini vertex fetch must stay within the 8 dwords the
  preceding full fetch prefetches (`fetch_inputs`).
- Vertex shader memory export runs in multipass export mode and ends like
  Direct3D's memexport shaders (`memexport_shader`).
- PsParamGen.xy is the pixel position only with
  SQ_CONTEXT_MISC.sc_output_screen_xy, and multisampling takes
  PA_SC_AA_CONFIG as well as RB_SURFACE_INFO.msaa_samples.
- An EVENT_WRITE_ZPD report whose RB_SAMPLE_COUNT_CTL doesn't copy (bit 1)
  makes every report after it lag a draw, and the dashboard couldn't launch
  titles after a run with them: gpugen refuses them.

Suites:

- `cp`, `smoke`: bring-up checks.
- `alu`, `specials`, `ops2`, `dot`, `dotprobe`, `transcendental`: shader ALU
  results, specials and rounding. `a0`: the address register. `sweep`: every
  mantissa of one op over a binade, or sin/cos over a linear range, chosen
  with `GPUGEN_SWEEP` (`GPUGEN_VERTICES` shortens it); its results are raw
  `.bin` files.
- `vfetch`, `edges`, `mexport`: vertex fetch conversions and indexing, memory
  export formats and addressing.
- `tfetch`, `tfetch_formats`, `tfilter`, `tfilter_probes`: texture fetch
  conversions, block formats, endianness, filtering and addressing.
- `getbcf`: getBCF in each clamp mode, filter, format and dimension, next to
  a tfetch of the border at the same coordinates.
- `pixel`, `pixel_probes`, `snap`, `msaa`: pixel shader inputs (interpolation,
  PsParamGen), vertex snapping and multisample coverage.
- `rb`, `blend`, `depth`, `zdepth`, `alpha`, `msaarb`: render target formats,
  blending, depth formats and polygon offset, alpha test and alpha to mask,
  MSAA samples and resolves.
- `blend_products`: blending on the fixed point formats one product at a time,
  each operand swept or held.
- `extreme`, `sweeprcp`, `sweeplogh`: products and sums at the edges of the
  float range, rcp over [1, 2) and log over [0.5, 1), every mantissa.
- `edramraw`: the converting Resolve that reads every EDRAM word back
  unchanged, for a dump of the whole EDRAM.
- Arithmetic (`smoke`, `alu`, `specials`, `ops2`): products and sums,
  vector and scalar, round toward zero, an overflow to the largest finite
  value and anything below 2^-126 to zero. mad rounds its product, then the
  sum, the product keeping its exponent unbounded.
- Dot products (`dot`, `dotprobe`): dp4, dp3 and dp2add round each product
  toward zero, exponent unbounded, then add the terms from the last (dp2add
  its addend, then y, then x) into a two's complement fixed point
  accumulator. Before each addition its unit becomes 2^(e - 24), e the
  largest exponent of the terms so far and of the accumulator as two's
  complement (-2^n counts as 2^(n-1)), and the accumulator and the term are
  floored to it. The sum rounds toward zero, as above.
- At the edges of the range (`extreme`): a denormal operand reads as a
  zero of its sign. mul (and muls, muls_prev) with a zero operand is +0
  whatever the other, NaN and Inf too; otherwise the first NaN operand
  comes out unchanged. add passes the first NaN, gives FFC00000 for
  opposite Infs and +0 for an exact zero sum unless both addends are -0.
  mad's product keeps an unbounded exponent into the sum, and an Inf
  product against an opposite Inf addend is FFC00000. In dp4, dp3 and
  dp2add any NaN term, or Infs of both signs, give FFC00000; the
  accumulator's exponent never goes below -127, and a result under 2^-126
  is a zero with the accumulator's sign, so a positive sum of tiny terms
  can be -0.
- max4 (`extreme`, `max4probe`): r = w, then r = v >= r ? v : r for z, y
  and x in turn, so a NaN in w comes out, NaNs elsewhere are skipped and a
  tie keeps the lower lane's bits. cube's doubled major axis is a mul
  (overflow to the largest finite value, zero to +0). muls_prev2 is
  -FLT_MAX unless src0.b > 0 and ps > -FLT_MAX.
- Transcendentals (`sweeprcp`, `sweeplogh`, `sweep`, `transcendental`):
  rcp, rsq, sqrt, log and exp are tables repeating per exponent, with no
  closed form found. rcp is within -2..+3 units of floor(2^24 / m) (equal for
  71%). rsq and sqrt take one table for even exponents and one for odd. log's
  fraction is a 25 bit fixed point value, and a negative sum (x < 1) is held
  as its one's complement before rounding toward zero. exp truncates its
  input to 23 fraction bits, a negative fraction taken as the one's
  complement. Input 16g + l of a table, l = 1..15, is floor(Z_g + a_s l),
  with one offset Z_g per 16 inputs and one slope a_s per 2^18:
  `transcendental_tables.py` packs the sweeps into Xenia's
  xenos_transcendentals table this way. Denormal inputs
  read as zero, results under 2^-126 are +0, and NaN comes back as is.
