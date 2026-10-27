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
- `stencil`, `depthstate`, `depthstencil`, `depthbias`, `depthbias2x`,
  `depthslope`, `depthgeometry`, `msaadepth`: D24S8 and D24FS8 conversion
  and compares, stencil, bias and slopes (`depthcases`).
- `merger1`-`merger3`, `gammablend`, `gammamsaa`, `color7e3`, `msaablend`,
  `samplecov`, `unormresolve`: render targets and exports, gamma, 7e3, MSAA
  blending, sample coverage and resolves (`merger`).
- `targets1`-`targets4`: the render targets and resolves games use
  (`targets`). `resolverect`: Resolve rectangles, destinations, window
  offsets, scissors, endianness and clears.
- `hiz1`-`hiz7`: the hierarchical depth and stencil tests, for every
  RB_HIZCONTROL value titles draw with (`hiz`).
- `hiz9a`-`hiz9e`: the blocks a hierarchical depth draw writes, clears and
  compares, block addresses and Metro 2033's VIZ predicated shadow cascades,
  each test writing every block first (`hiz9`). `hiz8a`-`hiz8d` read what
  the chain left, because their depth copying clears never write the buffer.
- `reinterp1`-`reinterp3`: EDRAM written as one format or sample count
  and read through a view of another, for every pair titles reinterpret,
  and where each sample of each pixel sits (`reinterp`).
- `tuples1`-`tuples3`: the per-draw output merger states titles combine
  (sample count, targets' formats, blending and masks, depth, stencil, alpha
  test and to mask, polygon offset), each as the commonest draw with it in
  the game traces sets its registers (`tuples`).
- `extreme`, `sweeprcp`, `sweeplogh`: products and sums at the edges of the
  float range, rcp over [1, 2) and log over [0.5, 1), every mantissa.
- `cwround`, `blendk`, `alpharef`, `gmexp`, `sprite10`: colour write
  rounding at each format's boundaries, blend constants per format, more
  alpha test references, 8_8_8_8_GAMMA at 2x and 4x through every resolve
  and Metro: Last Light's sprite blend.
- `zplane1`-`zplane3`, `zvertex1`, `zvertex2`, `zoffset1`, `zoffset2`,
  `zresolve1`: depth interpolated over triangles, from a vertex's Z, W and
  viewport, with constant polygon offsets, and depth resolved to other
  formats (`zplane`).
- `destbase`, `destpitch`, `surfpitch`: Resolve destination bases off 4
  KiB, destination pitches off 32 texels and surface pitches off 80-pixel
  tiles, which titles never program (`cpmem`).
- `edramraw`: the converting Resolve that reads every EDRAM word back
  unchanged, for a dump of the whole EDRAM.
- `aamask`: which samples of which pixels each PA_SC_AA_MASK bit enables,
  at 1x, 2x and 4x, read back through `edramraw`'s view.

What the console does, beyond what the expected results show:

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
- Indices (`indices`): an index is clamped to VGT_MIN_VTX_INDX and
  VGT_MAX_VTX_INDX after VGT_INDX_OFFSET is added; the offset wraps at 24
  bits and 32-bit indices keep bits 0-23. Primitive reset compares the raw
  index, before the offset, in point and triangle lists too.
- Control flow (`flow`): a vertex vector is 64 vertices. Control flow sees a
  setp only from an exec marked predicate-dirty (CF bit 41 clear), ALU
  predication either way. A predicated exec or call runs for the whole vector
  if any vertex passes; a predicated jmp or loop break is taken only if every
  vertex does.
- Pixel shader control flow (`psflow`, `psflow1`, `psflow2`): a pixel vector is 64
  pixels, 16 2x2 quads: a rectangle's are its window-aligned 8x8 tiles, a
  triangle's 16 of its quads (partly covered ones too) in rows of 4, packed
  differently each run, a point list's 16 consecutive points. A predicated
  cond_exec or call runs the whole vector if any pixel passes, killed ones
  included; a predicated jmp or loop break is taken only if all do, so a
  loop runs the vector's largest count. Predicated ALU, fetches, setps and
  kills act per pixel. Control flow takes the predicate at its first
  predicated instruction after a dirty exec, as it is then, and keeps it
  until the next dirty exec; a cond_exec on a bool constant (ops 3 and 4)
  is dirty, ops 13 and 14 aren't, and the predicate starts clear. A dirty
  exec counts only where it runs: one skipped on its bool constant or on the
  predicate, or jumped over, leaves control flow's copy as it was. A killed
  pixel keeps running and voting but exports nothing after the kill. Loop
  N reads loop constant N, aL and a0 relative reads are per pixel, and an
  export's dest_rel bits don't add aL.
- Resolves (`resolverect`): the destination texel is the rectangle's pixel
  after PA_SC_WINDOW_OFFSET, even with vtx_window_offset_enable clear.
  For a 32bpp tiled destination at RB_COPY_DEST_BASE B, the texel at tiled
  offset p goes to the swizzle of ((p + ((B >> 3) & ~0x1F)) & ~0x3F) | (p &
  0x3F): bits 0-7 of B do nothing and bits 8-11 move texels within their
  tile, even below B (`destbase`). A destination row has floor(pitch / 32)
  macro tiles, the later row winning a tile two land on (`destpitch`). A
  surface row has floor(pitch / 80) EDRAM tiles (2 pitch / 80 at 4x),
  pixels past them wrapping into the next row and the later one winning;
  the pitch clamps neither the draw nor the Resolve (`surfpitch`). Titles
  program none of these.
- Shader constants and loads (`cpload`, `cpload2`, `cpalign`):
  LOAD_ALU_CONSTANT copies its dwords into the constants from its index (4
  per constant, the pixel shader's from 1024: 2032 and 2016 load its
  c252-c255 and c248-c255), as type 0 writes do, and the later of the two
  wins. IM_LOAD replaces the bound vertex or pixel shader, from the ring or
  an indirect buffer at depth 1 or 2, also after INVALIDATE_STATE 0x7FFF. An
  indirect buffer runs its packets (constants, loads, draws) as the ring
  would, at depth 2 too. Data in a NOP is skipped. All of these run the same
  from 16, 32 or 64 bytes past a 128-byte line (LOAD_ALU_CONSTANT), 32 or 64
  past (IM_LOAD, indirect buffers, nested ones), and from an indirect buffer
  ending where its packets do; so does CONTEXT_UPDATE 0.
  SET_SHADER_CONSTANTS 0x43F0 writes c252-c255 as Xenia does. SET_CONSTANT2
  writes a context register by its index (0x2102, VGT_INDX_OFFSET), as
  Xenia does; 0x43F0 writes nothing a shader reads, predicated or not, and
  0x4102 stalls the console. MEM_WRITE writes its first dword, swapped by
  the address's endian bits (0-3 as Xenia), and the command processor then
  reads the packet's remaining dwords as packets: a type 0 write there sets
  its registers, a type 2 is filler, and a dword that reads as some other
  header loses the stream (`cpload`'s 0x55667788, a type 1 header, stalled
  it).
- Bin predication (`cpbins`): a predicated packet (bit 0 of the type 3
  header) runs when bin select & bin mask, as 64-bit values, is not 0, for
  DRAW_INDX, DRAW_INDX_2, INDIRECT_BUFFER (its contents skipped as a whole),
  SET_CONSTANT (ALU and registers) and SET_CONSTANT2. SET_BIN_MASK and
  SET_BIN_SELECT take the low dword first, then the high. An indirect
  buffer's SET_BIN_MASK_LO holds after it returns.
- Screen extents (`cpext`, `cpext2`): EVENT_WRITE_EXT SCREEN_EXT_RPT writes
  12 bytes: min x, max x, min y, max y, min z, max z, little-endian 16-bit
  each, swapped by the address's endian bits (8in16 makes them big-endian
  halfwords). x and y are the union, since the last SCREEN_EXT_INIT, of the
  boxes of the primitives drawn, >> 3 (8-pixel units). A primitive's box
  comes from its vertices, in pixel coordinates with centers on integers: x
  from ceil(min x) to ceil(max x) - 1, likewise y, whether it covers a pixel
  or not (a rectangle at x 7.6-7.9 gives min 1, max 0: pixels 8 to 7);
  points by PA_SU_POINT_SIZE, lines likewise. It is after the window offset,
  not cut by the window scissor, the low end clamped at 0. Culled
  primitives don't count; RB_COLOR_MASK 0 doesn't matter. SCREEN_EXT_INIT
  sets 03FF 0000 03FF 0000 07FF 0000, which a report with no draw since
  writes; a report doesn't reset them, so a second report gives the union.
  z stays 07FF 0000 with the depth test off or on (ALWAYS, with or without
  depth writes, no HiZ). A predicated report that's skipped writes nothing.
- Culling, user clip planes, VTE 0x400 (`cullclip`): coverage as Xenia for
  cull none, front and back with face ccw and cw, in screen space and through
  Direct3D's viewport; for PA_CL_CLIP_CNTL 0x80001 and 0x80003 (kept where
  the plane's dot product with the clip-space position is >= 0), 0x90000 and
  0x90003 (planes ignored when clipping is disabled); and for
  PA_CL_VTE_CNTL 0x400 with W 1, 2 and 0.5. The word differences are the
  pixel probe's known ones (`primitives`): -0 where an attribute is 0,
  PsParamGen.zw, and attributes of clipped polygons a few units off
  (9.99696 for 10).
- Vertex snapping (`snap`, `snap2`): PA_SU_VTX_CNTL quant_mode 5, 6 and 7
  rasterize nothing, at every round_mode. Modes 0 to 4 move the edges of
  `snap2`'s wide rows as Xenia doesn't, which snaps no vertices.
- Interpolation (`pixel`, `centroid`): a set SQ_INTERPOLATOR_CNTL.param_shade
  bit interpolates, a clear one is flat. sc_sample_cntl 0 evaluates smooth
  interpolators at the centroid, 1 at the center, 2 by sampling_pattern
  (bit set center, clear centroid). The centroid is a covered sample, the
  center only at 2x with both covered. PsParamGen is always the center. The
  sample offsets are (1/4, -1/4) and (-1/4, 1/4) at 2x, and (3/8, -3/8),
  (-1/8, -1/8), (1/8, 1/8) and (-3/8, 3/8) at 4x, y down.
- Colour writes (`cwround`, `cwulp`): after the exponent bias (a float32
  scale by 2^bias), unorm channels (8_8_8_8, 2_10_10_10, AS_10_10_10_10's
  colour and GAMMA's 10-bit linear value) clamp to [0, 1], truncate the
  float to 23 fraction bits and store (Q23 (2^n - 1) + 2^22) >> 23; NaN goes
  by its sign, + to 1 and - to 0. GAMMA stores the truncating piecewise
  linear encode of that, and AS_10_10_10_10's 2-bit alpha is the 10-bit
  alpha >> 8. 16_16 and 16_16_16_16 store v 1024 rounded half up and
  clamped, NaN as infinity of its sign. The 16-bit floats round half away
  from zero with no specials (exponent 31 is normal, NaN 0x7FFF or 0xFFFF
  by sign). 7e3 rounds half up, and its AS_16_16_16_16 view truncates the
  16_16_16_16 value into 7e3, alpha the fixed value >> 8.
- Blend constants (`blendk`): 8_8_8_8 and 2_10_10_10 take q = the constant
  written as a 10-bit unorm, and the factor q + (q >> 9) out of 1024 (INV
  1024 minus that); 7e3 the same, shifted left 6 out of 65536.
  16_16_16_16 takes the constant's fixed point value, unclamped, and
  16_16_16_16_FLOAT reads 0.
- Blending (`sprite10`): the 10-bit blender expands operands to 10 bits,
  takes factors q + (q >> 9), and floors the sum of products >> 12 before
  clamping; a host that rounds lands one high in most pixels.
- Gamma MSAA resolves (`gmexp`): the colour is the encode of the truncated
  mean of the samples' 10-bit linear decodes, decoded again, over 1023; the
  alpha the truncated mean of the 10-bit expansions, cut to 8 bits ((a10 >>
  2) / 255) even into 16-bit formats. Then the exponent bias, then the
  destination's conversion, which into 8 bits truncates the float to 16-19
  fraction bits before rounding half up.
- Alpha test (`alpharef`, `arulp`): oC0.a against RB_ALPHA_REF as exact
  float32, denormals kept, +0 equal to -0, and NaN on either side passing
  only NOTEQUAL and ALWAYS. When both have the sign bit set (-0 too) and
  differ, the alpha counts as the smaller. Alpha to mask's offset index is
  (x & 1) + 2 (y & 1).
- Hierarchical depth (`hiz1`-`hiz5`): RB_HIZCONTROL bit 0 writes the
  buffer, bit 1 tests it, bit 4 makes the test greater-than; bits above 17
  move it. The test ignores the depth compare, so it rejects ALWAYS draws,
  and a pixel it rejects skips its stencil ops; a buffer left stale by
  draws with 0 rejects pixels the depth test would pass. Its blocks are 4x4
  samples, written only where a draw covers a whole block, and indexed by
  RB_SURFACE_INFO's hiz_pitch: 0 rejects nothing, less than the width
  aliases blocks. Which blocks a draw leaves written depends on its shape
  and order: at 1x a draw within rows 0-15 leaves none, the same draw in
  rows 16-31 all, and x < 20 over every row only rows 0-15.
- Hierarchical depth clears and compares (`hiz9b`, `hiz9e`): a clearing
  Resolve with RB_HIZCONTROL bit 0 sets every block to RB_HIZ_CLEAR's bits
  0-10 when it copies a colour target (copy source 0-3, raw or converting);
  one copying depth (source 4) leaves the buffer as it was. A draw with bit
  0 sets each block it touches to the top 11 bits of the farthest depth the
  block holds after it. The less-than test rejects a 2x2 quad whose nearest
  depth is at least (S + 1) / 2048, S the block's 11 bits, so 0x7FF never
  rejects; greater-than rejects one whose farthest depth is below the 20e4
  value of S - 1. The buffer keeps its blocks across Resolves, tests and
  titles; only a reboot sets them far.
- Hierarchical depth addresses (`hiz9c`): blocks are 4x4 samples in 16x16
  sample tiles, and tile (tx, ty) is stored at (ty floor(P / 16) + tx) xor
  (ty & 1), P hiz_pitch, doubled at 4x. A hiz_pitch below 16 folds every
  tile row onto the first; one below the width wraps rows into the next.
- Hierarchical depth writes (`hiz9a`): 0x3 draws starting in tile row 0 at
  x 0 leave the top 8 rows of tiles 0-3 as they were, unless a test-only
  draw over two or more tiles, or another 0x3 draw over the tile row below,
  follows. 0x1 draws write every block they touch.
- Hierarchical stencil (`hiz6`, `hiz7`): bit 2 writes it, bit 3 tests it.
  A draw with bit 2 records per 4x4-sample block whether the stencil it
  leaves (after write masks and failed tests, or untouched) equals bits
  8-15; with bit 5 clear the test then culls the blocks that did, set the
  ones that didn't. Bits 6 and 7 change nothing, nor does the testing
  draw's own compare or reference. A clearing Resolve with bit 2 makes
  every block pass; one without leaves them. Titles pair them, as Halo 3's
  0x4035 marking pass then 0x4038 and 0x403d at stencil EQUAL 0x40.
- EDRAM reinterpretation (`reinterp1`-`reinterp3`): a raw Resolve copies
  the bits whatever the view's format; every 32bpp color view, and D24S8
  and D24FS8, copy the same words. A tile is 80 x 16 32bpp sample slots, in
  which pixel (x, y) sample s sits at (x, y) at 1x, at (4 (x >> 2) + (x & 1)
  + 2 s, 4 (y >> 1) + (y & 1) + 2 ((x >> 1) & 1)) at 2x and at (4 (x >> 1)
  + (x & 1) + 2 (s & 1), 4 (y >> 1) + (y & 1) + 2 (s >> 1)) at 4x; tiles go
  row-major, pitch / 80 of them a row, twice that at 4x. A 64bpp pixel takes
  its slot in a pair of tiles: dword 1 in the first, dword 0 16 slots on
  (mod 80) in the second. A depth view swaps the halves of each tile row,
  slot x for x +- 40. Its depth test takes the word as depth << 8 |
  stencil, unorm24 for D24S8 and 20e4 for D24FS8, and compares them as
  integers (0.5 is 0x800000 and 0xE00000). A converting read decodes
  whatever bits are there as the view's format. PA_SC_AA_MASK is a sample
  mask for each pixel of a 2x2 quad: bit 4 ((x & 1) + 2 (y & 1)) + s
  enables sample s, so at 1x only bits 0, 4, 8 and 12 count (`aamask`).
- Per-draw state (`tuples1`-`tuples3`): a draw repeated with the same
  vertices passes a depth EQUAL test on every sample it repeats and nowhere
  else, as titles' depth prepasses rely on (D24FS8, 2x).
