# Xbox trace replay and comparison

`play.py` turns a captured Xenia GPU trace into a small Xbox 360 executable.
The console executes the captured GPU commands and records each resolve's
output. The same trace can then be replayed through Metal or Vulkan and
compared with those console results, including their original pixel bits.

The replay entry point and report tools are in `tools/gpugen`. This workflow
is separate from building a synthetic suite with `gpugen.py build`.

## Commands

`python3 tools/gpugen/run.py --help` lists the entry points; each command has
its own `--help`.

| Command | Purpose |
|---|---|
| `trace TRACE OUT` | Build a console replay executable and memory sidecar. |
| `console OUT` | Execute that build on Xbox and collect reference results. |
| `replay OUT --gpu BACKEND` | Execute recorded stages in a backend trace dump. |
| `report OUT` | Analyze recorded results and generate HTML; no GPU execution. |
| `diff A.json B.json` | Compare decoded results in the terminal. |
| `isolate OUT TRACE` | Build an optional resolved-memory diagnostic trace. |
| `expect OUT EXPECTED --source TEXT` | Store a verified synthetic console oracle. |
| `check EXPECTED OUT` | Check a replay against a stored oracle. |
| `xenia OUT` | Execute the generated title inside the emulator. |

The direct `play.py` and `stage_report.py` entry points remain available.

## Requirements

- Python 3 and a complete checkout, including `tools/xex` and `tools/xamgen`.
  These tools use Python's standard library. Run them from the repository root.
- An RGH/JTAG console that can run the unsigned generated executable, either
  at Aurora with FTP access or through the XBDM/XeCLI route described in
  [the gpugen README](README.md).
- A `.xtr` capture and your own decrypted, decompressed image of a title that
  links Direct3D. `--microcode IMAGE` extracts the command processor microcode
  from that image. It is required for console execution and is not distributed.
- A backend build providing `xenia-gpu-metal-trace-dump` or
  `xenia-gpu-vulkan-trace-dump`. With an existing CMake configuration, enable
  `XENIA_BUILD_MISC` and build the appropriate target. The tool checkout and
  backend build may be separate; `TRACE_DUMP` selects the executable explicitly.

## Capture the console reference

Build one replay title. Keep the result name short because it is embedded in
the executable's console paths.

```sh
python3 tools/gpugen/run.py trace /path/to/capture.xtr out/replay/console \
  --microcode /path/to/direct3d-image.bin --results replay --reboot
```

This writes `gpugen.xex`, `gpugen.mem`, `gpugen.json` and `gpugen.xtr`.
The large memory sidecar is uploaded under a content-derived name; the title
loads it alongside the executable. Do not rename or copy only the XEX.

Configure `GPUGEN_AURORA` in your shell as described in the README, then run:

```sh
python3 tools/gpugen/run.py console out/replay/console
```

The runner uploads the title and memory, launches it, and collects
`console_results.bin`, `console_results.json` and the decoded raw outputs.
`console_edram_before.bin` is the initial console eDRAM snapshot used to seed
backend replays. Check that every segment completed before comparing results.
A stopped or unrun stage is not a passing stage.

Leave the default initial eDRAM capture and zero initialization enabled for
normal comparisons. `--no-edram` and `--no-zero` are diagnostic options.
`--watchdog SECONDS` overrides the automatically chosen console watchdog.

## Replay Metal and Vulkan

Give each run its own directory, preserving identical inputs and the console
seed. For example:

```sh
mkdir -p out/replay/metal out/replay/vulkan
for backend in metal vulkan; do
  cp out/replay/console/gpugen.json out/replay/$backend/
  cp out/replay/console/gpugen.xtr out/replay/$backend/
  cp out/replay/console/console_edram_before.bin out/replay/$backend/
done

TRACE_DUMP=/path/to/xenia-gpu-metal-trace-dump \
  python3 tools/gpugen/run.py replay out/replay/metal --gpu metal

TRACE_DUMP=/path/to/xenia-gpu-vulkan-trace-dump \
  python3 tools/gpugen/run.py replay out/replay/vulkan --gpu vulkan
```

Run each backend on a host that supports it, then collect its directory with
the other results. Only include available backends in the report configuration.
Extra settings can be supplied with repeated `--dump-arg=--name=value` options.
`--timeout` is the timeout for each stage replay.

The output includes `xenia_results.bin`, decoded JSON, and
`xenia_provenance.json` with the executable's SHA-256 and arguments. The
runner also probes all-zero and all-one initial eDRAM to identify dependence
on initial contents. Preserve those diagnostic files.

For a terminal comparison:

```sh
python3 tools/gpugen/run.py diff \
  out/replay/console/console_results.json out/replay/metal/xenia_results.json
```

## Generate the HTML report

Save this as `out/replay/console/stage-report.json`:

```json
{
  "title": "Captured trace",
  "primary": "Metal",
  "modes": {
    "played": {
      "out": "../report",
      "backends": [["Metal", "../metal"], ["Vulkan", "../vulkan"]]
    }
  }
}
```

Paths in this configuration are relative to the console run directory.
For a Vulkan-only comparison, remove the Metal entry and use `"Vulkan"` as
the primary backend.

```sh
python3 tools/gpugen/run.py report out/replay/console --mode played
```

Open `out/replay/report/report.html`. Keep `report.json`, `img/` and `data/`
beside it when moving the report. Generating the report analyzes existing
results; it does not execute the console or a backend. `--render` rewrites
only the HTML from an existing `report.json` after a template change.
Unchanged pixel analyses are cached in `.stage_report_cache/`; only changed
inputs are recomputed. `--no-cache` forces recomputation. Progress names the
command and run being analyzed and distinguishes cache reuse from new work.

Xbox remains the image reference. The main run selector chooses the Metal
or Vulkan image compared with it. The separate progress section measures
changes between two backend runs against that same reference.

## Keep before/after runs

Replay a changed build into a new directory with the same trace, metadata
and console seed. Add it to the mode's `variants`, for example:

```json
"variants": [{
  "label": "Metal after change",
  "of": "Metal",
  "dir": "../metal-after-change",
  "name": "After change",
  "change": "Describe the tested change",
  "run_id": "metal-change-1",
  "parent_run": "Metal"
}],
"selected_runs": {"Metal": "Metal after change"},
"comparison_runs": {"Metal": "Metal"}
```

Add further variants for subsequent iterations and regenerate the report.
Do not overwrite previous results. The generator rejects a variant whose
trace, metadata or initial console eDRAM differs from its base run.

Executable hashes are recorded automatically. A source commit is displayed
only when `xenia_provenance.json` contains a verified `build_source` record.
The checkout at replay time may differ from the executable's source; it is
not used as proof of build identity. Uncommitted builds should retain their
binary hash and the names and hashes of all cumulative patches.

## Interpreting the comparison

- The trace executes draw and resolve commands, but console images are
  captured at resolves. Sidebar command numbers refer to the original trace;
  gaps do not mean that the intervening draw commands were skipped.
- The first differing resolve locates the first measured divergence. Finding
  the exact draw within it can require a smaller probe or extra instrumentation.
- One trace may contain multiple frames. Report frame boundaries come from
  recorded swap events, not GPU interrupts.
- Target formats, sizes and draw state supply descriptive stage names. These
  do not establish semantic names such as a game's lighting or shadow pass.
- Difference counts compare original bits. HDR previews may clamp displayed
  colors; the pixel inspector retains the recorded channel values.
- An optional `stage_isolate.py` diagnostic restores the console's resolved
  memory after each resolve. It does not restore eDRAM, so inherited render
  target differences can remain. Keep ordinary replay as the primary result.

Unsupported trace packets or layouts fail with a diagnostic. A watchdog stop,
missing reference, or unverifiable initial state must be investigated rather
than presented as a successful comparison.
