#!/usr/bin/env python3
"""Paired A/B timing of GPU trace replays with verified identical output.

Each arm is a trace dump binary plus an optional JSON file mapping cvar names to
values. Arms run as alternating processes (AB, BA, AB, ...); every process
warms up, replays the trace --samples times, and verifies its output. The
comparison is refused if the verified output or the guest work differs.

With --corpus and --reference, it instead replays each window of a corpus
file (`<name> <frame> <last_command> [flags...]`, <name> being an XTR file
stem in --trace-dir) once with --binary and once with --reference, and
compares the SHA-256 of their verification files (window image, EDRAM and
written guest memory).
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import re
import shutil
import statistics
import subprocess
import tempfile

METRICS = {
    "gpu_buffer_union": "gpu_buffer_interval_union_ns",
    "gpu_buffer_sum": "gpu_buffer_duration_sum_ns",
    "command_thread_cpu": "command_thread_cpu_ns",
    "process_cpu": "process_cpu_ns",
    "replay_wall": "replay_wall_ns",
    "gpu_drain_wall": "gpu_drain_wall_ns",
}
GUEST_COUNTS = ("draw_requests", "dxil_draws", "memexport_draws",
                "resolve_requests")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_flags(config):
    flags = []
    for name, setting in sorted(config.items()):
        if (not re.fullmatch(r"[a-z][a-z0-9_]*", name) or
                not isinstance(setting, (str, bool, int, float))):
            raise ValueError("Configuration must map cvar names to scalars")
        if isinstance(setting, bool):
            setting = str(setting).lower()
        flags.append(f"--{name}={setting}")
    return flags


def paired_result(a, b, trials=5000):
    if (len(a) != len(b) or len(a) < 2 or
            any(not math.isfinite(x) or x <= 0 for x in a + b)):
        raise ValueError("Paired metrics need equal-length positive samples")
    ratios = [math.log(y / x) for x, y in zip(a, b)]
    rng = random.Random(0x58454E49)
    reductions = sorted(100 * (1 - math.exp(statistics.mean(
        rng.choices(ratios, k=len(ratios))))) for _ in range(trials))
    return {
        "a_mean_ms": statistics.mean(a) / 1e6,
        "b_mean_ms": statistics.mean(b) / 1e6,
        "paired_reduction_percent":
            100 * (1 - math.exp(statistics.mean(ratios))),
        "ci95_percent": [reductions[int(trials * .025)],
                         reductions[int(trials * .975)]],
        "pairs": len(a),
    }


def parse_corpus(text):
    entries = []
    for line in text.splitlines():
        fields = line.split("#", 1)[0].split()
        if not fields:
            continue
        if len(fields) < 3:
            raise ValueError(f"Corpus line needs name, frame and command: {line}")
        entries.append({"name": fields[0], "frame": int(fields[1]),
                        "last_command": int(fields[2]), "args": fields[3:]})
    return entries


def run_process(binary, trace, directory, flags, args):
    directory.mkdir(parents=True)
    profile = directory / "profile.json"
    command = [str(binary), str(trace), *flags, *args.arg,
               "--async_shader_compilation=false",
               "--async_shader_skip_draws=false", "--framerate_limit=0",
               "--log_level=0", f"--trace_dump_path={directory}",
               f"--trace_profile_path={profile}",
               f"--trace_profile_warmup={args.warmup}",
               f"--trace_profile_samples={args.samples}"]
    if args.frame is not None:
        command += [f"--trace_profile_frame={args.frame}",
                    f"--trace_profile_first_command={args.first_command}",
                    f"--trace_profile_last_command={args.last_command}"]
    (directory / "command.json").write_text(json.dumps(command, indent=2))
    with (directory / "stdout.log").open("wb") as out, \
            (directory / "stderr.log").open("wb") as err:
        code = subprocess.run(command, cwd=directory, stdout=out, stderr=err,
                              timeout=args.timeout).returncode
    if code != 0:
        raise RuntimeError(f"Replay exited with {code}; inspect {directory}")
    result = json.loads(profile.read_text())
    if not result["valid"]:
        raise RuntimeError(f"Replay validation failed; inspect {directory}")
    result["verification_sha256"] = sha(result["verification_file"])
    return result


def run_corpus(args):
    binaries = {"binary": args.binary.resolve(),
                "reference": args.reference.resolve()}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for entry in parse_corpus(args.corpus.read_text()):
        if shutil.disk_usage(output).free < 5 * 1024**3:
            raise RuntimeError("Less than 5 GiB free before the next trace")
        trace = args.trace_dir.resolve() / f"{entry['name']}.xtr"
        window = argparse.Namespace(
            arg=[*args.arg, *entry["args"]], warmup=2, samples=1,
            frame=entry["frame"], first_command=1,
            last_command=entry["last_command"], timeout=args.timeout)
        row = {**entry}
        try:
            with tempfile.TemporaryDirectory(
                    prefix="xenia-exact-corpus-") as directory:
                for arm, binary in binaries.items():
                    row[arm] = run_process(binary, trace,
                                           Path(directory) / arm, [],
                                           window)["verification_sha256"]
            row["match"] = row["binary"] == row["reference"]
        except Exception as error:
            row["error"] = str(error)
            row["match"] = False
        rows.append(row)
        print(f"{entry['name']}: "
              f"{'match' if row['match'] else row.get('error', 'DIFFERS')}",
              flush=True)
        (output / "summary.json").write_text(json.dumps(
            {"binaries": {k: str(v) for k, v in binaries.items()},
             "traces": rows}, indent=2) + "\n")
    return 0 if all(row["match"] for row in rows) else 1


def run_ab(args, parser):
    if args.binary and (args.a_binary or args.b_binary):
        parser.error("Use --binary or both --a-binary and --b-binary")
    if not args.binary and not (args.a_binary and args.b_binary):
        parser.error("Use --binary or both --a-binary and --b-binary")
    if not args.trace:
        parser.error("--trace is required")
    if args.frame is not None and args.last_command is None:
        parser.error("--frame requires --last-command")
    if args.processes < 2 or args.samples < 1 or args.warmup < 2:
        parser.error("Use at least 2 processes, 1 sample and 2 warmups")

    binaries = ([args.binary.resolve()] * 2 if args.binary else
                [args.a_binary.resolve(), args.b_binary.resolve()])
    configs = [json.loads(path.read_text()) if path else {}
               for path in (args.a_config, args.b_config)]
    flags = [config_flags(config) for config in configs]
    trace, output = args.trace.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "provenance.json").write_text(json.dumps({
        "binaries": [str(binary) for binary in binaries],
        "binary_sha256": [sha(binary) for binary in binaries],
        "configs": configs, "args": args.arg, "trace": str(trace),
        "trace_sha256": sha(trace), "host": platform.platform(),
        "processes": args.processes, "samples": args.samples,
        "warmup": args.warmup}, indent=2))

    results = {"a": [], "b": []}
    for index in range(args.processes):
        for arm in ("ab" if index % 2 == 0 else "ba"):
            arm_index = "ab".index(arm)
            results[arm].append(run_process(
                binaries[arm_index], trace, output / f"{index}{arm}",
                flags[arm_index], args))
            print(f"process {index + 1}{arm} complete", flush=True)

    runs = results["a"] + results["b"]
    hashes = [{run["verification_sha256"] for run in results[arm]}
              for arm in "ab"]
    if any(len(arm_hashes) != 1 for arm_hashes in hashes):
        raise RuntimeError("Verified output differs between processes of an arm")
    if hashes[0] != hashes[1] and not args.allow_output_mismatch:
        raise RuntimeError("Verified output differs between arms")
    counts = [{key: sample["counts"][key] for key in GUEST_COUNTS}
              for run in runs for sample in run["samples"]]
    if any(count != counts[0] for count in counts):
        raise RuntimeError("Guest work differs between samples")
    if len({run["device"] for run in runs}) != 1:
        raise RuntimeError("GPU device differs between processes")

    def process_means(arm, key):
        return [statistics.mean(sample[key] for sample in run["samples"])
                for run in results[arm]]

    metrics = {name: paired_result(process_means("a", key),
                                   process_means("b", key))
               for name, key in METRICS.items()}
    summary = {"method": "per-process means; paired log ratios; "
                         "deterministic 5000-resample bootstrap",
               "positive_means": "B takes less time than A",
               "verification_sha256": [arm_hashes.pop() for arm_hashes in hashes],
               "guest_counts": counts[0], "metrics": metrics}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = ["| Metric | A ms/pass | B ms/pass | Reduction | 95% CI |",
             "|---|---:|---:|---:|---:|"]
    for name, row in metrics.items():
        lo, hi = row["ci95_percent"]
        lines.append(f"| {name} | {row['a_mean_ms']:.3f} | "
                     f"{row['b_mean_ms']:.3f} | "
                     f"{row['paired_reduction_percent']:+.2f}% | "
                     f"[{lo:+.2f}, {hi:+.2f}]% |")
    print("\n".join(lines))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--a-binary", type=Path)
    parser.add_argument("--b-binary", type=Path)
    parser.add_argument("--a-config", type=Path)
    parser.add_argument("--b-config", type=Path)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arg", action="append", default=[],
                        help="Additional flag passed to every replay")
    parser.add_argument("--frame", type=int)
    parser.add_argument("--first-command", type=int, default=1)
    parser.add_argument("--last-command", type=int)
    parser.add_argument("--processes", type=int, default=3,
                        help="Processes per arm")
    parser.add_argument("--samples", type=int, default=4,
                        help="Measured replays per process")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--allow-output-mismatch", action="store_true",
                        help="Only require identical output within each arm, "
                             "for changes that deliberately alter output")
    parser.add_argument("--corpus", type=Path,
                        help="Compare verified output against --reference "
                             "over the windows of this corpus file")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--trace-dir", type=Path)
    args = parser.parse_args()
    if args.corpus:
        if not (args.binary and args.reference and args.trace_dir):
            parser.error("--corpus requires --binary, --reference and "
                         "--trace-dir")
        return run_corpus(args)
    return run_ab(args, parser)


if __name__ == "__main__":
    raise SystemExit(main())
