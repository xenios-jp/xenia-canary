#!/usr/bin/env python3
"""Runs a built gpugen suite in Xenia or on a console and compares runs.

    python3 tools/gpugen/run.py xenia OUTDIR
    python3 tools/gpugen/run.py replay OUTDIR
    python3 tools/gpugen/run.py console OUTDIR [OUTDIR...]
    python3 tools/gpugen/run.py diff A.json B.json
    python3 tools/gpugen/run.py expect OUTDIR EXPECTED.json --source "..."
    python3 tools/gpugen/run.py check EXPECTED.json OUTDIR

Xenia and console runs work like xamgen's (tools/xamgen/run.py, whose helpers
this uses): the executor leaves gpugen.bin beside the XEX or on the console's
Hdd:\\oracle, and both decode to OUTDIR/<where>_results.json, a raw suite's
results to OUTDIR/<where>_<test>.bin. A console run needs a build with the
command processor microcode.

With GPUGEN_AURORA set to ftp://USER:PASSWORD@HOST/FOLDER, console runs
through Aurora's FTP instead, with no XBDM: it puts each OUTDIR's XEX in
FOLDER as <results>.xex (and a play.py build's memory file, unless it is
there from an earlier run), has Aurora launch the first, and collects every
<results>.bin once Aurora is back. Several OUTDIRs are a chain built with
--results and --next, each launching the next; a run that stopped short
rebooted the console, so the chain ends there.

replay plays OUTDIR/gpugen.xtr, the stream without the executor, through a
backend's trace dump (TRACE_DUMP, or xenia-gpu-<gpu>-trace-dump beside the
Release build), with no window, and decodes the outputs it leaves in memory
like a Xenia run's. It records which trace dump ran, with its arguments, in
OUTDIR/xenia_provenance.json. A play build's stages replay from the eDRAM
the console had before the trace when its run read it out (--edram, by
default OUTDIR/console_edram_before.bin), and from eDRAM all zeros and all
ones, whose disagreement marks the pixels that show it.

expect keeps a console run of OUTDIR as the suite's expected results, the
ones checked in as src/xenia/gpu/testing/gpu__gen_<suite>.json: the run's
source and the gpugen.xtr it ran, with the results file beside it,
xz-compressed, as gpu__gen_<suite>.bin.xz. check compares OUTDIR's
xenia_results.json (a replay's or a Xenia run's) with them, failing on any
word that differs or test not run, and refuses a build of other tests.
"""

import argparse
import ftplib
import hashlib
import importlib.util
import io
import json
import lzma
import os
import re
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gpugen  # noqa: E402
import play  # noqa: E402

# xamgen's run.py, by path since this file shares its name.
_spec = importlib.util.spec_from_file_location(
    'xamgen_run', os.path.join(os.path.dirname(HERE), 'xamgen', 'run.py'))
xamgen_run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(xamgen_run)

XENIA = xamgen_run.XENIA
# Every draw rendered: asynchronous compilation skips or stands in for a
# draw whose pipeline isn't ready.
REPLAY_ARGS = ['--async_shader_compilation=false']
# Where Aurora's FTP shows a title's cache: drive; game: is the folder the
# XEX runs from.
AURORA_CACHE = ('/HddX', '/Hdd1/Cache')
# A reboot, and Aurora starting and bringing its FTP back up.
AURORA_BOOT_SECONDS = 300
# FTP transfers in big blocks: ftplib's 8 KiB ones are slow with Aurora.
AURORA_BLOCK_BYTES = 1 << 20
# A play title answering no status probe for this long while it runs its
# segments has frozen the console. Allocating the block keeps the network
# quiet for longer.
STATUS_SILENT_SECONDS = 5


def decode_into(outdir, where, results_path):
    with open(os.path.join(outdir, 'gpugen.json')) as f:
        meta = json.load(f)
    with open(results_path, 'rb') as f:
        decoded = gpugen.decode(meta, f.read())
    # Only a console's run reads its eDRAM out; a replay's holds the eDRAM
    # it started from, if it was seeded.
    if 'edram' in meta and where != 'console' and not \
            seed_of(outdir, where):
        decoded.get('raw', {}).pop(meta['edram']['name'], None)
    gpugen.save_raw(decoded, outdir, where)
    out = os.path.join(outdir, f'{where}_results.json')
    with open(out, 'w') as f:
        json.dump(decoded, f, indent=1)
    if 'failed' in decoded:
        stopped = f', {decoded["failed"]}'
    elif decoded['segments_done'] == 0:
        stopped = ', stopped in the setup'
    elif decoded['not_run']:
        stopped = f', stopped in {decoded["not_run"][0]!r}'
    else:
        stopped = ''
    print(f'{where}: {decoded["segments_done"]} of {decoded["segments"]} '
          f'segments, {len(decoded["tests"])} tests{stopped} -> {out}')
    return decoded


def run_xenia(outdir, timeout, extra_args):
    outdir = os.path.abspath(outdir)
    if not os.path.exists(XENIA):
        sys.exit(f'xenia: no {XENIA}, set XENIA_EXE to the build to run')
    xex = os.path.join(outdir, 'gpugen.xex')
    log = os.path.join(outdir, 'xenia.log')
    results = os.path.join(outdir, 'gpugen.bin')
    for path in (results, log):
        if os.path.exists(path):
            os.remove(path)
    from_wsl = xamgen_run.from_wsl(XENIA)
    before = xamgen_run.windows_pids(XENIA) if from_wsl else set()
    started = time.time()
    process = subprocess.Popen(
        [XENIA, *xamgen_run.XENIA_ARGS, *extra_args,
         f'--log_file={xamgen_run.host_path(log, XENIA)}',
         xamgen_run.host_path(xex, XENIA)],
        cwd=os.path.dirname(XENIA), stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL)
    deadline = started + timeout
    summary = None
    try:
        while time.time() < deadline and process.poll() is None:
            time.sleep(0.5)
            if not os.path.exists(log):
                continue
            with open(log, errors='replace') as f:
                for line in f:
                    if '(DbgPrint) gpugen:' in line:
                        summary = line.strip()
            if summary and os.path.exists(results):
                time.sleep(0.5)
                break
    finally:
        exit_code = process.poll()
        if from_wsl:
            for pid in xamgen_run.windows_pids(XENIA) - before:
                subprocess.run(['taskkill.exe', '/PID', str(pid), '/F'],
                               capture_output=True)
        elif exit_code is None:
            process.kill()
        process.wait(timeout=10)
    if not summary:
        folder, dumps = xamgen_run.crash_dumps(started)
        if dumps:
            sys.exit(f'xenia crashed, dump {dumps[0]} in {folder}')
        if exit_code is not None:
            sys.exit(f'xenia exited with {exit_code} before the summary, '
                     f'see {log}')
        sys.exit(f'xenia: no summary within {timeout} s, see {log}')
    written = re.search(r'results file ([0-9A-Fa-f]{8})', summary)
    if not written or int(written.group(1), 16):
        sys.exit(f'xenia: no results from this run, summary {summary!r}')
    print(f'xenia: {summary}')
    copy = os.path.join(outdir, 'xenia_results.bin')
    os.replace(results, copy)
    return decode_into(outdir, 'xenia', copy)


def trace_dump(gpu):
    if os.environ.get('TRACE_DUMP'):
        return os.environ['TRACE_DUMP']
    name = f'xenia-gpu-{gpu}-trace-dump'
    if sys.platform == 'win32' or xamgen_run.WSL:
        name += '.exe'
    return os.path.join(os.path.dirname(xamgen_run.default_xenia()), name)


def record_provenance(outdir, dump, args, seed=None):
    """OUTDIR/xenia_provenance.json: the trace dump a replay ran (its hash,
    and a wrapper script's text, which names the binary and its cvars), its
    arguments, the git tree it lies in as it was at replay time, which the
    binary may predate, and `seed`, the eDRAM dump a play build's replay
    started from."""
    real = os.path.realpath(dump)
    with open(real, 'rb') as f:
        data = f.read()
    out = {'trace_dump': real, 'sha256': hashlib.sha256(data).hexdigest(),
           'mtime': time.strftime('%Y-%m-%dT%H:%M:%S%z',
                                  time.localtime(os.path.getmtime(real))),
           'args': list(args),
           'replayed': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
           'tree_at_replay': None, 'edram_seed': seed}
    if data.startswith(b'#!'):
        out['wrapper'] = data.decode(errors='replace')

    def git(*command):
        return subprocess.run(['git', '-C', os.path.dirname(real), *command],
                              capture_output=True, text=True, timeout=30,
                              check=True).stdout.strip()
    try:
        out['tree_at_replay'] = {
            'path': git('rev-parse', '--show-toplevel'),
            'commit': git('rev-parse', 'HEAD'),
            'committed': git('log', '-1', '--format=%cI'),
            'dirty': bool(git('status', '--porcelain',
                              '--untracked-files=no'))}
    except (OSError, subprocess.SubprocessError):
        pass
    with open(os.path.join(outdir, 'xenia_provenance.json'), 'w') as f:
        json.dump(out, f, indent=1)


def read_edram(path, meta):
    """The eDRAM dump in `path`: the dump itself (a console run's
    console_edram_before.bin) or a results file holding it, xz-compressed
    or not, of this build."""
    with open(path, 'rb') as f:
        data = f.read()
    if path.endswith('.xz'):
        data = lzma.decompress(data)
    if data[:4] == struct.pack('>I', gpugen.RESULTS_MAGIC):
        data = gpugen.decode(meta, data).get('raw', {}).get(
            meta['edram']['name'])
        if data is None:
            sys.exit(f'replay: {path} has no eDRAM dump')
    if len(data) != play.EDRAM_BYTES:
        sys.exit(f'replay: {path} is no eDRAM dump')
    return data


def seed_of(outdir, where='xenia'):
    """The SHA-1 of the eDRAM dump OUTDIR's replay started from, as its
    provenance records, or None."""
    if where != 'xenia':
        return None
    try:
        with open(os.path.join(outdir, 'xenia_provenance.json')) as f:
            return (json.load(f).get('edram_seed') or {}).get('sha1')
    except OSError:
        return None


def run_replay(outdir, gpu, timeout, extra_args, edram=None):
    outdir = os.path.abspath(outdir)
    dump = trace_dump(gpu)
    if not os.path.exists(dump):
        sys.exit(f'replay: no {dump}, set TRACE_DUMP to the one to run')
    with open(os.path.join(outdir, 'gpugen.json')) as f:
        meta = json.load(f)
    seed = seed_data = None
    if 'edram' in meta:
        if edram is None:
            edram = os.path.join(outdir,
                                 f'console_{meta["edram"]["name"]}.bin')
            edram = edram if os.path.exists(edram) else None
        if edram is not None:
            seed_data = read_edram(os.path.abspath(edram), meta)
            seed = {'source': os.path.abspath(edram),
                    'sha1': hashlib.sha1(seed_data).hexdigest()}
    elif edram is not None:
        sys.exit('replay: this build reads no eDRAM out')
    record_provenance(outdir, dump, REPLAY_ARGS + list(extra_args), seed)
    if 'play' in meta:
        return run_play_replay(outdir, dump, meta, timeout, extra_args,
                               seed_data)
    memory = os.path.join(outdir, 'gpugen_mem.bin')
    if os.path.exists(memory):
        os.remove(memory)
    log = os.path.join(outdir, 'replay.log')
    subprocess.run(
        [dump, f'--log_file={xamgen_run.host_path(log, dump)}',
         f'--trace_dump_path={xamgen_run.host_path(outdir, dump)}',
         *REPLAY_ARGS, '--trace_dump_memory_frame=0',
         f'--trace_dump_memory_base={meta["trace"]["base"]}',
         f'--trace_dump_memory_size={4 * meta["trace"]["words"]}',
         *extra_args,
         xamgen_run.host_path(os.path.join(outdir, 'gpugen.xtr'), dump)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout)
    if not os.path.exists(memory):
        sys.exit(f'replay: no memory written, see {log}')
    with open(memory, 'rb') as f:
        body = f.read()
    # The executor's results header: every segment done.
    header = struct.pack('>IIIiI', gpugen.RESULTS_MAGIC, gpugen.VERSION,
                         meta['segments'], 0, 0x20 + len(body))
    copy = os.path.join(outdir, 'xenia_results.bin')
    with open(copy, 'wb') as f:
        f.write(header + bytes(0x20 - len(header)) + body)
    return decode_into(outdir, 'xenia', copy)


def replay_stage(dump, trace, stage, directory, timeout, extra_args):
    """The memory a stage's resolve wrote, from the trace replayed up to
    its command."""
    os.makedirs(directory, exist_ok=True)
    for name in os.listdir(directory):
        os.remove(os.path.join(directory, name))
    log = os.path.join(directory, 'replay.log')
    subprocess.run(
        [dump, f'--log_file={xamgen_run.host_path(log, dump)}',
         f'--trace_dump_path={xamgen_run.host_path(directory, dump)}',
         *REPLAY_ARGS, '--trace_dump_edram_frame=0',
         f'--trace_dump_stop_command={stage["command"]}',
         '--trace_dump_memory_frame=0',
         f'--trace_dump_memory_base={stage["base"]}',
         f'--trace_dump_memory_size={stage["size"]}', *extra_args,
         xamgen_run.host_path(trace, dump)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout)
    # Stopped before the frame's end, it writes no image, only the memory.
    memory = [name for name in os.listdir(directory)
              if name.endswith('_mem.bin')]
    if len(memory) != 1:
        sys.exit(f'replay: no memory written for command '
                 f'{stage["command"]}, see {log}')
    with open(os.path.join(directory, memory[0]), 'rb') as f:
        data = f.read()
    if len(data) != stage['size']:
        sys.exit(f'replay: {len(data)} bytes for command {stage["command"]}, '
                 f'not {stage["size"]}')
    return data


def run_play_replay(outdir, dump, meta, timeout, extra_args, edram=None):
    """A play build's stages, each from the trace replayed up to its
    command, as xenia_results.bin, starting from `edram`, the console's
    eDRAM dump, if given; and as xenia_edram00_results.bin and
    xenia_edramff_results.bin, from replays starting from eDRAM all zeros
    and all ones, which tell the pixels a stage shows of eDRAM from before
    the trace. Each holds the eDRAM it started from where the console's run
    holds its dump, the unseeded replay nothing."""
    trace = os.path.join(outdir, 'gpugen.xtr')
    with tempfile.TemporaryDirectory(dir=outdir) as temp:
        traces = {'xenia': trace}
        seeds = {}
        if edram is not None:
            traces['xenia'] = os.path.join(temp, 'edram.xtr')
            play.seeded_trace(trace, traces['xenia'],
                              play.edram_snapshot(edram))
            seeds['xenia'] = edram
        for fill in (0x00, 0xFF):
            where = f'xenia_edram{fill:02x}'
            traces[where] = os.path.join(temp, f'edram{fill:02x}.xtr')
            play.seeded_trace(trace, traces[where], fill)
            seeds[where] = bytes([fill]) * play.EDRAM_BYTES
        last = meta['tests'][-1]
        bodies = {where: bytearray(last['offset'] + last['output_bytes'] +
                                   3 & ~3) for where in traces}
        if 'edram' in meta:
            at = meta['edram']['offset']
            for where, seed in seeds.items():
                bodies[where][at:at + len(seed)] = seed
        for test in meta['tests']:
            for where, path in traces.items():
                data = replay_stage(dump, path, test['stage'],
                                    os.path.join(temp, 'stage'), timeout,
                                    extra_args)
                bodies[where][test['offset']:test['offset'] + len(data)] = data
            print(f'replay: {test["name"]}', flush=True)
    for where, body in bodies.items():
        header = struct.pack('>IIIiI', gpugen.RESULTS_MAGIC, gpugen.VERSION,
                             meta['segments'], 0, 0x20 + len(body))
        with open(os.path.join(outdir, f'{where}_results.bin'), 'wb') as f:
            f.write(header + bytes(0x20 - len(header)) + body)
    return decode_into(outdir, 'xenia',
                       os.path.join(outdir, 'xenia_results.bin'))


def run_console(outdir, timeout):
    outdir = os.path.abspath(outdir)
    with open(os.path.join(outdir, 'gpugen.json')) as f:
        meta = json.load(f)
    if not meta.get('console_ready'):
        sys.exit('console: this build has no command processor microcode')
    rgh = xamgen_run.rgh
    rgh_exe = xamgen_run.rgh_exe()
    xex = f'{xamgen_run.CONSOLE_DIR}\\gpugen.xex'
    results = f'{xamgen_run.CONSOLE_DIR}\\gpugen.bin'
    code, out = rgh('title')
    if 'Aurora' not in out:
        sys.exit('console: not at Aurora, refusing to launch')
    code, out = rgh('fs', 'put', '--path', xex, '--in',
                    xamgen_run.host_path(os.path.join(outdir, 'gpugen.xex'),
                                         rgh_exe), timeout=180)
    if code:
        sys.exit(f'console: upload failed\n{out}')
    rgh('fs', 'rm', '--path', results)
    code, out = rgh('launch', '--xex', xex)
    if code:
        sys.exit(f'console: launch failed\n{out}')
    watch_log = os.path.join(outdir, 'console_watch.jsonl')
    if os.path.exists(watch_log):
        os.remove(watch_log)
    watcher = subprocess.Popen(
        [rgh_exe, 'debug', 'watch', '--duration', str(timeout), '--raw',
         '--out', xamgen_run.host_path(watch_log, rgh_exe)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + timeout
    fault = summary = None
    try:
        while time.time() < deadline and watcher.poll() is None:
            time.sleep(0.5)
            events = xamgen_run.watch_events(watch_log)
            for event in events:
                raw = event.get('Raw', '')
                if event.get('Type') == 'debugstr' and 'gpugen:' in raw:
                    summary = raw
                if event.get('Type') == 'exception':
                    fault = raw
            if fault or any(e.get('Type') == 'modunload' and
                            'gpugen.xex' in e.get('Raw', '') for e in events):
                break
    finally:
        watcher.terminate()
        watcher.wait()
    if fault or not summary:
        code, out = rgh('mem', 'peek', '--addr',
                        f'{meta["control"] + gpugen.C_PROGRESS:#x}',
                        '--type', 'u32')
        stages = ', '.join(f'{i} {name}'
                           for i, name in enumerate(gpugen.PROGRESS, 1))
        if fault:
            rgh('launch', '--xex', xamgen_run.AURORA)
        sys.exit(f'console: {fault or "no summary"}\nprogress: {out.strip()}'
                 f'\n({stages})')
    written = re.search(r'results file ([0-9A-Fa-f]{8})', summary)
    if not written or int(written.group(1), 16):
        sys.exit(f'console: no results from this run, summary {summary!r}')
    print(f'console: {summary}')
    for _ in range(30):
        code, out = rgh('title')
        if 'Aurora' in out:
            break
        time.sleep(1)
    copy = os.path.join(outdir, 'console_results.bin')
    code, out = rgh('fs', 'get', '--path', results, '--out',
                    xamgen_run.host_path(copy, rgh_exe))
    if code:
        sys.exit(f'console: no results file\n{out}')
    return decode_into(outdir, 'console', copy)


def aurora():
    """A function connecting to Aurora's FTP as GPUGEN_AURORA says, the
    folder the XEXes go to and the console's host."""
    url = urllib.parse.urlsplit(os.environ['GPUGEN_AURORA'])
    folder = urllib.parse.unquote(url.path).rstrip('/')
    if url.scheme != 'ftp' or not url.hostname or not folder:
        sys.exit('console: GPUGEN_AURORA is not '
                 'ftp://USER:PASSWORD@HOST/FOLDER')

    def connect():
        f = ftplib.FTP(timeout=10)
        f.connect(url.hostname, url.port or 21)
        f.login(urllib.parse.unquote(url.username or ''),
                urllib.parse.unquote(url.password or ''))
        return f
    return connect, folder, url.hostname


def aurora_files(f, folder):
    """The names in `folder`, made the current directory; Aurora's NLST
    lists like LIST, so the name is the last field."""
    try:
        f.cwd(folder)
        return {line.split(None, 8)[-1] for line in f.nlst()}
    except ftplib.Error:
        return set()


def aurora_put_memory(f, have, path, name):
    """Puts a play build's memory file in the current folder as NAME, which
    its SHA-1 names, unless it is there; a whole one, from a rename once it
    is all there."""
    if name in have:
        print(f'console: {name} already there', flush=True)
        return
    with open(path, 'rb') as mem:
        f.storbinary(f'STOR {name}.part', mem, blocksize=AURORA_BLOCK_BYTES)
    f.rename(f'{name}.part', name)


def run_aurora(outdirs):
    outdirs = [os.path.abspath(outdir) for outdir in outdirs]
    metas = []
    for outdir in outdirs:
        with open(os.path.join(outdir, 'gpugen.json')) as f:
            metas.append(json.load(f))
    for outdir, meta, following in zip(outdirs, metas, metas[1:] + [None]):
        if not meta.get('console_ready'):
            sys.exit(f'console: {outdir} has no command processor microcode')
        if meta['next'] != (following and following['results']):
            sys.exit(f'console: {outdir} is built to launch '
                     f'{meta["next"]!r} next, not the next OUTDIR')
    names = [meta['results'] for meta in metas]
    if len(set(names)) != len(names):
        sys.exit('console: two OUTDIRs write the same results')
    connect, folder, host = aurora()
    places = (folder, *AURORA_CACHE)
    with connect() as f:
        for place in places:
            stale = aurora_files(f, place)
            for name in names:
                if f'{name}.bin' in stale:
                    f.delete(f'{name}.bin')
        have = aurora_files(f, folder)
        for outdir, name, meta in zip(outdirs, names, metas):
            with open(os.path.join(outdir, 'gpugen.xex'), 'rb') as xex:
                f.storbinary(f'STOR {name}.xex', xex,
                             blocksize=AURORA_BLOCK_BYTES)
            if meta.get('memory'):
                aurora_put_memory(f, have, os.path.join(outdir, 'gpugen.mem'),
                                  meta['memory'])
    # For a while after a boot Aurora refuses to launch titles. Its FTP goes
    # away with it, so the connection closes without QUIT.
    for _ in range(30):
        try:
            f = connect()
            f.sendcmd(f'SITE LAUNCH {folder}/{names[0]}.xex')
            f.close()
            break
        except (OSError, ftplib.Error) as e:
            print(f'console: launch refused ({e}), retrying', flush=True)
            time.sleep(10)
    else:
        sys.exit('console: Aurora kept refusing the launch')
    print(f'console: launched {names[0]}.xex', flush=True)
    deadline = (time.time() + sum(meta['watchdog'] for meta in metas) +
                AURORA_BOOT_SECONDS)
    got = {}
    stopped = None
    status = Status(host) if any('play' in meta for meta in metas) else None
    while len(got) < len(names) and not stopped and time.time() < deadline:
        if status:
            if status.watch(5):
                sys.exit(f'console: froze, {status.describe()}')
        else:
            time.sleep(5)
        try:
            with connect() as f:
                for place in places:
                    have = aurora_files(f, place)
                    for name in names:
                        if name not in got and f'{name}.bin' in have:
                            data = io.BytesIO()
                            f.retrbinary(f'RETR {name}.bin', data.write,
                                         blocksize=AURORA_BLOCK_BYTES)
                            got[name] = data.getvalue()
        except (OSError, ftplib.Error):
            continue  # down while a title runs
        for meta in metas:
            data = got.get(meta['results'])
            if data and gpugen.decode(meta, data)['segments_done'] != \
                    meta['segments']:
                stopped = meta['results']
    for outdir, name in zip(outdirs, names):
        if name not in got:
            sys.exit(f'console: no results from {name}.xex on')
        copy = os.path.join(outdir, 'console_results.bin')
        with open(copy, 'wb') as f:
            f.write(got[name])
        decode_into(outdir, 'console', copy)
        if name == stopped:
            sys.exit(f'console: {name} stopped short and rebooted the '
                     'console, the chain ended there')


class Status:
    """A play title's status thread, probed over UDP: prints its progress as
    it goes and tells when it stops answering."""

    def __init__(self, host):
        self.address = (host, gpugen.STATUS_PORT)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.settimeout(1)
        self.last = None    # (elapsed, progress, segment, step, error)
        self.answered = None        # when it last answered

    def describe(self):
        elapsed, progress, segment, step, error = self.last
        phase = (gpugen.PROGRESS[progress - 1] if 0 < progress <=
                 len(gpugen.PROGRESS) else progress)
        return (f'{elapsed} s in, {phase}, segment {segment} step {step}, '
                + (f'receive error {error}, ' if error else '') +
                f'last answer {time.time() - self.answered:.0f} s ago')

    def watch(self, seconds):
        """Probes for `seconds`; True once a title running its segments has
        been silent for STATUS_SILENT_SECONDS."""
        end = time.time() + seconds
        while time.time() < end:
            sent = time.time()
            try:
                self.socket.sendto(b'gpugen', self.address)
                reply, _ = self.socket.recvfrom(64)
            except OSError:
                reply = b''
            if len(reply) == gpugen.STATUS_REPLY_BYTES:
                magic, *last = struct.unpack('>6I', reply)
                if magic == gpugen.STATUS_MAGIC:
                    if (self.last is None or last[1:3] != self.last[1:3] or
                            time.time() - self.answered > 2):
                        self.last, self.answered = tuple(last), time.time()
                        print(f'console: {self.describe()}', flush=True)
                    self.last, self.answered = tuple(last), time.time()
            elif (self.answered and self.last[1] == gpugen.P_SUBMIT and
                    time.time() - self.answered > STATUS_SILENT_SECONDS):
                return True
            time.sleep(max(0.0, sent + 1 - time.time()))
        return False


def load_words(path):
    """{test: hex words} from a results JSON, reading a raw suite's .bin
    files beside it."""
    with open(path) as f:
        tests = json.load(f)['tests']
    for name, words in tests.items():
        if isinstance(words, str):
            with open(os.path.join(os.path.dirname(path), words), 'rb') as f:
                data = f.read()
            tests[name] = [f'{w:08X}' for w in
                           struct.unpack(f'>{len(data) // 4}I', data)]
    return tests


def compare(a_path, b_path, quiet=False):
    """Prints the tests of A and how B's words differ, only the ones that do
    if `quiet`; returns the tests the same, differing and not run in B."""
    a, b = load_words(a_path), load_words(b_path)
    same = differ = not_run = 0
    for name, want in a.items():
        got = b.get(name)
        if got is None:
            not_run += 1
            print(f'{name}: not run in {b_path}')
            continue
        bad = [i for i, (x, y) in enumerate(zip(want, got)) if x != y]
        if bad or len(want) != len(got):
            differ += 1
            print(f'{name}: {len(bad)} of {len(want)} words differ, first '
                  f'{[(i, want[i], got[i]) for i in bad[:4]]}')
        else:
            same += 1
            if not quiet:
                print(f'{name}: all {len(want)} words same')
    print(f'{same} tests same, {differ} different' +
          (f', {not_run} not run' if not_run else ''))
    return same, differ, not_run


def diff(a_path, b_path):
    compare(a_path, b_path)


def sha1(path):
    with open(path, 'rb') as f:
        return hashlib.sha1(f.read()).hexdigest()


def results_path(expected_path):
    return expected_path[:-len('.json')] + '.bin.xz'


def expect(outdir, expected_path, source):
    if not expected_path.endswith('.json'):
        sys.exit(f'{expected_path}: not a .json')
    with open(os.path.join(outdir, 'gpugen.json')) as f:
        meta = json.load(f)
    with open(os.path.join(outdir, 'console_results.bin'), 'rb') as f:
        data = f.read()
    decoded = gpugen.decode(meta, data)
    if decoded['segments_done'] != decoded['segments']:
        sys.exit(f'{outdir}: the console run stopped in '
                 f'{decoded["not_run"][0]!r}')
    with open(results_path(expected_path), 'wb') as f:
        f.write(lzma.compress(data, preset=9 | lzma.PRESET_EXTREME))
    with open(expected_path, 'w') as f:
        json.dump({'suite': meta['suite'], 'source': source,
                   'xtr': sha1(os.path.join(outdir, 'gpugen.xtr')),
                   'results': hashlib.sha1(data).hexdigest()}, f, indent=1)
        f.write('\n')
    print(f'{len(decoded["tests"])} tests -> {expected_path}')


def check(expected_path, outdir):
    with open(expected_path) as f:
        expected = json.load(f)
    if sha1(os.path.join(outdir, 'gpugen.xtr')) != expected['xtr']:
        sys.exit(f'{outdir} is a build other than the one {expected_path} '
                 'expects: rebuild it, or record the suite again')
    actual = os.path.join(outdir, 'xenia_results.json')
    if not os.path.exists(actual):
        sys.exit(f'{outdir}: no xenia_results.json, run it in Xenia or '
                 'replay it first')
    with open(results_path(expected_path), 'rb') as f:
        data = lzma.decompress(f.read())
    if hashlib.sha1(data).hexdigest() != expected['results']:
        sys.exit(f'{results_path(expected_path)}: corrupt')
    with open(os.path.join(outdir, 'gpugen.json')) as f:
        meta = json.load(f)
    if 'play' in meta:
        check_play(meta, data, outdir)
        return
    with tempfile.TemporaryDirectory() as temp:
        for name in ('gpugen.json',):
            with open(os.path.join(outdir, name), 'rb') as src, \
                    open(os.path.join(temp, name), 'wb') as dst:
                dst.write(src.read())
        results = os.path.join(temp, 'expected.bin')
        with open(results, 'wb') as f:
            f.write(data)
        decode_into(temp, 'expected', results)
        _, differ, not_run = compare(
            os.path.join(temp, 'expected_results.json'),
            actual, quiet=True)
    if differ or not_run:
        sys.exit(1)


def check_play(meta, data, outdir):
    """Compares a play build's stages, the pixels of each resolve's
    destination. With the console's eDRAM dump, which the replay must have
    started from, every pixel; without, leaving out the ones the replays
    from eDRAM all zeros and all ones disagree on, which show eDRAM from
    before the trace."""
    results = {}
    for where in ('xenia', 'xenia_edram00', 'xenia_edramff'):
        path = os.path.join(outdir, f'{where}_results.bin')
        if not os.path.exists(path):
            sys.exit(f'{outdir}: no {where}_results.bin, replay it first')
        with open(path, 'rb') as f:
            results[where] = gpugen.decode(meta, f.read())['raw']
    expected = gpugen.decode(meta, data).get('raw', {})
    dump = expected.get(meta['edram']['name']) if 'edram' in meta else None
    seeded = dump is not None
    if seeded and not play.replay_seeded(outdir, dump):
        sys.exit(f'{outdir}: replayed from other eDRAM than the console\'s '
                 'dump, replay it with --edram of these results')
    differ = not_run = before = before_same = 0
    for test in meta['tests']:
        name = test['name']
        if name not in expected:
            not_run += 1
            print(f'{name}: not run on the console')
            continue
        differing, first, undefined, same = play.compare_stage(
            test['stage'], expected[name], results['xenia'][name],
            results['xenia_edram00'][name], results['xenia_edramff'][name],
            mask=not seeded)
        before += undefined
        before_same += same
        note = ('' if not undefined else
                f', {undefined} from before the trace ({same} same)' if seeded
                else f', {undefined} undefined')
        if differing:
            differ += 1
            print(f'{name}: {differing} pixels differ, first at {first}'
                  f'{note}')
        else:
            print(f'{name}: same{note}')
    print(f'{len(meta["tests"]) - differ - not_run} stages same, {differ} '
          'different' + (f', {not_run} not run' if not_run else ''))
    if seeded:
        print(f'{before} pixels show eDRAM from before the trace (the '
              'eDRAM-00 and -FF replays disagree), compared from the '
              f'console\'s dump: {before_same} same, '
              f'{before - before_same} different')
    if differ or not_run:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('xenia')
    p.add_argument('outdir')
    p.add_argument('--timeout', type=int, default=30)
    p.add_argument('--xenia-arg', action='append', default=[],
                   help='extra Xenia argument, e.g. --xenia-arg=--gpu=vulkan')
    p = sub.add_parser('replay')
    p.add_argument('outdir')
    p.add_argument('--gpu', default='metal' if sys.platform == 'darwin'
                   else 'vulkan')
    p.add_argument('--timeout', type=int, default=120)
    p.add_argument('--dump-arg', action='append', default=[],
                   help='extra trace dump argument')
    p.add_argument('--edram', metavar='FILE',
                   help='a play build\'s replay starts from the console\'s '
                   'eDRAM dump in FILE, the dump or results holding it '
                   '(default OUTDIR/console_edram_before.bin)')
    p = sub.add_parser('console')
    p.add_argument('outdir', nargs='+')
    p.add_argument('--timeout', type=int, default=30,
                   help='how long an XBDM run may take')
    p = sub.add_parser('diff')
    p.add_argument('a')
    p.add_argument('b')
    p = sub.add_parser('expect')
    p.add_argument('outdir')
    p.add_argument('expected')
    p.add_argument('--source', required=True,
                   help='where the run came from, e.g. the console and date')
    p = sub.add_parser('check')
    p.add_argument('expected')
    p.add_argument('outdir')
    args = parser.parse_args()
    if args.command == 'xenia':
        run_xenia(args.outdir, args.timeout, args.xenia_arg)
    elif args.command == 'replay':
        run_replay(args.outdir, args.gpu, args.timeout, args.dump_arg,
                   args.edram)
    elif args.command == 'console':
        if 'GPUGEN_AURORA' in os.environ:
            run_aurora(args.outdir)
        elif len(args.outdir) > 1:
            sys.exit('console: a chain of OUTDIRs runs through Aurora, set '
                     'GPUGEN_AURORA')
        else:
            run_console(args.outdir[0], args.timeout)
    elif args.command == 'expect':
        expect(args.outdir, args.expected, args.source)
    elif args.command == 'check':
        check(args.expected, args.outdir)
    else:
        diff(args.a, args.b)


if __name__ == '__main__':
    main()
