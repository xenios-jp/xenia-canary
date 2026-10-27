"""stage_report.py RUN [--mode MODE] [--config FILE]: the console-vs-Xenia
report for a play build, one command for the whole thing.

RUN holds the console's run of the build (console_results.bin, gpugen.json,
gpugen.xtr, notes.json). RUN/stage-report.json (or --config) names the
backend replays and where each report goes:

  {"title": "Metro: Last Light", "primary": "Metal RTV",
   "modes": {
     "isolated": {"out": "../9-isolated/diff2",
                  "backends": [["Metal RTV", "../9-isolated/metal"], ...],
                  "variants": [{"label": "Metal, console ALU",
                                "dir": "../9-isolated-rtz/metal",
                                "of": "Metal RTV", "area": "alu-rounding",
                                "change": "alu_round_toward_zero=true",
                                "name": "Console ALU rounding"}]},
     "played": {...}}}

Replay history: append variants rather than replacing old directories. Each
variant may record run_id, parent_run, and patches (name + SHA-256 of each
cumulative patch). selected_runs and comparison_runs map backend family names
to the default run labels. The viewer can compare any two recorded runs.
Use xenia_provenance.json build_source only for a verified executable source
commit; tree_at_replay describes the checkout, not the executable. Uncommitted
builds must retain their binary hash and complete cumulative patch identities.
Variants must use byte-identical trace, metadata, and initial eDRAM inputs to
their base backend. Settings changes belong in recorded args, not invented
patch entries. Existing result files should be retained under a new run path.

Paths are relative to RUN. "isolated" replays are of stage_isolate.py's trace,
where every stage's console output is put back in memory after its resolve;
"played" replays are of the plain trace. Each mode's report shows the other's
wrong-pixel share beside its own, and in played mode the isolated replay is
the counterfactual that proves a difference inherited. A variant is a named replay of a backend with a recorded cumulative patch
set and settings. Improvements establish an effect of that tested set;
they do not individually attribute a cause to every patch in the set.

Each report is OUT/report.html (one page, overview and per-pass drill-down),
OUT/report.json (everything the page shows), OUT/img/*.png (console, backend
and signed difference images, full size) and OUT/data/*.js (the words behind
the pixel inspector, loaded when it is first used). Nothing is inlined as
base64, so the page itself stays small.

Causes are split into what differs (the observation: coverage, depth values,
colour channels, signed step histograms) and why (hypotheses, each with a
pixel share and a confidence):
- Proven: a counterfactual replay (a variant, or isolation in played mode)
  puts these pixels right, or both sides kept an earlier pass's words in eDRAM
  (the difference is that pass's).
- Likely: the pattern matches a console rule from the coverage map and the
  pass's registers enable it (hi-Z/hi-S tests, gamma targets); the rule is
  cited.
- Unknown: nothing tested explains it yet.
- Not comparable: when the initial eDRAM seed cannot be verified against the
  console, the backend's eDRAM-00 and -FF probes identify pre-trace dependence.
  A verified console seed makes these pixels fully comparable.
Stage names come from the registers, overridden by RUN/notes.json "stages"
({"COMMAND": name}); frames come from the trace's swap events, never GPU interrupts."""
import argparse
import collections
import datetime
import hashlib
import html
import json
import math
import mmap
import os
import re
import shutil
import struct
import sys
import time
import zlib
import base64
from array import array
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import play  # noqa: E402
import pm4  # noqa: E402

TEMPLATE = HERE / 'stage_report_ui.html'
COVERAGE = HERE / 'coverage'

REG_NAMES = (
    'RB_SURFACE_INFO', 'RB_COLOR_INFO', 'RB_COLOR1_INFO', 'RB_COLOR2_INFO',
    'RB_COLOR3_INFO', 'RB_DEPTH_INFO', 'RB_DEPTHCONTROL', 'RB_COLOR_MASK',
    'RB_HIZCONTROL', 'RB_MODECONTROL', 'RB_COPY_CONTROL', 'RB_COPY_DEST_BASE',
    'RB_COPY_DEST_INFO', 'RB_COPY_DEST_PITCH', 'RB_BLENDCONTROL0',
    'RB_BLENDCONTROL1', 'RB_BLENDCONTROL2', 'RB_BLENDCONTROL3',
    'RB_COLORCONTROL', 'RB_STENCILREFMASK', 'RB_STENCILREFMASK_BF',
    'PA_SC_AA_CONFIG', 'PA_SU_SC_MODE_CNTL', 'RB_COLOR_CLEAR',
    'RB_DEPTH_CLEAR')
R = {name: pm4.register(name) for name in REG_NAMES}
COLOR_INFO = ('RB_COLOR_INFO', 'RB_COLOR1_INFO', 'RB_COLOR2_INFO',
              'RB_COLOR3_INFO')
COLOR_FORMATS = {
    0: '8_8_8_8', 1: '8_8_8_8_GAMMA', 2: '2_10_10_10',
    3: '2_10_10_10_FLOAT', 4: '16_16', 5: '16_16_16_16', 6: '16_16_FLOAT',
    7: '16_16_16_16_FLOAT', 10: '2_10_10_10_AS_10_10_10_10',
    12: '2_10_10_10_FLOAT_AS_16_16_16_16', 14: '32_FLOAT', 15: '32_32_FLOAT'}
DEST_FORMATS = {2: '8', 6: '8_8_8_8', 7: '2_10_10_10', 10: '8_8',
                24: '16', 26: '16_16_16_16', 31: '16_16_FLOAT',
                32: '16_16_16_16_FLOAT', 36: '32_FLOAT',
                37: '32_32_FLOAT', 38: '32_32_32_32_FLOAT'}
GAMMA = 1
FUNCS = ('NEVER', 'LESS', 'EQUAL', 'LEQUAL', 'GREATER', 'NOTEQUAL', 'GEQUAL',
         'ALWAYS')
STENCIL_OPS = ('KEEP', 'ZERO', 'REPLACE', 'INCR_SAT', 'DECR_SAT', 'INVERT',
               'INCR', 'DECR')
BLEND_FACTORS = {0: '0', 1: '1', 4: 'Cs', 5: '1-Cs', 6: 'As', 7: '1-As',
                 8: 'Cd', 9: '1-Cd', 10: 'Ad', 11: '1-Ad', 12: 'K', 13: '1-K',
                 14: 'Ka', 15: '1-Ka', 16: 'sat(As)'}
BLEND_OPS = ('+', 's-d', 'min', 'max', 'd-s')
PRIMS = {1: 'points', 2: 'lines', 3: 'line strip', 4: 'triangles',
         5: 'fan', 6: 'strip', 8: 'rects', 13: 'quads'}
NO_BLEND = 0x00010001
DEPTH_FAR = 1 << 16       # a depth step this large is another surface

# Signed step histograms: integer channels and logarithmic ones (depth in
# units of 2^-24, floats in ULPs).
STEP_BINS = ((-1 << 30, -16), (-15, -4), (-3, -3), (-2, -2), (-1, -1),
             (1, 1), (2, 2), (3, 3), (4, 15), (16, 1 << 30))
STEP_LABELS = ('≤−16', '−15…−4', '−3', '−2', '−1', '+1', '+2', '+3',
               '+4…15', '≥+16')
LOG_BINS = ((-1 << 40, -65536), (-65535, -256), (-255, -16), (-15, -4),
            (-3, -2), (-1, -1), (1, 1), (2, 3), (4, 15), (16, 255),
            (256, 65535), (65536, 1 << 40))
LOG_LABELS = ('≤−2¹⁶', '−2¹⁶…−256', '−255…−16', '−15…−4', '−3…−2', '−1',
              '+1', '+2…3', '+4…15', '+16…255', '+256…2¹⁶', '≥+2¹⁶')

# Heatmap colours (validated pair blue/red for the sign; coverage colours
# are always labelled in the legend and the inspector).
HIGH = (0xE3, 0x49, 0x48)
LOW = (0x2A, 0x78, 0xD6)
EXTRA = (0xF5, 0xC4, 0x00)      # Xenia drew, the console left the pixel
MISSING = (0x1B, 0xAF, 0x7A)    # the console drew, Xenia left it
PRETRACE = (0x8E, 0x8E, 0x93)
EDRAM = (0xB0, 0x7C, 0xE8)      # both kept an earlier pass's words


def log(*args):
    print(*args, flush=True)


# The trace.

def fetch_slots(code):
    """The texture fetch constants a shader's exec clauses fetch from."""
    used, i, end = set(), 0, len(code) // 3
    while i < end:
        d0, d1, d2 = code[3 * i:3 * i + 3]
        for cf in (d0 | (d1 & 0xFFFF) << 32, d1 >> 16 | d2 << 16):
            if cf >> 44 & 15 in (1, 2, 3, 4, 5, 6, 13, 14):
                address, count = cf & 0xFFF, cf >> 12 & 7
                if address:
                    end = min(end, address)
                for k in range(count):
                    if cf >> 16 >> 2 * k & 1:
                        fetch = code[3 * (address + k):3 * (address + k) + 3]
                        if len(fetch) == 3 and fetch[0] & 31 == 1:
                            used.add(fetch[0] >> 20 & 31)
        i += 1
    return used


class Walker(play.Player):
    """Every draw's registers, primitive and sampled textures, and where the
    rendering state. Frame boundaries are read separately from swap events."""

    def __init__(self):
        super().__init__()
        self.shader = {0: [], 1: []}
        self.draws = []
        self.swaps = []
        self.frame_boundary_source = 'none'

    def prepare(self, words, depth):
        header = words[0]
        op = header >> 8 & 127 if header >> 30 == 3 else None
        prepared = super().prepare(words, depth)
        if prepared is None:
            return None
        if op == 0x27 and len(words) >= 3:
            self.shader[words[1] & 3] = self.memory.words(
                words[1] & 0x1FFFFFFC, words[2] & 0xFFFF)
        elif op == 0x2B and len(words) >= 3:
            self.shader[words[1] & 3] = list(words[3:])
        if op in (0x22, 0x36) and prepared is not None:
            initiator = words[2] if op == 0x22 and len(words) > 2 else \
                words[1] if len(words) > 1 else 0
            slots = fetch_slots(self.shader[0]) | fetch_slots(self.shader[1])
            self.draws.append({
                'command': self.command,
                'regs': {n: self.registers.get(r, 0) for n, r in R.items()},
                'prim': initiator & 63, 'count': initiator >> 16,
                'textures': [self.fetch[6 * s + 1] & 0x1FFFF000
                             for s in sorted(slots)
                             if self.fetch[6 * s] & 3 == 2]})
        return prepared


def resolve_source(regs):
    """(kind, eDRAM base, surface format, samples) a resolve reads."""
    select = regs['RB_COPY_CONTROL'] & 7
    samples = 1 << (regs['RB_SURFACE_INFO'] >> 16 & 3)
    if select < 4:
        info = regs[COLOR_INFO[select]]
        return 'color', info & 0xFFF, info >> 16 & 15, samples
    info = regs['RB_DEPTH_INFO']
    return 'depth', info & 0xFFF, info >> 16 & 1, samples


def writes(regs, kind, base):
    mode = regs['RB_MODECONTROL'] & 7
    if kind == 'depth':
        dc = regs['RB_DEPTHCONTROL']
        return mode in (4, 5) and regs['RB_DEPTH_INFO'] & 0xFFF == base and \
            (mode == 5 or dc & 1 or dc & 6 == 6)
    return mode == 4 and any(
        regs[name] & 0xFFF == base and regs['RB_COLOR_MASK'] >> 4 * i & 15
        for i, name in enumerate(COLOR_INFO))


def hiz_text(v):
    bits = [name for bit, name in ((0, 'hi-Z write'), (1, 'hi-Z test'),
                                   (2, 'hi-S write'), (3, 'hi-S test'))
            if v >> bit & 1]
    if v >> 4 & 1:
        bits.append('greater')
    return ', '.join(bits) or 'off'


def depth_text(dc, mode):
    if mode == 5:
        pre = 'depth only, '
    else:
        pre = ''
    if not dc & 2 and not dc & 1:
        return pre + 'Z off'
    out = []
    if dc & 2:
        out.append(f'Z {FUNCS[dc >> 4 & 7]}' + (' write' if dc & 4 else ''))
    if dc & 1:
        out.append(f'stencil {FUNCS[dc >> 8 & 7]} '
                   f'{STENCIL_OPS[dc >> 11 & 7]}/{STENCIL_OPS[dc >> 14 & 7]}'
                   f'/{STENCIL_OPS[dc >> 17 & 7]}')
    return pre + ', '.join(out)


def blend_text(b):
    if b == NO_BLEND:
        return 'off'
    def part(src, op, dst):
        s, d = BLEND_FACTORS.get(src, str(src)), BLEND_FACTORS.get(dst, str(dst))
        o = BLEND_OPS[op] if op < len(BLEND_OPS) else str(op)
        if o == '+':
            return f'{s}·S + {d}·D'
        return f'{o}({s}·S, {d}·D)'
    rgb = part(b & 31, b >> 5 & 7, b >> 8 & 31)
    alpha = part(b >> 16 & 31, b >> 21 & 7, b >> 24 & 31)
    return rgb if rgb == alpha else f'{rgb}; α {alpha}'


def target_text(regs, mode):
    out = []
    if mode == 4:
        mask = regs['RB_COLOR_MASK']
        for i, name in enumerate(COLOR_INFO):
            if mask >> 4 * i & 15:
                info = regs[name]
                out.append(f'RT{i} {COLOR_FORMATS.get(info >> 16 & 15, info >> 16 & 15)}'
                           f'@{info & 0xFFF}')
    d = regs['RB_DEPTH_INFO']
    out.append(f'{"D24FS8" if d >> 16 & 1 else "D24S8"}@{d & 0xFFF}')
    return ' · '.join(out)


def draw_state(d):
    regs = d['regs']
    mode = regs['RB_MODECONTROL'] & 7
    cc = regs['RB_COLORCONTROL']
    alpha = (f'alpha {FUNCS[cc & 7]}' if cc & 8 else '') + \
        (' A2C' if cc & 16 else '')
    surface = regs['RB_SURFACE_INFO']
    return {
        'hiz': f'0x{regs["RB_HIZCONTROL"]:X}',
        'hiz_text': hiz_text(regs['RB_HIZCONTROL']),
        'depth': f'0x{regs["RB_DEPTHCONTROL"]:X}',
        'depth_text': depth_text(regs['RB_DEPTHCONTROL'], mode),
        'stencil': f'0x{regs["RB_STENCILREFMASK"]:08X}'
                   if regs['RB_DEPTHCONTROL'] & 1 else '',
        'blend': f'0x{regs["RB_BLENDCONTROL0"]:08X}',
        'blend_text': blend_text(regs['RB_BLENDCONTROL0']),
        'alpha': alpha.strip(),
        'mask': f'0x{regs["RB_COLOR_MASK"]:X}',
        'targets': target_text(regs, mode),
        'msaa': 1 << (surface >> 16 & 3),
        'pitch': surface & 0x3FFF, 'hiz_pitch': surface >> 18,
        'cull': ('back' if regs['PA_SU_SC_MODE_CNTL'] & 2 else '') +
                ('front' if regs['PA_SU_SC_MODE_CNTL'] & 1 else '') or 'none',
    }


def frame_boundaries(path):
    """Draw-command counts at actual swaps, matching play.read_trace's walk.

    Encoded memory/register/gamma records are skipped using their encoded
    lengths, without decompressing them. EVENT kSwap is authoritative; the
    emulator's XE_SWAP packet is a fallback for traces without event records.
    PM4_INTERRUPT is a GPU notification and must not split rendering frames.
    """
    with open(path, 'rb') as source, mmap.mmap(source.fileno(), 0,
                                              access=mmap.ACCESS_READ) as data:
        version = play.TRACE_HEADER.unpack_from(data)[0]
        if version != 1:
            raise play.TraceError(f'{path}: unsupported trace version {version}')
        offset, command, packet = play.TRACE_HEADER.size, 0, None
        events, packets = [], []
        while offset < len(data):
            kind = struct.unpack_from('<I', data, offset)[0]
            if kind == play.PACKET_START:
                count = struct.unpack_from('<I', data, offset + 8)[0]
                packet = (struct.unpack_from('>I', data, offset + 12)[0]
                          if count else None)
                size = 12 + 4 * count
            elif kind == play.PACKET_END:
                if packet is not None and packet >> 30 == 3:
                    op = packet >> 8 & 127
                    if op in (0x22, 0x36):
                        command += 1
                    elif op == 0x64:
                        packets.append(command)
                packet = None
                size = 4
            elif kind in (play.PRIMARY_START, play.INDIRECT_START):
                if kind == play.INDIRECT_START:
                    packet = None
                size = 12 + 4 * struct.unpack_from('<I', data, offset + 8)[0]
            elif kind in (play.PRIMARY_END, play.INDIRECT_END):
                size = 4
            elif kind in (play.MEMORY_READ, play.MEMORY_WRITE):
                size = 20 + struct.unpack_from('<I', data, offset + 12)[0]
            elif kind == play.REGISTERS:
                size = 24 + struct.unpack_from('<I', data, offset + 20)[0]
            elif kind == play.EDRAM_SNAPSHOT:
                size = 12 + struct.unpack_from('<I', data, offset + 8)[0]
            elif kind == play.GAMMA_RAMP:
                size = 16 + struct.unpack_from('<I', data, offset + 12)[0]
            elif kind == play.EVENT:
                if struct.unpack_from('<I', data, offset + 4)[0] == 0:
                    events.append(command)
                size = 8
            else:
                raise play.TraceError(f'unknown record {kind} at {offset}')
            if offset + size > len(data):
                raise play.TraceError(f'truncated record {kind} at {offset}')
            offset += size
    return (events if events else packets), command, (
        'trace swap events' if events else 'XE_SWAP packets' if packets else
        'no swap boundary recorded')


def walk(run):
    walker = Walker()
    _, records = play.read_trace(run / 'gpugen.xtr')
    walker.play(records)
    walker.swaps, commands, walker.frame_boundary_source = frame_boundaries(
        run / 'gpugen.xtr')
    if commands != walker.command:
        raise play.TraceError(f'frame walk counted {commands} commands, '
                              f'render walk counted {walker.command}')
    return walker


# Stage words.

def stage_geometry(stage):
    width = stage['pitch'] & 0x3FFF
    height = stage['pitch'] >> 16 & 0x3FFF
    return width, height


_index_cache = {}


def untile_index(stage):
    """Word indices of a stage's pixels, row by row, and their byte size."""
    key = (stage['pitch'], stage['info'] & 0x1FFF,
           stage.get('storage_height'))
    if key not in _index_cache:
        offsets, bpp = [], None
        for x, y, o, b in play.stage_pixels(stage):
            offsets.append(o)
            bpp = b
        _index_cache[key] = (offsets, bpp)
    return _index_cache[key]


class WideWords(list):
    """Full-width pixels, never truncated to a host uint32."""
    def __init__(self, values, item_bytes):
        super().__init__(values)
        self.item_bytes = item_bytes

    def tobytes(self):
        return b''.join(v.to_bytes(self.item_bytes, 'little') for v in self)


class Results:
    """A results file, mapped, giving each stage's words untiled."""

    def __init__(self, path, meta):
        self.path = Path(path)
        f = open(self.path, 'rb')
        self.map = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        f.close()
        self.tests = {t['stage']['command']: t for t in meta['tests']}
        self.cache = collections.OrderedDict()

    def words(self, command):
        if command in self.cache:
            self.cache.move_to_end(command)
            return self.cache[command]
        test = self.tests[command]
        stage = test['stage']
        start = 0x20 + test['offset']
        raw = self.map[start:start + test['output_bytes']]
        offsets, bpp = untile_index(stage)
        swap = {0: 0, 1: 1, 2: 3, 3: 2}[stage['info'] & 7]
        if bpp == 4:
            raw = raw[:len(raw) // 4 * 4]
            if swap:
                b = bytearray(len(raw))
                for k in range(4):
                    b[k::4] = raw[k ^ swap::4]
                raw = bytes(b)
            a = array('I')
            a.frombytes(raw)
            if sys.byteorder == 'big':
                a.byteswap()
            out = array('I', map(a.__getitem__, [o >> 2 for o in offsets]))
        else:
            values = (int.from_bytes(bytes(
                raw[(o + i) ^ swap] for i in range(bpp)), 'little')
                for o in offsets)
            out = WideWords(values, bpp) if bpp > 4 else array('I', values)
        self.cache[command] = out
        if len(self.cache) > 6:
            self.cache.popitem(last=False)
        return out


def embedded_seed_sha1(path, meta):
    """Hash the measured initial eDRAM block, never an adjacent sidecar."""
    edram = meta.get('edram')
    if not edram or edram.get('output_bytes') != play.EDRAM_BYTES:
        return None
    try:
        with Path(path).open('rb') as f:
            f.seek(0x20 + edram['offset'])
            seed = f.read(edram['output_bytes'])
    except OSError:
        return None
    if len(seed) != play.EDRAM_BYTES:
        return None
    return hashlib.sha1(seed).hexdigest()


class Replay:
    """Backend results, with unknown-state probes only when seed is unverified."""

    def __init__(self, label, directory, meta, console_seed_sha1=None):
        self.label, self.dir = label, Path(directory)
        self.main = Results(self.dir / 'xenia_results.bin', meta)
        try:
            evidence = json.loads((self.dir / 'xenia_provenance.json').read_text())
            claimed_seed = (evidence.get('edram_seed') or {}).get('sha1')
        except (OSError, ValueError, AttributeError):
            claimed_seed = None
        actual_seed = embedded_seed_sha1(self.dir / 'xenia_results.bin', meta)
        self.seed_verified = bool(console_seed_sha1 and
                                  claimed_seed == console_seed_sha1 and
                                  actual_seed == console_seed_sha1)
        self.seed_status = {
            'verified_console_snapshot': self.seed_verified,
            'console_sha1': console_seed_sha1,
            'provenance_sha1': claimed_seed,
            'embedded_replay_sha1': actual_seed,
            'unknown_state_probes_applied': False}
        self.low = self.high = None
        if not self.seed_verified and (self.dir / 'xenia_edram00_results.bin').exists() and \
                (self.dir / 'xenia_edramff_results.bin').exists():
            self.low = Results(self.dir / 'xenia_edram00_results.bin', meta)
            self.high = Results(self.dir / 'xenia_edramff_results.bin', meta)
            self.seed_status['unknown_state_probes_applied'] = True


def slug(text):
    return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')


# Provenance.

_file_digests = {}


def file_digest(path):
    path = Path(path).resolve()
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns,
           stat.st_ino)
    if key not in _file_digests:
        h = hashlib.sha256()
        with path.open('rb') as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b''):
                h.update(chunk)
        _file_digests[key] = h.hexdigest()
    return _file_digests[key]


class ReportCache:
    """Content-addressed derived statistics, images and inspector payloads."""

    def __init__(self, out, enabled=True):
        self.root = out / '.stage_report_cache'
        self.enabled = enabled
        self.hits = self.misses = 0

    def key(self, context):
        return hashlib.sha256(json.dumps(
            context, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

    @staticmethod
    def payload_digest(stats, data):
        # Preserve insertion order across JSON roundtrips, including histogram
        # dictionaries whose integer keys become strings when stored.
        payload = json.dumps({'stats': stats, 'data': data},
                             separators=(',', ':')).encode()
        return hashlib.sha256(payload).hexdigest()

    def load(self, key):
        if self.enabled:
            directory = self.root / key
            try:
                entry = json.loads((directory / 'entry.json').read_text())
                valid = isinstance(entry, dict) and entry.get('key') == key
                stats = entry.get('stats') if valid else None
                data = entry.get('data') if valid else None
                images = entry.get('images') if valid else None
                valid = (valid and isinstance(stats, dict) and
                         isinstance(data, dict) and isinstance(images, dict))
                if valid:
                    valid = (type(stats.get('pixels')) is int and
                             type(stats.get('differing')) is int and
                             0 <= stats['differing'] <= stats['pixels'] and
                             isinstance(data.get('main'), str) and
                             all(name in data and
                                 (data[name] is None or isinstance(data[name], str))
                                 for name in ('low', 'high')))
                if (valid and entry.get('payload_sha256') ==
                        self.payload_digest(stats, data) and all(
                            file_digest(directory / name) == images.get(name)
                            for name in ('image.png', 'delta.png'))):
                    self.hits += 1
                    return entry
            except (OSError, ValueError, KeyError, TypeError):
                pass
        self.misses += 1
        return None

    @staticmethod
    def write(path, data):
        temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
        temporary.write_bytes(data)
        temporary.replace(path)

    def store(self, key, stats, data, image, delta):
        if not self.enabled:
            return
        directory = self.root / key
        directory.mkdir(parents=True, exist_ok=True)
        self.write(directory / 'image.png', image)
        self.write(directory / 'delta.png', delta)
        self.write(directory / 'entry.json', json.dumps(
            {'key': key, 'stats': stats, 'data': data,
             'payload_sha256': self.payload_digest(stats, data),
             'images': {'image.png': hashlib.sha256(image).hexdigest(),
                        'delta.png': hashlib.sha256(delta).hexdigest()}},
            separators=(',', ':')).encode())

    def publish(self, key, images, out):
        for source, target in [('image.png', images['image']),
                               ('delta.png', images['delta'])]:
            shutil.copyfile(self.root / key / source, out / target)


def provenance(directory):
    """What is known of how a replay was made: the recorded provenance
    (run.py's xenia_provenance.json) or, failing that, the identity.txt and
    wrapper script recorded beside it; None for what isn't known."""
    d = Path(directory)
    out = {'source': 'not recorded', 'commit': None, 'dirty': None,
           'binary': None, 'sha256': None, 'cvars': None, 'gpu': None,
           'replayed': None, 'built': None}
    results = d / 'xenia_results.bin'
    if results.exists():
        out['replayed'] = datetime.datetime.fromtimestamp(
            results.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
    recorded = d / 'xenia_provenance.json'
    if recorded.exists():
        p = json.loads(recorded.read_text())
        out['source'] = 'xenia_provenance.json'
        out['binary'] = p.get('trace_dump')
        out['sha256'] = p.get('sha256')
        out['built'] = p.get('mtime')
        out['gpu'] = p.get('gpu')
        args = [a for a in p.get('args', [])]
        wrapper = p.get('wrapper')
        if wrapper:
            m = re.search(r'exec\s+(\S+)\s+"\$@"\s*(.*)', wrapper)
            if m:
                out['binary'] = m.group(1)
                args += m.group(2).split()
        out['cvars'] = ' '.join(args) or '(defaults)'
        out['checkout_at_replay'] = p.get('tree_at_replay')
        out['build_source'] = p.get('build_source')
        tree = p.get('build_source')
        if tree:
            out['commit'] = tree.get('commit')
            out['dirty'] = tree.get('dirty')
            out['committed'] = tree.get('committed')
        out['replayed'] = p.get('replayed') or out['replayed']
        out['patches'] = (tree or {}).get('patches', [])
        out['inputs'] = {name: file_digest(d / name) for name in
                         ('gpugen.xtr', 'gpugen.json', 'console_edram_before.bin')
                         if (d / name).exists()}
        out['result_sha256'] = file_digest(results) if results.exists() else None
        out['run_id'] = p.get('run_id')
        out['parent_run'] = p.get('parent_run')
        return out
    found = []
    for identity in (d / 'identity.txt', d.parent / 'identity.txt'):
        if identity.exists():
            text = identity.read_text()
            m = re.search(r'binary\s+([0-9a-f]{64})', text)
            if m:
                out['sha256'] = m.group(1)
            gpu = text.strip().splitlines()[0] if text.strip() else ''
            if ',' in gpu:
                out['gpu'] = gpu.split(',')[1].strip()
            found.append(identity.name)
            break
    wrapper = d.parent / f'{d.name}.sh'
    if wrapper.exists():
        m = re.search(r'exec\s+(\S+)\s+"\$@"\s*(.*)', wrapper.read_text())
        if m:
            out['binary'] = m.group(1)
            out['cvars'] = m.group(2).strip() or '(defaults)'
            found.append(wrapper.name)
    if found:
        out['source'] = ' + '.join(found)
    return out


# Decoding.

def ordered(v):
    """A float's bits as an integer that orders like the float."""
    return v if v < 0x80000000 else 0x80000000 - v


def as_float(v):
    return struct.unpack('<f', struct.pack('<I', v))[0]


def bin_of(d, bins):
    for k, (lo, hi) in enumerate(bins):
        if lo <= d <= hi:
            return k
    return None


class Kind:
    """How a stage's words read: channels, their names and step units."""

    def __init__(self, depth, fmt, number=0, depth_float=False):
        self.depth, self.fmt, self.number = depth, fmt, number
        self.depth_float = bool(depth and depth_float)
        # Component order follows pixel_formats.xesli packing, low component
        # first for 16/32-bit lanes. Existing packed BGRA previews are retained.
        self.bits, self.count = {2: (8, 1), 10: (8, 2), 24: (16, 1),
            26: (16, 4), 31: (16, 2), 32: (16, 4), 36: (32, 1),
            37: (32, 2), 38: (32, 4)}.get(fmt, (32, 1))
        self.floating = fmt in (31, 32, 36, 37, 38)
        if depth:
            self.channels, self.log = ('depth', 'stencil'), (True, False)
        elif fmt == 6:
            self.channels, self.log = ('R', 'G', 'B', 'A'), (False,) * 4
        elif fmt == 7:
            self.channels, self.log = ('R', 'G', 'B', 'A'), (False,) * 4
        else:
            self.channels = tuple('RGBA'[:self.count])
            self.log = (self.floating,) * self.count

    def components(self, value):
        mask = (1 << self.bits) - 1
        values = tuple((value >> (i * self.bits)) & mask
                       for i in range(self.count))
        if not self.floating and self.number in (1, 3):
            sign = 1 << (self.bits - 1)
            values = tuple(v - (1 << self.bits) if v & sign else v
                           for v in values)
        return values

    def values(self, value):
        raw = self.components(value)
        if self.floating:
            code = '<e' if self.bits == 16 else '<f'
            return tuple(struct.unpack(code, v.to_bytes(self.bits // 8, 'little'))[0]
                         for v in raw)
        divisor = (1 << (self.bits - (self.number == 1))) - 1
        return tuple(v / divisor for v in raw)

    def deltas(self, c, b):
        """Signed per-channel steps of backend word b from console word c."""
        if self.depth:
            return ((b >> 8) - (c >> 8), (b & 255) - (c & 255))
        if self.fmt == 6:
            return ((b >> 16 & 255) - (c >> 16 & 255),
                    (b >> 8 & 255) - (c >> 8 & 255),
                    (b & 255) - (c & 255), (b >> 24) - (c >> 24))
        if self.fmt == 7:
            return ((b >> 20 & 1023) - (c >> 20 & 1023),
                    (b >> 10 & 1023) - (c >> 10 & 1023),
                    (b & 1023) - (c & 1023), (b >> 30) - (c >> 30))
        cv, bv = self.components(c), self.components(b)
        if self.floating:
            sign = 1 << (self.bits - 1)
            order = lambda v: v if v < sign else sign - v
            return tuple(order(y) - order(x) for x, y in zip(cv, bv))
        return tuple(y - x for x, y in zip(cv, bv))

    def depth_value(self, word):
        value = word >> 8
        if not self.depth_float:
            return value / 16777215.0
        exponent, mantissa = value >> 20, value & 0xFFFFF
        return (mantissa * 2.0 ** -34 if not exponent else
                (1.0 + mantissa / 1048576.0) * 2.0 ** (exponent - 15))

    def close(self, c, b):
        d = self.deltas(c, b)
        if self.depth:
            return abs(d[0]) <= 16 and not d[1]
        if self.log[0]:
            return all(abs(x) <= 16 for x in d)
        limit = 12 if self.fmt == 7 else 3
        return all(abs(x) <= (limit if k < 3 else 3) for k, x in enumerate(d))

    def unit(self):
        if self.depth:
            return ('float20e4 encoding steps' if self.depth_float else
                    'steps of 1/16777215')
        if self.fmt == 7:
            return 'steps of 1/1023'
        if self.fmt == 6:
            return 'steps of 1/255'
        return 'ULPs' if self.floating else ('integer steps' if self.number in (2, 3) else f'steps of 1/{(1 << self.bits) - 1}')


# Display images.

def png_bytes(width, height, rgb):
    def chunk(kind, data):
        return (struct.pack('>I', len(data)) + kind + data +
                struct.pack('>I', zlib.crc32(kind + data)))
    stride = 3 * width
    raw = b''.join(b'\0' + rgb[stride * y:stride * (y + 1)]
                   for y in range(height))
    return (b'\x89PNG\r\n\x1a\n' +
            chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0,
                                       0)) +
            chunk(b'IDAT', zlib.compress(raw, 6)) + chunk(b'IEND', b''))


class Shade:
    """Display colour for a stage's words: colour formats as they are,
    depth as grey over the console's 1st-99th percentile (its background
    excluded), floats as log-scaled grey the same way."""

    def __init__(self, kind, console, background):
        self.kind = kind
        sample = console[::max(1, len(console) // 200000)]
        if kind.depth:
            values = sorted(kind.depth_value(v) for v in sample
                            if v != background) or \
                [kind.depth_value(v) for v in sample]
        elif kind.floating and kind.count == 1:
            values = sorted(x for x in (self.flog(kind.values(v)[0]) for v in sample
                                        if v != background) if x is not None)
            values = values or [0.0, 1.0]
        else:
            values = [0, 1]
        self.lo = values[len(values) // 100]
        self.hi = values[min(len(values) - 1, len(values) * 99 // 100)]
        if self.hi <= self.lo:
            self.hi = self.lo + 1

    @staticmethod
    def flog(v):
        f = v
        if f != f or math.isinf(f):
            return None
        return math.copysign(math.log10(1 + abs(f)), f)

    def image(self, words):
        n = len(words)
        out = bytearray(3 * n)
        if self.kind.fmt == 6 and not self.kind.depth:
            b = words.tobytes() if sys.byteorder == 'little' else None
            if b is not None:
                out[0::3] = b[2::4]
                out[1::3] = b[1::4]
                out[2::3] = b[0::4]
                return out
        if self.kind.depth:
            lo, span = self.lo, self.hi - self.lo
            grey = bytes(max(0, min(255, int(
                         (self.kind.depth_value(v) - lo) * 255 / span)))
                         for v in words)
        elif self.kind.fmt == 7:
            out[0::3] = bytes(v >> 22 & 255 for v in words)
            out[1::3] = bytes(v >> 12 & 255 for v in words)
            out[2::3] = bytes(v >> 2 & 255 for v in words)
            return out
        elif self.kind.floating and self.kind.count == 1:
            lo, span = self.lo, self.hi - self.lo
            def g(v):
                x = self.flog(self.kind.values(v)[0])
                return 255 if x is None else max(0, min(255, int(
                    (x - lo) / span * 255)))
            grey = bytes(g(v) for v in words)
        else:
            def byte(v):
                return 255 if not math.isfinite(v) else max(0, min(255, round(v * 255)))
            for i, word in enumerate(words):
                values = self.kind.values(word)
                rgb = values * 3 if len(values) == 1 else (*values, 0)[:3]
                out[3*i:3*i+3] = bytes(byte(v) for v in rgb)
            return out
        out[0::3] = out[1::3] = out[2::3] = grey
        return out


def dim(rgb):
    """The console image as a quiet grey backdrop for the heatmap."""
    n = len(rgb) // 3
    r, g, b = rgb[0::3], rgb[1::3], rgb[2::3]
    lum = bytes(40 + (77 * x + 150 * y + 29 * z >> 8) * 70 // 255
                for x, y, z in zip(r, g, b))
    out = bytearray(3 * n)
    out[0::3] = out[1::3] = out[2::3] = lum
    return out


# Stages: the draws behind each resolve, what they sample, what eDRAM they
# draw over, and the frames.

def build_stages(meta, walker, notes):
    draws = walker.draws
    by_command = {t['stage']['command']: t for t in meta['tests']}
    stages, starts = [], [-1]
    for draw in draws:
        regs = draw['regs']
        if regs['RB_MODECONTROL'] & 7 != 6 or \
                draw['command'] not in by_command:
            continue
        kind, base, fmt, samples = resolve_source(regs)
        window = []
        for start in reversed(starts):
            window = [d for d in draws
                      if start < d['command'] < draw['command'] and
                      writes(d['regs'], kind, base)]
            if window:
                break
        starts.append(draw['command'])
        test = by_command[draw['command']]
        st = test['stage']
        width, height = stage_geometry(st)
        stages.append({
            'command': draw['command'], 'test': test['name'], 'stage': st,
            'kind': kind, 'depth': kind == 'depth', 'base': base,
            'surface_format': fmt, 'samples': samples,
            'pitch': regs['RB_SURFACE_INFO'] & 0x3FFF,
            'width': width, 'height': height,
            'dest_format': st['info'] >> 7 & 63,
            'copy_control': regs['RB_COPY_CONTROL'],
            'regs': regs, 'window': window,
            'signature': (kind, base, fmt, samples, regs['RB_COPY_DEST_BASE'],
                          regs['RB_COPY_DEST_INFO'],
                          regs['RB_COPY_DEST_PITCH'])})
    resolved = []
    for s in stages:
        sampled = set()
        for d in s['window']:
            for address in d['textures']:
                latest = [c for c, b, n in resolved
                          if c < d['command'] and b <= address < b + n]
                if latest:
                    sampled.add(max(latest))
        s['inputs'] = sorted(sampled)
        resolved.append((s['command'], s['stage']['base'], s['stage']['size']))
    # eDRAM a pass draws over: the latest earlier resolve of the same tiles
    # laid out the same way, with no clearing resolve since.
    for i, s in enumerate(stages):
        s['prior'] = None
        for e in reversed(stages[:i]):
            if e['base'] != s['base']:
                continue
            if e['copy_control'] & 0x300:
                break
            if e['pitch'] == s['pitch'] and e['samples'] == s['samples'] and \
                    e['width'] == s['width'] and \
                    play.COPY_BPP[e['dest_format']] == 4 and \
                    play.COPY_BPP[s['dest_format']] == 4:
                s['prior'] = e['command']
            break
    # hi-Z / hi-S written in the trace before or in a pass (bits 0 and 2).
    for s in stages:
        last = s['window'][-1]['command'] if s['window'] else s['command']
        bases = {d['regs']['RB_DEPTH_INFO'] & 0xFFF for d in s['window']}
        s['hiz_tests'] = sum(1 for d in s['window']
                             if d['regs']['RB_HIZCONTROL'] & 2)
        s['his_tests'] = sum(1 for d in s['window']
                             if d['regs']['RB_HIZCONTROL'] & 8)
        # Written by a draw before the first one testing it: a draw that
        # writes and tests at once tests what was there before.
        for bit, test, key in ((1, 2, 'hiz_written'), (4, 8, 'his_written')):
            first = next((d['command'] for d in s['window']
                          if d['regs']['RB_HIZCONTROL'] & test), last)
            writers = [d for d in draws if d['command'] < first and
                       d['regs']['RB_HIZCONTROL'] & bit and
                       d['regs']['RB_DEPTH_INFO'] & 0xFFF in bases]
            s[key] = bool(writers)
            if bit == 1:
                s['hiz_writers'] = writers
        first = next((d['command'] for d in s['window']
                      if d['regs']['RB_HIZCONTROL'] & 10), last)
        s['depth_written'] = any(
            d['regs']['RB_MODECONTROL'] & 7 in (4, 5) and
            d['regs']['RB_DEPTH_INFO'] & 0xFFF in bases and
            (d['regs']['RB_DEPTHCONTROL'] & 6 == 6 or
             d['regs']['RB_DEPTHCONTROL'] & 1)
            for d in draws if d['command'] < first)
        s['depth_tests'] = sum(1 for d in s['window']
                               if d['regs']['RB_DEPTHCONTROL'] & 2)
        s['gamma'] = s['kind'] == 'color' and (
            s['surface_format'] == GAMMA or any(
                d['regs'][name] & 0xFFF == s['base'] and
                d['regs'][name] >> 16 & 15 == GAMMA
                for d in s['window'] for name in COLOR_INFO))
        s['blend'] = s['kind'] == 'color' and any(
            d['regs']['RB_BLENDCONTROL0'] != NO_BLEND for d in s['window'])
        s['alpha_test'] = any(d['regs']['RB_COLORCONTROL'] & 8
                              for d in s['window'])
        s['textured'] = any(d['textures'] for d in s['window'])
        s['average'] = s['samples'] > 1 and \
            s['copy_control'] >> 4 & 7 >= 4
        s['depth_only'] = s['window'] and sum(
            1 for d in s['window'] if d['regs']['RB_MODECONTROL'] & 7 == 5) \
            * 2 > len(s['window'])
        s['derived'] = derived_name(s)
        s['name'] = notes.get('stages', {}).get(str(s['command'])) or \
            s['derived']['name']
    # A swap ends the preceding command range. Interrupt packets may occur
    # many times within that range and are never frame boundaries.
    total = walker.command
    ends = sorted({c - 1 for c in walker.swaps if 0 < c <= total})
    labels = [f['label'] for f in notes.get('frames', [])]
    frames, start = [], 0
    for k, end in enumerate(ends + [total - 1]):
        if start > total - 1:
            break
        cut = k == len(ends)
        frames.append({'start': start, 'end': end, 'cut': cut})
        start = end + 1
    if frames and frames[-1]['cut'] and frames[-1]['start'] > \
            max(s['command'] for s in stages):
        frames.pop()
    for k, f in enumerate(frames):
        default = f'Frame {k}' if not f['cut'] else f'Frame {k}'
        label = labels[k] if len(labels) == len(frames) else default
        # Drop a stale ", cut off": the range says it.
        f['label'] = re.sub(r',?\s*cut off', '', label)
        f['range'] = (f'commands {f["start"]}–{f["end"]}' if not f['cut']
                      else f'from command {f["start"]}; the trace ends at '
                      f'command {f["end"]}')
    for s in stages:
        s['frame'] = next(k for k, f in enumerate(frames)
                          if f['start'] <= s['command'] <= f['end'])
    return stages, frames, total


def derived_name(s):
    w, h = s['width'], s['height']
    msaa = f' · {s["samples"]}× MSAA' if s['samples'] > 1 else ''
    n = len(s['window'])
    if s['depth']:
        fmt = 'D24FS8' if s['surface_format'] else 'D24S8'
        name = f'Depth {w}×{h}' + (', depth-only pass' if s['depth_only']
                                   else '')
    else:
        fmt = COLOR_FORMATS.get(s['surface_format'],
                                f'format {s["surface_format"]}')
        rt = s['copy_control'] & 7
        name = (f'{"Gamma colour" if s["gamma"] else "Colour"} {w}×{h}' +
                (f', RT{rt}' if rt else '') +
                (', blended' if s['blend'] else ''))
    return {'name': name,
            'detail': f'{fmt} · {w}×{h}{msaa} · {n} draw{"s" if n != 1 else ""}'
                      f' · eDRAM tile {s["base"]}'}


def draw_groups(window):
    """Runs of consecutive draws with the same key state."""
    groups = []
    for d in window:
        st = draw_state(d)
        key = tuple(sorted(st.items()))
        prim = PRIMS.get(d['prim'], f'prim {d["prim"]}')
        if groups and groups[-1]['key'] == key:
            g = groups[-1]
            g['last'] = d['command']
            g['draws'] += 1
            g['vertices'] += d['count']
            g['prims'][prim] += 1
            g['textured'] += bool(d['textures'])
        else:
            groups.append({'key': key, 'first': d['command'],
                           'last': d['command'], 'draws': 1,
                           'vertices': d['count'], 'state': st,
                           'prims': collections.Counter({prim: 1}),
                           'textured': int(bool(d['textures']))})
    for g in groups:
        del g['key']
        g['prims'] = ', '.join(f'{k}' if v == g['draws'] else f'{v} {k}'
                               for k, v in g['prims'].most_common())
    return groups


# Coverage areas.

def load_areas():
    areas = {}
    for f in sorted((COVERAGE / 'areas').glob('*.json')):
        try:
            a = json.loads(f.read_text())
        except ValueError:
            continue
        areas[a.get('id', f.stem)] = a
    return areas


def metro_mentions(areas):
    """{command: [area ids]} from the areas' metro notes."""
    out = collections.defaultdict(list)
    for aid, a in areas.items():
        for note in a.get('metro', []):
            head = note.split(':')[0]
            for n in re.findall(r'\b\d{2,4}\b', head):
                out[int(n)].append(aid)
    return out


# Comparing one backend's stage with the console's.

def block_size(samples):
    return {1: (4, 4), 2: (4, 2), 4: (2, 2)}.get(samples, (4, 4))


def background(console, backend):
    """The console's commonest word when it is a background both share: at
    least 5% of the pixels, and the backend keeps it on 30% of them."""
    value, n = collections.Counter(console[::7]).most_common(1)[0]
    n = sum(1 for v in console if v == value)
    if n * 20 < len(console):
        return None
    kept = sum(1 for c, b in zip(console, backend) if c == value and b == value)
    return value if kept * 10 >= n * 3 else None


def bands(words, width):
    """For a target holding a few words, each word's rows."""
    rows = collections.defaultdict(set)
    for i, v in enumerate(words):
        rows[v].add(i // width)
    out = []
    for v, ys in sorted(rows.items(), key=lambda kv: min(kv[1])):
        ys = sorted(ys)
        runs, a = [], ys[0]
        for p, q in zip(ys, ys[1:] + [None]):
            if q != p + 1:
                runs.append(f'{a}–{p}' if a != p else f'{a}')
                if q is not None:
                    a = q
        out.append({'word': f'{v:08X}', 'rows': ', '.join(runs[:3]) +
                    (' …' if len(runs) > 3 else '')})
    return out


def compare(s, kind, C, B, low, high, prior_c, prior_b, variant, other,
            played_mode, disjoint, bg, primary):
    """One backend's stage against the console's: observation, attribution
    counts, per-pixel heatmap codes."""
    n = len(C)
    width = s['width']
    diff = [i for i, (c, b) in enumerate(zip(C, B)) if c != b]
    nch = len(kind.channels)
    bins = [LOG_BINS if kind.log[k] else STEP_BINS for k in range(nch)]
    hist = [[0] * len(b) for b in bins]
    obs = collections.Counter()
    why = collections.Counter()
    code = {}            # pixel -> (category, sign, level)
    extra_px = []
    extra_words = collections.Counter()
    high_ch = [0] * nch
    low_ch = [0] * nch
    small = 0            # value pixels whose every channel is within 1
    worst_steps = []
    stencil_only = depth_far = 0
    box = None
    lp = len(prior_c) if prior_c is not None else 0
    for i in diff:
        c, b = C[i], B[i]
        x, y = i % width, i // width
        box = [x, y, x, y] if box is None else [
            min(box[0], x), min(box[1], y), max(box[2], x), max(box[3], y)]
        if low is not None and low[i] != high[i]:
            obs['pretrace'] += 1
            why['pretrace'] += 1
            code[i] = (5, 0, 3)
            continue
        ck = disjoint or (i < lp and c == prior_c[i]) or \
            (bg is not None and c == bg)
        bk = (i < lp and b == prior_b[i]) or (bg is not None and b == bg)
        # A small step from the earlier word is a value difference (an
        # additive blend of almost nothing), not a pixel left alone.
        if ck != bk and kind.close(c, b):
            ck = bk = False
        if ck and bk and i < lp:
            cat = 'edram'
        elif ck and not bk:
            cat = 'extra'
        elif bk and not ck:
            cat = 'missing'
        else:
            cat = 'value'
        obs[cat] += 1
        if played_mode and other is not None and other[i] == c:
            why['inherited'] += 1
        elif variant is not None and variant[i] == c:
            why['variant'] += 1
        elif cat == 'value':
            why['value'] += 1
        elif cat == 'extra':
            extra_px.append(i)
        else:
            why[cat] += 1
        if cat == 'extra':
            extra_words[b] += 1
            code[i] = (2, 0, 3)
            continue
        if cat == 'missing':
            code[i] = (3, 0, 3)
            continue
        if cat == 'edram':
            code[i] = (4, 0, 3)
            continue
        d = kind.deltas(c, b)
        worst, sign = 0, 0
        all_small = True
        for k in range(nch):
            dk = d[k]
            if not dk:
                continue
            bk_ = bin_of(dk, bins[k])
            if bk_ is not None:
                hist[k][bk_] += 1
            if dk > 0:
                high_ch[k] += 1
            else:
                low_ch[k] += 1
            mag = abs(dk)
            if kind.depth and k == 1:
                mag = 0
            if mag > 1 or (kind.depth and k == 1):
                all_small = False
            if mag > worst or (worst == 0 and not sign):
                worst, sign = mag, 1 if dk > 0 else -1
        if kind.depth:
            if d[0] == 0:
                stencil_only += 1
            elif abs(d[0]) >= DEPTH_FAR:
                depth_far += 1
        small += all_small
        worst_steps.append(worst)
        if kind.fmt == 7 and not kind.depth:
            worst >>= 2
        if kind.log[0]:
            level = 0 if worst <= 1 else 1 if worst < 16 else \
                2 if worst < 256 else 3
        else:
            level = 0 if worst <= 1 else 1 if worst <= 3 else \
                2 if worst < 16 else 3
        code[i] = (1, sign, level)
    # Of the pixels Xenia drew where the console left the target, those in
    # whole blocks (hi-Z and hi-S cull blocks of 4x4 samples).
    bw, bh = block_size(s['samples'])
    blocks = collections.Counter((i % width // bw, i // width // bh)
                                 for i in extra_px)
    whole = {k for k, v in blocks.items() if v == bw * bh}
    in_whole = sum(1 for i in extra_px
                   if (i % width // bw, i // width // bh) in whole)
    prior_differs = sum(1 for i in extra_px if i < lp and
                        prior_b[i] != prior_c[i]) if lp else None
    why['extra_blocks'] = in_whole
    why['extra_loose'] = len(extra_px) - in_whole
    worst_steps.sort()
    ch_total = [h + l for h, l in zip(high_ch, low_ch)]
    out = {
        'pixels': n, 'differing': len(diff),
        'observation': dict(obs), 'attribution': dict(why),
        'histogram': hist,
        'high': high_ch, 'low': low_ch, 'channel_pixels': ch_total,
        'small': small, 'value_pixels': obs['value'],
        'median': worst_steps[len(worst_steps) // 2] if worst_steps else 0,
        'p99': worst_steps[min(len(worst_steps) - 1,
                               len(worst_steps) * 99 // 100)]
        if worst_steps else 0,
        'stencil_only': stencil_only, 'depth_far': depth_far,
        'extra_blocks': len(whole), 'extra_block_share':
            in_whole / len(extra_px) if extra_px else 0,
        'extra_word': [f'{w:08X}' for w, _ in extra_words.most_common(1)] +
            [round(extra_words.most_common(1)[0][1] / len(extra_px), 3)]
            if extra_px else None,
        'extra_prior_differs': prior_differs,
        'box': box,
        'local': box is not None and
            (box[2] - box[0] + 1) * (box[3] - box[1] + 1) * 20 < n,
    }
    if variant is not None:
        out['variant'] = {
            'fixes': why['variant'],
            'breaks': sum(1 for c, b, v in zip(C, B, variant)
                          if c == b and v != c),
            'differing': sum(1 for c, v in zip(C, variant) if c != v)}
    if other is not None:
        out['other_differing'] = sum(1 for c, o in zip(C, other) if c != o)
    if primary is not None:
        out['match_primary'] = sum(1 for b, p in zip(B, primary) if b == p)
    return out, code


def heatmap(base, code):
    out = bytearray(base)
    alpha = (0.5, 0.68, 0.84, 1.0)
    for i, (cat, sign, level) in code.items():
        if cat == 1:
            col = HIGH if sign > 0 else LOW
        else:
            col = {2: EXTRA, 3: MISSING, 4: EDRAM, 5: PRETRACE}[cat]
        a = alpha[level]
        o = 3 * i
        out[o] = int(out[o] * (1 - a) + col[0] * a)
        out[o + 1] = int(out[o + 1] * (1 - a) + col[1] * a)
        out[o + 2] = int(out[o + 2] * (1 - a) + col[2] * a)
    return out


def packed(words):
    """Words deflated, byte planes first when that is smaller (depth and
    float words), as base64; 's:' marks the byte-plane order."""
    raw = words.tobytes()
    if isinstance(words, WideWords):
        return f'w{words.item_bytes}:' + base64.b64encode(zlib.compress(raw, 6)).decode()
    plain = zlib.compress(raw, 6)
    planes = zlib.compress(b''.join(raw[k::4] for k in range(4)), 6)
    if len(planes) < len(plain) * 0.9:
        return 's:' + base64.b64encode(planes).decode()
    return base64.b64encode(plain).decode()


def xor(a, b):
    values = (x ^ y for x, y in zip(a, b))
    if isinstance(a, WideWords):
        return WideWords(values, a.item_bytes)
    return array('I', values)


# Why: hypotheses with confidence.

def pct(n, d, digits=0):
    if not d:
        return '0%'
    v = 100 * n / d
    if digits == 0 and 0 < v < 1:
        return '<1%'
    if digits == 0 and 99 < v < 100:
        return '>99%'
    return f'{v:.{digits}f}%'


def area_ref(areas, aid):
    a = areas.get(aid)
    if not a:
        return None
    return {'id': aid, 'title': a.get('title', aid), 'status': a.get('status'),
            'rule': a.get('rule', '')}


def area_next(areas, aid):
    a = areas.get(aid)
    if not a:
        return None
    for t in a.get('next_tests', []):
        if t.get('status') in ('planned', 'running'):
            owner = t.get('owner') or ''
            return (f'{t.get("suite")}: {t.get("probes")} ({t["status"]}'
                    f'{", " + owner if owner and owner != "unassigned" else ""})')
    return None


def direction(st, kind):
    """(share, 'high'|'low') of the wrong channels' commonest sign, over the
    value differences."""
    channels = range(1 if kind.depth else len(kind.channels))
    h = sum(st['high'][k] for k in channels)
    lo = sum(st['low'][k] for k in channels)
    if not h + lo:
        return 0, None
    return (h / (h + lo), 'high') if h >= lo else (lo / (h + lo), 'low')


def within(st, kind, steps):
    """Share of wrong channel values within ±steps."""
    tot = inside = 0
    for k, hist in enumerate(st['histogram']):
        if kind.depth and k == 1:
            continue
        bins = LOG_BINS if kind.log[k] else STEP_BINS
        for (lo, hi), v in zip(bins, hist):
            tot += v
            if -steps <= lo and hi <= steps:
                inside += v
    return inside / tot if tot else 0


def describe_values(st, kind):
    if not st['value_pixels']:
        return ''
    share, way = direction(st, kind)
    if kind.depth:
        far = 'numerically higher' if way == 'high' else 'numerically lower'
        text = (f'depth off a median {st["median"]:,} {kind.unit()}, '
                f'Xenia {far} on {pct(share * 100, 100)}')
        extra = []
        if st['depth_far']:
            extra.append(f'{st["depth_far"]:,} px exceed '
                         f'{DEPTH_FAR:,} encoded depth steps')
        if st['stencil_only']:
            extra.append(f'{st["stencil_only"]:,} px stencil only')
        return text + (f' ({"; ".join(extra)})' if extra else '')
    if kind.log[0]:
        return (f'values off a median {st["median"]:,} {kind.unit()}, '
                f'Xenia {way} on {pct(share * 100, 100)}')
    one = within(st, kind, 1)
    unit = 'step' if kind.fmt != 7 else 'step of 1/1023'
    if one >= 0.995 and share >= 0.995:
        return f'every wrong channel is one {unit} {way}'
    if one >= 0.95 and share >= 0.75:
        return f'{pct(one * 100, 100)} of wrong channels one {unit} off, ' \
               f'{pct(share * 100, 100)} {way}'
    three = within(st, kind, 3)
    even = share < 0.6
    return (f'{pct(three * 100, 100)} within ±3 steps (median '
            f'{st["median"]}), ' +
            ('evenly up and down' if even else
             f'Xenia {way} on {pct(share * 100, 100)}'))


def observe(s, kind, st, names):
    """What differs, in a sentence."""
    d = st['differing']
    if not d:
        return 'Identical to the console.'
    o = st['observation']
    parts = []
    if o.get('pretrace', 0) * 2 > d:
        return (f'Xenia shows eDRAM from before the trace on '
                f'{pct(o["pretrace"], d)} of the wrong pixels (its eDRAM-00 '
                f'and -FF replays disagree there).')
    if st.get('disjoint'):
        b = st['console_bands']
        where = '; '.join(f'{x["word"]} rows {x["rows"]}' for x in b[:3])
        word = st['extra_word'][0] if st['extra_word'] else '?'
        return (f'The console left the target as it was ({where}); no '
                f'console word is one Xenia wrote. Xenia wrote {word} on '
                f'{pct(st["extra_word"][1] * 100, 100) if st["extra_word"] else "?"}'
                f' of it.')
    if o.get('extra'):
        what = f"{names.get(s['prior'], 'its earlier')}'s words" \
            if s['prior'] is not None else \
            f'its background word {st["background"]}'
        t = (f'the console leaves {pct(o["extra"], d)} of the wrong pixels '
             f'holding {what} where Xenia draws')
        if st['extra_block_share'] >= 0.5:
            bw, bh = block_size(s['samples'])
            t += f' ({pct(st["extra_block_share"] * 100, 100)} in whole ' \
                 f'{bw}×{bh} blocks)'
        if st['extra_word'] and st['extra_word'][1] >= 0.5:
            t += f'; Xenia writes {st["extra_word"][0]} on ' \
                 f'{pct(st["extra_word"][1] * 100, 100)} of them'
        parts.append(t)
    if o.get('edram'):
        parts.append(f'{pct(o["edram"], d)} keep '
                     f'{names.get(s["prior"], "an earlier pass")}\'s words on '
                     f'both sides')
    if o.get('missing'):
        parts.append(f'Xenia leaves {pct(o["missing"], d)} the console draws')
    v = describe_values(st, kind)
    if v:
        parts.append(('the rest: ' if parts else '') + v)
    if o.get('pretrace'):
        parts.append(f'{pct(o["pretrace"], d)} show eDRAM from before the '
                     'trace')
    text = '; '.join(parts)
    if st['local']:
        b = st['box']
        text += f'. All inside a {b[2] - b[0] + 1}×{b[3] - b[1] + 1} box'
    return text[:1].upper() + text[1:] + '.'


def writers_text(s):
    """The latest run of draws that wrote hi-Z before a pass tested it, and
    through what surface."""
    ws = s.get('hiz_writers') or []
    if not ws:
        return 'no traced draw'
    last = ws[-1]
    run = [last['command']]
    for d in reversed(ws[:-1]):
        if d['command'] != run[-1] - 1:
            break
        run.append(d['command'])
    surface = last['regs']['RB_SURFACE_INFO']
    msaa = 1 << (surface >> 16 & 3)
    here = s['window'][0]['regs']['RB_SURFACE_INFO'] if s['window'] else 0
    same = surface & 0x3FFFF == here & 0x3FFFF
    cmds = f'draw {run[0]}' if len(run) == 1 else \
        f'draws {run[-1]}–{run[0]}'
    return (f'{cmds} through ' + ('the same surface' if same else
            f'a {msaa}× surface of pitch {surface & 0x3FFF}'))


def brief(s, kind, st, names):
    """What differs, in a few words, for the overview."""
    d = st['differing']
    if not d:
        return 'Identical.'
    o = st['observation']
    if o.get('pretrace', 0) * 2 > d:
        return 'Shows eDRAM from before the trace.'
    if st.get('disjoint'):
        word = st['extra_word'][0] if st['extra_word'] else '?'
        return f'Console drew nothing; Xenia filled it with {word}.'
    parts = []
    if o.get('extra', 0) * 50 >= d:
        what = f'{names.get(s["prior"], s["prior"])}\'s words' \
            if s['prior'] is not None else st['background'] or 'background'
        t = f'{pct(o["extra"], d)} left holding {what} where Xenia draws'
        if st['extra_block_share'] >= 0.9:
            t += ', in whole blocks'
        parts.append(t)
    if o.get('edram', 0) * 50 >= d:
        parts.append(f'{pct(o["edram"], d)} keep '
                     f'{names.get(s["prior"], s["prior"])}\'s eDRAM')
    if o.get('missing', 0) * 50 >= d:
        parts.append(f'{pct(o["missing"], d)} drawn only by the console')
    if st['value_pixels']:
        share, way = direction(st, kind)
        if kind.depth:
            v = (f'depth off a median {st["median"]:,} {kind.unit()}, Xenia '
                 f'{"numerically higher" if way == "high" else "numerically lower"} on '
                 f'{pct(share * 100, 100)}')
        elif kind.log[0]:
            v = (f'values off a median {st["median"]:,} ULPs, '
                 f'{pct(share * 100, 100)} {way}')
        else:
            one = within(st, kind, 1)
            if one >= 0.995 and share >= 0.995:
                v = f'every wrong channel one step {way}'
            elif one >= 0.95:
                v = f'one step off, {pct(share * 100, 100)} {way}'
            elif share < 0.6:
                v = (f'{pct(within(st, kind, 3) * 100, 100)} within ±3 '
                     'steps, evenly up and down')
            else:
                v = (f'{pct(within(st, kind, 3) * 100, 100)} within ±3 '
                     f'steps, {pct(share * 100, 100)} {way}')
        parts.append(('the rest: ' if parts else '') + v)
    text = '; '.join(parts)
    if st['local']:
        b = st['box']
        text += f', inside a {b[2] - b[0] + 1}×{b[3] - b[1] + 1} box'
    return text[:1].upper() + text[1:] + '.'


def hypotheses(s, kind, st, mode, variant, areas, mentions, names):
    d = st['differing']
    if not d:
        return []
    a = st['attribution']
    out = []
    N = len(s['window'])
    bw, bh = block_size(s['samples'])

    def test_by(k, what):
        if N == 1:
            return f'The pass\'s one draw tests {what}'
        return f'{k} of {N} draws test {what}'

    def per_draw(after=''):
        if N == 1:
            return ('The pass is one draw: console replays of it changing '
                    'one suspect state at a time.')
        return f'One console stage per draw of the {N} draws{after}.'

    def add(key, n, conf, title, text, evidence='', area=None, test=None):
        if n:
            out.append({'key': key, 'pixels': n, 'share': n / d,
                        'confidence': conf, 'title': title, 'text': text,
                        'evidence': evidence, 'area': area_ref(areas, area),
                        'next': test or area_next(areas, area) or ''})

    hiz_values = sorted({f'0x{x["regs"]["RB_HIZCONTROL"]:X}'
                         for x in s['window']})
    add('pretrace', a.get('pretrace', 0), 'na',
        'Shows eDRAM from before the trace',
        'Xenia\'s eDRAM-00 and -FF replays disagree on these pixels: the '
        'pass leaves or blends onto eDRAM no traced draw wrote, so neither '
        'side\'s value is a property of the trace.')
    inputs = ', '.join(f'{names.get(c, c)} ({c})' for c in s['inputs'])
    add('inherited', a.get('inherited', 0), 'proven',
        'Inherited from earlier passes',
        'The isolated replay, which gives this pass the console\'s own '
        'output of every earlier pass, gets these pixels right.',
        f'Samples {inputs}.' if inputs else '',
        test=f'Fix what it samples: {inputs}.' if inputs else '')
    if variant and a.get('variant'):
        v = st['variant']
        add('variant', a['variant'], 'proven', variant['name'],
            f'The {variant["label"]} replay ({variant["change"]}) puts these '
            'pixels right.',
            f'It also breaks {v["breaks"]:,} that were right; '
            f'{v["differing"]:,} still differ with it.',
            variant.get('area'), test=f'Adopt {variant["change"]}.')
    prior = names.get(s['prior'], s['prior'])
    add('edram', a.get('edram', 0), 'proven',
        f'Inherited through eDRAM from {prior}',
        f'Neither side drew these pixels: both kept {prior}\'s words '
        f'({s["prior"]}), so the difference is {prior}\'s. Isolation '
        'restores memory, not eDRAM.', test=f'Fix {prior} first.')
    blocks, loose = a.get('extra_blocks', 0), a.get('extra_loose', 0)
    shares = f'{pct(blocks, blocks + loose)} of them in whole {bw}×{bh} blocks'
    culled = False
    if st.get('disjoint') and (s['hiz_tests'] or s['his_tests']):
        pass
    elif s['hiz_tests'] and s['hiz_written']:
        add('hiz', blocks, 'likely', 'Hi-Z culling',
            'The console leaves these pixels where Xenia draws, in whole '
            f'{bw}×{bh}-sample blocks as hi-Z culls them; Xenia has no hi-Z.',
            test_by(s['hiz_tests'], 'hi-Z') + ' (RB_HIZCONTROL '
            f'{", ".join(hiz_values)}); {shares}.', 'hiz-depth',
            test=area_next(areas, 'hiz-depth') or
            'Console replay with RB_HIZCONTROL forced to 0.')
        culled = True
    elif s['his_tests'] and s['his_written']:
        add('his', blocks, 'likely', 'Hi-stencil culling',
            'The console leaves these pixels where Xenia draws, in whole '
            f'{bw}×{bh}-sample blocks as hi-stencil culls them; Xenia has '
            'no hi-stencil.',
            test_by(s['his_tests'], 'hi-S') + ' (RB_HIZCONTROL '
            f'{", ".join(hiz_values)}); {shares}.', 'hiz-stencil',
            test=area_next(areas, 'hiz-stencil') or
            'Console replay with the hi-stencil test off.')
        culled = True
    skipped = loose + (0 if culled else blocks)
    if st.get('disjoint'):
        why = ('The console\'s draws wrote no pixel: the target keeps words '
               'no Xenia draw produced. ')
        if s['hiz_tests']:
            why += (test_by(s['hiz_tests'], 'hi-Z') +
                    f' (RB_HIZCONTROL {", ".join(hiz_values)})' +
                    (f', last written by {writers_text(s)}'
                     if s['hiz_written'] else
                     ', which no traced draw wrote first') +
                    (', and the depth test, against depth no traced draw '
                     'wrote' if s['depth_tests'] and not s['depth_written']
                     else '') +
                    ': either could reject the whole draw.')
    else:
        why = 'The console\'s draws left these pixels; Xenia\'s wrote them. '
    if st.get('disjoint') and s['hiz_tests']:
        pass
    elif not s['hiz_tests'] and not s['his_tests']:
        why += ('Not hierarchical culling: ' +
                ('the pass\'s one draw tests neither hi-Z nor hi-S'
                 if N == 1 else f'none of the {N} draws tests hi-Z or hi-S') +
                f' (RB_HIZCONTROL {", ".join(hiz_values)}).')
    elif not (s['hiz_written'] or s['his_written']):
        why += (test_by(s['hiz_tests'] or s['his_tests'],
                        'hi-Z' if s['hiz_tests'] else 'hi-S') +
                ', but no traced draw '
                'wrote it first, so the console\'s buffer holds what it had '
                'before the trace.')
    elif culled:
        why += 'These are not whole blocks, so not hierarchical culling alone.'
    ev = ''
    if st['extra_word'] and st['extra_word'][1] >= 0.3:
        ev = (f'Xenia writes {st["extra_word"][0]} on '
              f'{pct(st["extra_word"][1] * 100, 100)} of them.')
    if not culled and blocks + loose:
        ev = (ev + ' ' if ev else '') + shares[:1].upper() + shares[1:] + '.'
    if st.get('extra_prior_differs') is not None and blocks + loose:
        ev += (f' {prior} ({s["prior"]}) already differed on '
               f'{pct(st["extra_prior_differs"], blocks + loose)} of them.')
    whole_draw = st.get('disjoint') and s['hiz_tests']
    add('skipped', skipped, 'unknown', 'Console skipped pixels Xenia drew',
        why, ev, 'hiz-depth' if whole_draw else None,
        test='Console replay with RB_HIZCONTROL forced to 0, then with the '
        'depth cleared first.' if whole_draw else
        per_draw(' (a resolve after each)'))
    add('missing', a.get('missing', 0), 'unknown',
        'Xenia skipped pixels the console drew',
        'The console drew these pixels; Xenia left the target\'s earlier '
        'words there.', test=per_draw())
    n = a.get('value', 0)
    share, way = direction(st, kind)
    values = describe_values(st, kind)
    values = values[:1].upper() + values[1:] + '.' if values else ''
    suspects = [x for x in mentions.get(s['command'], [])
                if x not in ('hiz-depth', 'hiz-stencil')]
    if kind.depth:
        add('depth', n, 'unknown', 'Depth values', values +
            ' Coverage agrees here; the console\'s depth interpolation is not '
            'written down yet.', '', 'depth-interpolation')
    elif s['gamma'] and s['samples'] > 1:
        add('gamma-msaa', n, 'unknown', 'Gamma MSAA resolve', values +
            f' The target is 8_8_8_8_GAMMA at {s["samples"]}×; how the '
            'console averages gamma samples is not known yet.', '',
            'gamma-msaa')
    elif s['gamma'] and within(st, kind, 1) >= 0.95 and share >= 0.75:
        add('gamma', n, 'likely', 'Gamma target rounding', values +
            ' The target is 8_8_8_8_GAMMA, whose console rule Xenia does '
            'not follow: a truncating piecewise-linear encode and blending in '
            '10-bit linear.',
            'A one-code bias in one direction is what that rule changes.',
            'gamma-blend-1x', test='Replay with Xenia\'s gamma path following '
            'the gamma-blend-1x rule.')
    else:
        feats = []
        if s['gamma']:
            feats.append(('gamma target', 'gamma-blend-1x'))
        if s['blend']:
            feats.append(('blending', 'blend'))
        if s['textured']:
            feats.append(('texture sampling', 'texture-filtering'))
        if s['alpha_test']:
            feats.append(('alpha test', 'alpha-test-a2c-mrt'))
        if s['average']:
            feats.append(('MSAA averaging', 'msaa-resolve'))
        if kind.fmt in (6, 7) and not kind.depth:
            feats.append(('float-to-unorm write', 'color-write-rounding'))
        sus = [{'label': lab, 'area': area_ref(areas, aid)} for lab, aid in
               feats if aid in areas]
        area = suspects[0] if suspects else None
        add('values', n, 'unknown',
            'Float values' if kind.log[0] else 'Colour values', values,
            'Its memory inputs are the console\'s (isolated replay).'
            if mode == 'isolated' and s['inputs'] else '', area,
            test=area_next(areas, area) if area else
            per_draw())
        if out and out[-1]['key'] == 'values':
            out[-1]['suspects'] = sus
    order = {'proven': 0, 'likely': 1, 'unknown': 2, 'na': 3}
    out.sort(key=lambda h: (-h['pixels'], order[h['confidence']]))
    return out


def agreement(entries, primary):
    """Whether the backends tell the same story for a stage."""
    p = entries[primary]
    def story(e):
        st = e['stats']
        top = [h['key'] for h in e['hypotheses'] if h['key'] != 'variant']
        return (st['differing'] / st['pixels'], top[:1], e['direction'])
    ps = story(p)
    differ = []
    for label, e in entries.items():
        if label == primary:
            continue
        s = story(e)
        far = abs(s[0] - ps[0]) > max(0.02, 0.1 * ps[0])
        other_top = s[1] != ps[1]
        flips = s[2][1] and ps[2][1] and s[2][1] != ps[2][1] and \
            s[2][0] > 0.6 and ps[2][0] > 0.6
        if far or other_top or flips:
            bits = [f'{pct(s[0] * 100, 100)} wrong']
            if s[2][1]:
                bits.append(f'{pct(s[2][0] * 100, 100)} {s[2][1]}')
            differ.append({'label': label, 'text': ', '.join(bits)})
    head = [f'{pct(ps[0] * 100, 100)} wrong'] + \
        ([f'{pct(ps[2][0] * 100, 100)} {ps[2][1]}'] if ps[2][1] else [])
    return {'agree': not differ, 'differ': differ,
            'primary': ', '.join(head)}


# The report.

def relative(path, start):
    return os.path.relpath(path, start)


def build(run, config, mode, walker, meta, notes, areas, only=None,
          use_cache=True):
    spec = config['modes'][mode]
    other_mode = next((m for m in config['modes'] if m != mode), None)
    played_mode = mode == 'played'
    out = (run / spec['out']).resolve()
    if out.name == 'diff' or (out / 'report.html').exists() and \
            not (out / '.stage_report').exists():
        raise SystemExit(f'{out}: refusing to overwrite a report this '
                         'script did not write')
    (out / 'img').mkdir(parents=True, exist_ok=True)
    (out / 'data').mkdir(exist_ok=True)
    (out / '.stage_report').write_text('written by stage_report.py\n')
    log(f'Report mode: {mode} | output: {out}')
    cache = ReportCache(out, use_cache)
    build_started = time.monotonic()

    console_seed_sha1 = embedded_seed_sha1(run / 'console_results.bin', meta)
    backends = [Replay(label, run / d, meta, console_seed_sha1)
                for label, d in spec['backends']]
    primary = config.get('primary') or backends[0].label
    variants = []
    labels = {b.label for b in backends}
    for v in spec.get('variants', []):
        log(f'Checking identical replay inputs: {v["label"]}')
        if v['label'] in labels:
            raise ValueError('Replay labels must be unique: ' + v['label'])
        labels.add(v['label'])
        base = next((b for b in backends if b.label == v['of']), None)
        if base is None:
            raise ValueError('Unknown replay backend: ' + v['of'])
        for filename in ('gpugen.xtr', 'gpugen.json', 'console_edram_before.bin'):
            left, right = base.dir / filename, (run / v['dir']) / filename
            if not left.exists() or not right.exists():
                raise ValueError('Missing comparison input: ' + filename)
            if file_digest(left) != file_digest(right):
                raise ValueError('Incompatible replay input for ' + v['label'] + ': ' + filename)
        variants.append({**v, 'replay': Replay(v['label'], run / v['dir'],
                                               meta, console_seed_sha1)})
    others = {}
    if other_mode:
        o = config['modes'][other_mode]
        for label, d in o['backends']:
            if (run / d / 'xenia_results.bin').exists():
                others[label] = Results(run / d / 'xenia_results.bin', meta)
        for v in o.get('variants', []):
            if (run / v['dir'] / 'xenia_results.bin').exists():
                others[v['label']] = Results(run / v['dir'] /
                                             'xenia_results.bin', meta)
    console = Results(run / 'console_results.bin', meta)
    stages, frames, total = build_stages(meta, walker, notes)
    names = {s['command']: s['name'] for s in stages}
    mentions = metro_mentions(areas)
    columns = [(b.label, b, None) for b in backends] + \
        [(v['label'], v['replay'], v) for v in variants]

    def result_digest(result):
        return file_digest(result.path) if result else None

    primary_result = next(b for b in backends if b.label == primary).main
    common_key = {
        'tool': file_digest(__file__), 'play': file_digest(play.__file__),
        'decoder_dependencies': {name: file_digest(path) for name, path in (
            ('pm4', pm4.__file__), ('gpugen', HERE / 'gpugen.py'),
            ('xenos', HERE / 'xenos.py'),
            ('registers', Path(pm4.REPO) / 'src/xenia/gpu/register_table.inc'))},
        'trace': file_digest(run / 'gpugen.xtr'),
        'metadata': file_digest(run / 'gpugen.json'),
        'console': file_digest(run / 'console_results.bin'),
        'primary': result_digest(primary_result), 'mode': mode, 'notes': notes}
    selected_stages = [s for s in stages if not only or s['command'] in only]
    log(f'{len(selected_stages)} measured commands, {len(columns)} recorded runs; '
        'reading saved results only (no console/GPU execution)')
    entries = []
    for stage_number, s in enumerate(selected_stages, 1):
        if only and s['command'] not in only:
            continue
        t0 = time.time()
        c = s['command']
        log(f'[{stage_number}/{len(selected_stages)}] Command {c}: {s["name"]}')
        kind = Kind(s['depth'], s['dest_format'],
                    s['stage']['info'] >> 13 & 7, s['surface_format'] == 1)
        C = console.words(c)
        prior_c = console.words(s['prior']) if s['prior'] is not None \
            else None
        # An earlier pass's eDRAM counts only where the console still holds
        # it: a pass that clears first draws over nothing of it.
        if prior_c is not None and sum(
                1 for a, b in zip(C, prior_c) if a == b) * 100 < len(C):
            s['prior'] = prior_c = None
        P = backends[[b.label for b in backends].index(primary)].main.words(c)
        bg = background(C, P)
        cset = set(C)
        disjoint = len(cset) <= 8 and not cset & set(P)
        shade = Shade(kind, C, bg)
        crgb = shade.image(C)
        base = dim(crgb)
        files = {'console': f'img/c{c}-console.png'}
        (out / files['console']).write_bytes(
            png_bytes(s['width'], s['height'], bytes(crgb)))
        data = {'w': s['width'], 'h': s['height'], 'console': packed(C),
                'backends': {}}
        per = {}
        for run_number, (label, replay, variant) in enumerate(columns, 1):
            run_started = time.monotonic()
            vspec = next((v for v in variants if v['of'] == label), None)                 if variant is None else None
            variant_result = vspec['replay'].main if vspec else None
            other_result = others.get(label)
            key = cache.key({**common_key, 'command': c, 'prior': s['prior'],
                             'main': result_digest(replay.main),
                             'low': result_digest(replay.low),
                             'high': result_digest(replay.high),
                             'variant': result_digest(variant_result),
                             'other': result_digest(other_result),
                             'is_primary': label == primary,
                             'seed': replay.seed_status})
            cached = cache.load(key)
            log(f'  [{run_number}/{len(columns)}] {label}: '
                + ('reusing unchanged analysis' if cached else 'analyzing pixels'))
            sl = slug(label)
            img = {'image': f'img/c{c}-{sl}.png',
                   'delta': f'img/c{c}-{sl}-delta.png'}
            if cached:
                st = cached['stats']
                data['backends'][label] = cached['data']
                cache.publish(key, img, out)
            else:
                B = replay.main.words(c)
                low = replay.low.words(c) if replay.low else None
                high = replay.high.words(c) if replay.high else None
                prior_b = replay.main.words(s['prior'])                     if s['prior'] is not None else None
                vrep = variant_result.words(c) if variant_result else None
                other = other_result.words(c) if other_result else None
                st, code = compare(s, kind, C, B, low, high, prior_c, prior_b,
                                   vrep, other, played_mode, disjoint, bg,
                                   P if label != primary else None)
                pixels = {
                    'main': packed(xor(B, C)),
                    'low': packed(xor(low, B)) if low is not None else None,
                    'high': packed(xor(high, B)) if high is not None else None}
                image = png_bytes(s['width'], s['height'], bytes(shade.image(B)))
                delta = png_bytes(s['width'], s['height'], bytes(heatmap(base, code)))
                (out / img['image']).write_bytes(image)
                (out / img['delta']).write_bytes(delta)
                cache.store(key, st, pixels, image, delta)
                data['backends'][label] = pixels
            st['disjoint'] = disjoint
            if disjoint:
                st['console_bands'] = bands(C, s['width'])
            st['background'] = f'{bg:08X}' if bg is not None else None
            hyps = hypotheses(s, kind, st, mode, vspec, areas, mentions, names)
            per[label] = {'stats': st, 'hypotheses': hyps, 'images': img,
                          'direction': direction(st, kind),
                          'observation': observe(s, kind, st, names),
                          'brief': brief(s, kind, st, names),
                          'variant': bool(variant),
                          'variant_of': variant['of'] if variant else None}
            log(f'    {st["differing"]:,}/{st["pixels"]:,} pixels differ from Xbox '
                f'| {time.monotonic() - run_started:.1f}s')
        (out / 'data' / f'c{c}.js').write_text(
            'window.__stageData&&window.__stageData(' + str(c) + ',' +
            json.dumps(data, separators=(',', ':')) + ');\n')
        real = {k: v for k, v in per.items() if not v['variant']}
        agree = agreement(real, primary)
        p = per[primary]['stats']
        log(f'{c:5d} {s["name"]}: {p["differing"]}/{p["pixels"]} differ; '
            + '; '.join(f'{h["confidence"]} {h["key"]} {h["share"]:.0%}'
                        for h in per[primary]['hypotheses'][:4]) +
            f' [command {time.time() - t0:.1f}s; '
            f'total {time.monotonic() - build_started:.1f}s]')
        entries.append({
            'command': c, 'name': s['name'], 'derived': s['derived'],
            'named': str(c) in notes.get('stages', {}),
            'frame': s['frame'], 'kind': s['kind'],
            'format': ('D24FS8' if kind.depth_float else 'D24S8') if s['depth'] else
            DEST_FORMATS.get(s['dest_format'], f'format {s["dest_format"]}'),
            'decode': ('depth-float' if kind.depth_float else 'depth') if s['depth'] else
            {6: '8888', 7: '2101010', 36: 'float'}.get(s['dest_format'],
                                                      'raw'),
            'channels': kind.channels, 'channel_logs': kind.log,
            'component_bits': kind.bits, 'component_count': kind.count,
            'floating': kind.floating, 'number_format': kind.number,
            'pixel_bytes': play.COPY_BPP[s['dest_format']],
            'preview_note': ('HDR colour preview clamps channels to 0–1; all original pixel bits are compared.'
                             if kind.floating and kind.count > 1 else
                             'Integer preview scales the full component range; raw values remain integers.'
                             if kind.number in (2, 3) and not kind.depth else ''),
            'surface_format': 'D24FS8' if s['depth'] and s['surface_format']
            else COLOR_FORMATS.get(s['surface_format'], '') if not s['depth']
            else 'D24S8',
            'width': s['width'], 'height': s['height'],
            'samples': s['samples'], 'edram_base': s['base'],
            'dest_base': f'0x{s["stage"]["base"]:08X}',
            'copy_control': f'0x{s["copy_control"]:X}',
            'inputs': [{'command': x, 'name': names.get(x)}
                       for x in s['inputs']],
            'prior': {'command': s['prior'], 'name': names.get(s['prior'])}
            if s['prior'] is not None else None,
            'draws': len(s['window']), 'groups': draw_groups(s['window']),
            'hiz_tests': s['hiz_tests'], 'his_tests': s['his_tests'],
            'hiz_written': s['hiz_written'], 'gamma': s['gamma'],
            'blend': s['blend'], 'signature': list(s['signature']),
            'console_image': files['console'], 'data': f'data/c{c}.js',
            'backends': per, 'agree': agree})

    rows = group_rows(entries, primary)
    report = {
        'title': config.get('title') or f'Title {meta.get("title_id", 0):08X}',
        'mode': mode, 'other_mode': other_mode,
        'other_report': relative((run / config['modes'][other_mode]['out'] /
                                  'report.html').resolve(), out)
        if other_mode else None,
        'suite': meta.get('suite'), 'commands': total,
        'command_timeline': [
            {'command': d['command'],
             'measured': d['command'] in {s['command'] for s in entries},
             'name': ('Resolve' if d['regs']['RB_MODECONTROL'] & 7 == 6
                      else 'Draw') + ' ' + str(d['command']),
             'primitive': PRIMS.get(d['prim'], str(d['prim'])),
             'count': d['count'], 'state': draw_state(d)}
            for d in walker.draws],
        'run': run.name, 'console_date': datetime.datetime.fromtimestamp(
            (run / 'console_results.bin').stat().st_mtime).strftime(
                '%Y-%m-%d'),
        'generated': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
        'primary': primary,
        'selected_runs': spec.get('selected_runs', {}),
        'comparison_runs': spec.get('comparison_runs', {}),
        'backends': [{'label': b.label, 'dir': relative(b.dir.resolve(), out),
                      'provenance': provenance(b.dir),
                      'seed_verification': b.seed_status} for b in backends],
        'variants': [{'label': v['label'], 'of': v['of'],
                      'change': v['change'], 'name': v['name'],
                      'run_id': v.get('run_id', v['label']),
                      'parent_run': v.get('parent_run', v['of']),
                      'patches': v.get('patches', []),
                      'dir': relative(v['replay'].dir.resolve(), out),
                      'provenance': provenance(v['replay'].dir),
                      'seed_verification': v['replay'].seed_status}
                     for v in variants],
        'frames': frames, 'frame_boundary_source': walker.frame_boundary_source,
        'rows': rows, 'stages': entries,
        'coverage': relative(COVERAGE / 'coverage.html', out)
        if (COVERAGE / 'coverage.html').exists() else None,
        'areas_dir': relative(COVERAGE / 'areas', out),
    }
    (out / 'report.json').write_text(json.dumps(report, indent=1))
    render(out)
    log(f'Finished {mode}: {len(entries)} commands in '
        f'{time.monotonic() - build_started:.1f}s | '
        f'{cache.hits} cached run comparisons, {cache.misses} analyzed')
    return out


def render(out):
    """OUT/report.html from OUT/report.json and the page template."""
    report = json.loads((out / 'report.json').read_text())
    page = TEMPLATE.read_text().replace(
        '__TITLE__', html.escape(f'{report["title"]} Parity')).replace(
        '/*__DATA__*/null', json.dumps(report, separators=(',', ':'))
        .replace('</', '<\\/'))
    (out / 'report.html').write_text(page)
    log(out / 'report.html')


def group_rows(entries, primary):
    """Overview rows: a pass and its repeats in the next frames when the
    outcome is the same kind, ranked by wrong pixels."""
    rows, used = [], set()
    for e in entries:
        if e['command'] in used:
            continue
        members = [e]
        used.add(e['command'])
        for f in entries:
            if f['command'] in used or f['frame'] != members[-1]['frame'] + 1:
                continue
            if f['signature'] == e['signature'] and f['name'] == e['name'] \
                    and outcome(f, primary) == outcome(e, primary):
                members.append(f)
                used.add(f['command'])
        stats = [m['backends'][primary]['stats'] for m in members]
        wrong = [st['differing'] / st['pixels'] for st in stats]
        top = members[0]['backends'][primary]['hypotheses']
        na = sum(st['observation'].get('pretrace', 0) for st in stats)
        dif = sum(st['differing'] for st in stats)
        rows.append({
            'commands': [m['command'] for m in members],
            'name': e['name'], 'wrong': wrong,
            'impact': (dif - na) / len(members),
            'state': 'identical' if not dif else
            'na' if na * 2 > dif else 'differs'})
    rank = {'differs': 0, 'na': 1, 'identical': 2}
    rows.sort(key=lambda r: (rank[r['state']], -r['impact'],
                             r['commands'][0]))
    return rows


def outcome(e, primary):
    st = e['backends'][primary]['stats']
    if not st['differing']:
        return 'identical'
    h = e['backends'][primary]['hypotheses']
    return h[0]['key'] if h else 'other'


def add_arguments(p):
    p.add_argument('run', help='the console run directory')
    p.add_argument('--config', help='default RUN/stage-report.json')
    p.add_argument('--mode', action='append',
                   help='build only these modes (default all)')
    p.add_argument('--only', help='only these commands (comma-separated), '
                   'for a quick look; the page then lacks the rest')
    p.add_argument('--render', action='store_true',
                   help='only rewrite report.html from report.json')
    p.add_argument('--no-cache', action='store_true',
                   help='recompute all pixel analysis instead of reusing cached results')


def run_from_args(args):
    run = Path(args.run).resolve()
    config = json.loads(Path(args.config or run / 'stage-report.json')
                        .read_text())
    if args.render:
        for mode in args.mode or list(config['modes']):
            render((run / config['modes'][mode]['out']).resolve())
        return
    log(f'Report: {config.get("title", run.name)}')
    log(f'Loading recorded comparison: {run}')
    meta = json.loads((run / 'gpugen.json').read_text())
    notes = json.loads((run / 'notes.json').read_text()) \
        if (run / 'notes.json').exists() else {}
    areas = load_areas()
    t0 = time.time()
    walker = walk(run)
    log(f'trace: {walker.command} commands, {len(walker.draws)} draws, '
        f'{walker.frame_boundary_source} at {walker.swaps} '
        f'[{time.time() - t0:.0f}s]')
    built = []
    for mode in args.mode or list(config['modes']):
        built.append(build(run, config, mode, walker, meta, notes, areas,
                           {int(x) for x in args.only.split(',')}
                           if args.only else None, not args.no_cache))
    for b in built:
        log(b / 'report.html')


def main(argv=None):
    parser = argparse.ArgumentParser(description='Generate an Xbox comparison report from recorded results.')
    add_arguments(parser)
    run_from_args(parser.parse_args(argv))


if __name__ == '__main__':
    main()
