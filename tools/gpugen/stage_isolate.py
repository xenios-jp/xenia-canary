"""stage_isolate.py OUTDIR TRACE: writes TRACE, OUTDIR's gpugen.xtr with
the console's own output of each stage put in memory right after the stage's
resolve, so the draws after it sample what the console resolved. This isolates inherited resolved-memory inputs. It does not restore eDRAM,
so later stages can still inherit differences through render targets. Keep
the ordinary replay as the primary comparison and use this as a diagnostic.

The memory comes after the resolve's packet, where a replay stopping at that
command has already stopped, so the stage itself is the backend's."""
import argparse
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpugen  # noqa: E402
import play  # noqa: E402


def packet_ends(data):
    """{command: offset after its PACKET_END}, commands counted as Xenia's
    trace reader counts them: every draw packet, in indirect buffers too."""
    ends, command, o, packet = {}, 0, play.TRACE_HEADER.size, None
    while o < len(data):
        kind = struct.unpack_from('<I', data, o)[0]
        if kind == play.PACKET_START:
            count = struct.unpack_from('<I', data, o + 8)[0]
            packet = struct.unpack_from('>I', data, o + 12)[0] if count else 0
            o += 12 + 4 * count
        elif kind == play.PACKET_END:
            o += 4
            if packet is not None and packet >> 30 == 3 and \
                    packet >> 8 & 127 in (0x22, 0x36):
                ends[command] = o
                command += 1
            packet = None
        elif kind in (play.PRIMARY_START, play.INDIRECT_START):
            count = struct.unpack_from('<I', data, o + 8)[0]
            o += 12 + 4 * count
        elif kind == play.PRIMARY_END:
            o += 4
        elif kind == play.INDIRECT_END:
            # Xenia's reader skips the PACKET_END wrapping the buffer.
            o += 8
            packet = None
        elif kind in (play.MEMORY_READ, play.MEMORY_WRITE):
            encoded = struct.unpack_from('<I', data, o + 12)[0]
            o += 20 + encoded
        elif kind == play.REGISTERS:
            size = struct.unpack_from('<I', data, o + 20)[0]
            o += 24 + size
        elif kind == play.EVENT:
            o += 8
        elif kind == play.EDRAM_SNAPSHOT:
            o += 12 + struct.unpack_from('<I', data, o + 8)[0]
        elif kind == play.GAMMA_RAMP:
            o += 16 + struct.unpack_from('<I', data, o + 12)[0]
        else:
            raise play.TraceError(f'record {kind} at {o}')
    return ends


def add_arguments(parser):
    parser.add_argument('outdir', help='console reference run directory')
    parser.add_argument('trace', help='new diagnostic trace path')


def run_from_args(args):
    outdir, path = Path(args.outdir), Path(args.trace)
    if path.resolve() == (outdir / 'gpugen.xtr').resolve():
        raise SystemExit('isolate: preserve the original trace; choose a new output path')
    meta = json.loads((outdir / 'gpugen.json').read_text())
    console = gpugen.decode(meta, (outdir / 'console_results.bin')
                            .read_bytes())['raw']
    data = (outdir / 'gpugen.xtr').read_bytes()
    ends = packet_ends(data)
    inserts = []
    for test in meta['tests']:
        stage = test['stage']
        output = console[test['name']]
        if len(output) != stage['size']:
            raise SystemExit(f'{test["name"]}: {len(output)} bytes, not '
                             f'{stage["size"]}')
        inserts.append((ends[stage['command']], struct.pack(
            '<IIIII', play.MEMORY_READ, stage['base'], play.ENCODING_NONE,
            len(output), len(output)) + output))
    out, last = bytearray(), 0
    for offset, record in sorted(inserts, key=lambda x: x[0]):
        out += data[last:offset] + record
        last = offset
    out += data[last:]
    path.write_bytes(out)
    print(f'{path}: {len(inserts)} stages')


def main(argv=None):
    parser = argparse.ArgumentParser(description='Create a resolved-memory isolation diagnostic trace.')
    add_arguments(parser)
    run_from_args(parser.parse_args(argv))


if __name__ == '__main__':
    main()
