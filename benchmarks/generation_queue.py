import argparse
import hashlib
import io
import json
import resource
import time
from pathlib import Path
import numpy as np
from schotten.neural.generate import _Worker, _arrays
from schotten.neural.stream_data import Progress, digest

parser = argparse.ArgumentParser()
parser.add_argument('--worker', type=Path, required=True)
parser.add_argument('--threads', type=int, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--requests', type=int, default=4, help='number of consecutive 1024-row requests')
args = parser.parse_args()
if args.requests < 1:
    parser.error('--requests must be positive')
total = args.requests * 1024
worker = args.worker.resolve()
blocks, cursors = [], []
checksum = hashlib.sha256()
before = resource.getrusage(resource.RUSAGE_CHILDREN)
start = time.perf_counter()
cursor = 0
with Progress(f'Queue benchmark ({args.threads} threads)') as progress, _Worker(worker, args.threads) as process:
    progress.update(0, total, 'generating fixed MCB:16 schedule')
    for i in range(args.requests):
        cursor, rows = process.request(dict(op='generate',seed=2001,split='train',start=cursor,
            count=1024,teacher='mcb:16',opponents=['builder','tactician','planner','mcb:16']))
        cursors.append(cursor)
        blocks.append(rows)
        checksum.update(rows.tobytes())
        progress.update((i+1)*1024,total,'native generation')
    shutdown_started = time.perf_counter()
elapsed = time.perf_counter()-start
shutdown_seconds = time.perf_counter()-shutdown_started
after = resource.getrusage(resource.RUSAGE_CHILDREN)
cpu = after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime
arrays = _arrays(np.concatenate(blocks))
compressed = io.BytesIO()
np.savez_compressed(compressed,**arrays)
result = dict(worker_sha256=digest(worker),threads=args.threads,examples=total,
    seed=2001,teacher='mcb:16',opponents=['builder','tactician','planner','mcb:16'],
    requests=[1024]*args.requests,wall_seconds=elapsed,cpu_seconds=cpu,average_cpu_cores=cpu/elapsed,
    examples_per_second=total/elapsed,rows_sha256=checksum.hexdigest(),cursors=cursors,
    shutdown_seconds=shutdown_seconds,
    compressed_npz_bytes=len(compressed.getbuffer()),
    compressed_bytes_per_example=len(compressed.getbuffer())/total,
    estimated_125m_decimal_GB=len(compressed.getbuffer())/total*125000000/1e9)
args.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
