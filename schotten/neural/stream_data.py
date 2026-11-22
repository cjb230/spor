"""Compact, bounded-memory training shards and progress for long-running jobs."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import threading
import time

import numpy as np

from .encoding import ENCODING_VERSION


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix='.manifest-',
                                         delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def exclusive_lock(root):
    """An advisory process lock released automatically even after interruption."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.writer.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f'another writer is using {root}') from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def pack_examples(features, masks, actions):
    features = np.asarray(features, dtype=np.float32)
    masks = np.asarray(masks)
    actions = np.asarray(actions)
    n = len(features)
    if features.shape != (n, 1082) or masks.shape != (n, 487) or actions.shape != (n,):
        raise ValueError('invalid example shapes')
    if not np.isfinite(features).all() or not np.isin(features[:, :1080], [0, 1]).all():
        raise ValueError('observation bits must be binary and finite')
    scaled = features[:, -2:] * np.array([42, 6], dtype=np.float32)
    rounded = np.rint(scaled)
    if (not np.allclose(scaled, rounded, atol=1e-5)
            or (rounded < 0).any() or (rounded > [42, 6]).any()):
        raise ValueError('invalid card counts')
    if not np.isin(masks, [0, 1]).all():
        raise ValueError('legal masks must be binary')
    if not np.issubdtype(actions.dtype, np.integer) or (actions < 0).any() or (actions >= 487).any():
        raise ValueError('invalid action labels')
    if not masks[np.arange(n), actions].all():
        raise ValueError('teacher labels must be legal')
    return {'packed': np.packbits(features[:, :1080].astype(np.uint8), axis=1, bitorder='little'),
            'counts': rounded.astype(np.uint8),
            'mask': np.packbits(masks.astype(np.uint8), axis=1, bitorder='little'),
            'actions': actions.astype(np.uint16)}


def _validate_arrays(arrays):
    if set(arrays) != {'packed', 'counts', 'mask', 'actions'}:
        raise ValueError('invalid shard fields')
    n = len(arrays['actions'])
    for name, shape, dtype in [('packed', (n, 135), np.uint8),
                               ('counts', (n, 2), np.uint8),
                               ('mask', (n, 61), np.uint8),
                               ('actions', (n,), np.uint16)]:
        if arrays[name].shape != shape or arrays[name].dtype != dtype:
            raise ValueError(f'invalid shard {name} shape or dtype')
    if (arrays['counts'] > [42, 6]).any() or (arrays['actions'] >= 487).any():
        raise ValueError('invalid shard counts or action')
    actions = arrays['actions'].astype(np.int64)
    if not ((arrays['mask'][np.arange(n), actions // 8] >> (actions % 8)) & 1).all():
        raise ValueError('shard contains illegal teacher labels')
    if (arrays['mask'][:, -1] & 128).any():
        raise ValueError('nonzero mask padding')
    return n


def unpack_batch(arrays, indices):
    packed, counts = arrays['packed'][indices], arrays['counts'][indices]
    features = np.empty((len(packed), 1082), dtype=np.float32)
    features[:, :1080] = np.unpackbits(packed, axis=1, count=1080, bitorder='little')
    features[:, -2:] = counts.astype(np.float32) / np.array([42, 6], dtype=np.float32)
    mask = np.unpackbits(arrays['mask'][indices], axis=1, count=487, bitorder='little').astype(bool)
    return features, mask, arrays['actions'][indices].astype(np.int64)


def create_dataset(root, config):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if any(p.name != '.writer.lock' for p in root.iterdir()):
        raise ValueError(f'dataset directory is not empty: {root}; use resume')
    value = {'schema_version': 1, 'encoding_version': ENCODING_VERSION,
             'config': config, 'shards': [], 'counts': {'train': 0, 'validation': 0},
             'cursor': {}, 'complete': False}
    _atomic_json(root / 'manifest.json', value)
    return value


def open_manifest(root):
    root = Path(root)
    value = json.loads((root / 'manifest.json').read_text())
    if value.get('schema_version') != 1 or value.get('encoding_version') != ENCODING_VERSION:
        raise ValueError('incompatible dataset version')
    totals = {'train': 0, 'validation': 0}
    seen = set()
    for entry in value['shards']:
        path = Path(entry['path'])
        if (path.is_absolute() or len(path.parts) != 1 or path.suffix != '.npz'
                or entry['path'] in seen or entry['split'] not in totals
                or type(entry['count']) is not int or entry['count'] < 1):
            raise ValueError('invalid shard manifest entry')
        seen.add(entry['path'])
        totals[entry['split']] += entry['count']
    if totals != value['counts']:
        raise ValueError('dataset manifest count mismatch')
    return value


def append_shard(root, split, arrays, *, cursor):
    root = Path(root)
    manifest = open_manifest(root)
    if manifest['complete']:
        raise ValueError('dataset is already complete')
    if split not in ('train', 'validation'):
        raise ValueError('unknown dataset split')
    n = _validate_arrays(arrays)
    if n < 1:
        raise ValueError('cannot append an empty shard')
    target = manifest['config'].get('examples' if split == 'train' else 'validation_examples')
    if target is not None and manifest['counts'][split] + n > target:
        raise ValueError('shard exceeds requested example count')
    name = f'{split}-{len(manifest["shards"]):06d}.npz'
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=root, prefix='.shard-', delete=False) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        # Only an unreferenced orphan from an interrupted commit can occupy name.
        os.replace(temporary, root / name)
        entry = {'path': name, 'split': split, 'count': n, 'sha256': digest(root / name)}
        manifest['shards'].append(entry)
        manifest['counts'][split] += n
        manifest['cursor'] = cursor
        _atomic_json(root / 'manifest.json', manifest)
        return manifest
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def finish_dataset(root):
    manifest = open_manifest(root)
    for split, key in [('train', 'examples'), ('validation', 'validation_examples')]:
        if manifest['counts'][split] != manifest['config'][key]:
            raise ValueError(f'{split} examples do not match requested count')
    manifest['complete'] = True
    _atomic_json(Path(root) / 'manifest.json', manifest)
    return manifest


class Dataset:
    def __init__(self, root, require_complete=True):
        self.root = Path(root)
        self.manifest = open_manifest(self.root)
        if require_complete and not self.manifest['complete']:
            raise ValueError('generation must complete before training')
        self.fingerprint = hashlib.sha256(json.dumps(self.manifest, sort_keys=True,
                                                    separators=(',', ':')).encode()).hexdigest()

    def count(self, split):
        return self.manifest['counts'][split]

    def entries(self, split):
        return [e for e in self.manifest['shards'] if e['split'] == split]

    def load(self, entry):
        path = self.root / entry['path']
        if digest(path) != entry['sha256']:
            raise ValueError(f'shard integrity hash mismatch: {path}')
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        if _validate_arrays(arrays) != entry['count']:
            raise ValueError(f'shard count mismatch: {path}')
        return arrays


class Progress:
    """Heartbeat stays visible while the foreground is inside slow native/I/O calls."""
    def __init__(self, label, interval=1, stream=None):
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError('progress interval must be positive and finite')
        self.label, self.interval = label, interval
        self.stream = sys.stderr if stream is None else stream
        self.started = time.monotonic()
        self.completed, self.total, self.detail = 0, 0, 'starting'
        self._baseline = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = None

    def update(self, completed, total, detail=''):
        with self._lock:
            if self._baseline is None:
                self._baseline = completed
            self.completed, self.total, self.detail = completed, total, detail

    def _render(self):
        with self._lock:
            elapsed = max(time.monotonic() - self.started, 1e-6)
            rate = max(0, self.completed - (self._baseline or 0)) / elapsed
            eta = f'{max(0, self.total-self.completed)/rate:.0f}s' if rate else '--'
            text = (f'{self.label}: {self.completed:,} / {self.total:,} examples | '
                    f'{rate:,.0f}/s | elapsed {elapsed:.0f}s | ETA {eta} | {self.detail}')
        self.stream.write('\r' + text + '    ')
        self.stream.flush()

    def _run(self):
        while not self._stop.wait(self.interval):
            self._render()

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._render()
        self.stream.write('\n')
        self.stream.flush()

    def __exit__(self, *exc):
        self.close()
