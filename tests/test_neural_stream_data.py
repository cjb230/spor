from pathlib import Path

import numpy as np
import pytest


def sample(n=7):
    from schotten.neural.stream_data import pack_examples
    rng = np.random.default_rng(42)
    features = np.zeros((n, 1082), dtype=np.float32)
    features[:, :1080] = rng.integers(0, 2, (n, 1080))
    features[:, -2] = np.arange(n) % 43 / 42
    features[:, -1] = np.arange(n) % 7 / 6
    masks = np.zeros((n, 487), dtype=bool)
    actions = np.arange(n, dtype=np.int64)
    masks[np.arange(n), actions] = True
    return features, masks, actions, pack_examples(features, masks, actions)


def test_packed_roundtrip_and_bounded_batch():
    from schotten.neural.stream_data import unpack_batch
    features, masks, actions, arrays = sample()
    assert arrays['packed'].shape == (7, 135)
    assert arrays['mask'].shape == (7, 61)
    f, m, a = unpack_batch(arrays, [6, 1, 4])
    np.testing.assert_array_equal(f, features[[6, 1, 4]])
    np.testing.assert_array_equal(m, masks[[6, 1, 4]])
    np.testing.assert_array_equal(a, actions[[6, 1, 4]])


def test_atomic_dataset_roundtrip_integrity_and_cursor(tmp_path):
    from schotten.neural.stream_data import (
        create_dataset, append_shard, finish_dataset, Dataset, open_manifest,
    )
    root = tmp_path / 'data'
    create_dataset(root, {'examples': 7, 'validation_examples': 7})
    arrays = sample()[-1]
    append_shard(root, 'train', arrays, cursor={'next_game': 3})
    with pytest.raises(ValueError, match='complete'):
        Dataset(root)
    append_shard(root, 'validation', arrays, cursor={'next_game': 6})
    finish_dataset(root)
    dataset = Dataset(root)
    assert dataset.count('train') == 7
    assert dataset.count('validation') == 7
    assert open_manifest(root)['cursor'] == {'next_game': 6}
    entry = dataset.entries('train')[0]
    np.testing.assert_array_equal(dataset.load(entry)['actions'], arrays['actions'])
    assert dataset.fingerprint == Dataset(root).fingerprint
    path = root / entry['path']
    path.write_bytes(path.read_bytes() + b'corrupt')
    with pytest.raises(ValueError, match='hash|integrity'):
        dataset.load(entry)


def test_rejects_invalid_labels_and_features():
    from schotten.neural.stream_data import pack_examples
    f, m, a, _ = sample()
    a[0] = 486
    with pytest.raises(ValueError, match='legal'):
        pack_examples(f, m, a)
    a[0] = 0
    f[0, 0] = 0.25
    with pytest.raises(ValueError, match='binary'):
        pack_examples(f, m, a)


def test_finish_requires_exact_counts_and_no_append_after_finish(tmp_path):
    from schotten.neural.stream_data import create_dataset, append_shard, finish_dataset
    create_dataset(tmp_path, {'examples': 7, 'validation_examples': 7})
    append_shard(tmp_path, 'train', sample()[-1], cursor={})
    with pytest.raises(ValueError, match='count|examples'):
        finish_dataset(tmp_path)
    append_shard(tmp_path, 'validation', sample()[-1], cursor={})
    finish_dataset(tmp_path)
    with pytest.raises(ValueError, match='complete'):
        append_shard(tmp_path, 'train', sample()[-1], cursor={})


def test_heartbeat_runs_without_updates():
    import io
    import time
    from schotten.neural.stream_data import Progress
    output = io.StringIO()
    with Progress('test', interval=0.01, stream=output) as progress:
        progress.update(1, 10, 'working')
        time.sleep(0.04)
    assert output.getvalue().count('\r') >= 2
    assert 'ETA' in output.getvalue()


def test_writer_lock_releases_after_error(tmp_path):
    from schotten.neural.stream_data import exclusive_lock
    with exclusive_lock(tmp_path):
        with pytest.raises(ValueError, match='another writer'):
            with exclusive_lock(tmp_path):
                pass
    with exclusive_lock(tmp_path):
        pass


def test_resumed_progress_does_not_count_previous_work_as_current_rate():
    import io
    from schotten.neural.stream_data import Progress
    output = io.StringIO()
    progress = Progress('resumed', stream=output)
    progress.update(40000000, 125000000, 'resuming')
    progress.close()
    assert '| 0/s |' in output.getvalue()


def test_manifest_rejects_path_traversal(tmp_path):
    import json
    from schotten.neural.stream_data import create_dataset, append_shard, open_manifest
    create_dataset(tmp_path, {'examples': 7, 'validation_examples': 0})
    append_shard(tmp_path, 'train', sample()[-1], cursor={})
    path = tmp_path / 'manifest.json'
    value = json.loads(path.read_text())
    value['shards'][0]['path'] = '../elsewhere.npz'
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='manifest'):
        open_manifest(tmp_path)
