import sys
from pathlib import Path
from types import SimpleNamespace

from schotten.neural.__main__ import main


def test_generate_cli_dispatches_exact_example_targets(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setitem(sys.modules, 'schotten.neural.generate',
                        SimpleNamespace(generate=lambda **kw: calls.append(kw)))
    assert main(['generate', '--output', str(tmp_path), '--examples', '40000000',
                 '--validation-examples', '100000', '--seed', '1001']) == 0
    assert calls[0]['examples'] == 40000000
    assert calls[0]['validation_examples'] == 100000
    assert calls[0]['output'] == tmp_path
    assert calls[0]['progress_interval'] == 1
    assert calls[0]['threads'] is None  # Resolve all available cores in generation.


def test_stream_cli_passes_total_epochs_and_resume(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setitem(sys.modules, 'schotten.neural.stream_train',
                        SimpleNamespace(train_stream=lambda **kw: calls.append(kw)))
    assert main(['train-stream', '--dataset', 'datasets/test', '--initial', 'initial.pt',
                 '--output', str(tmp_path), '--epochs', '3', '--resume', 'last.pt',
                 '--device', 'cpu']) == 0
    assert calls[0]['epochs'] == 3
    assert calls[0]['resume'] == Path('last.pt')
    assert calls[0]['batch_size'] == 512
