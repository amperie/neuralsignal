from pathlib import Path

from neuralsignal.config import load_config
from neuralsignal.tests.batch_invariance import assert_batch_invariant


def test_batch_invariance_config_matches_batch_size_one_baseline():
    config = load_config(Path("configs/tests/batch_invariance.yaml"))

    results = assert_batch_invariant(config)

    assert results
    assert max(result.max_abs_diff for result in results) == 0.0

