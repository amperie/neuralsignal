# Tests and verification

Run the complete local suite from the repository root:

```bash
uv run pytest neuralsignal/tests -q
```

Last verified September 12, 2026: **175 passed**. This includes the bug sweeps and XGBoost trainer changes. The suite runs without downloading pretrained weights or launching pods.

## Coverage

- Config/CLI precedence, evaluator result counts, source normalization and feature selection.
- Parquet writing, manifest checksums/paths, bundle extraction, sync states, S3 errors.
- Training binary-label checks, manifest-backed reads, and MALT pooling/completeness.
- XGBoost parameter overrides and isolated MLflow registration/save/load predictions.
- Mocked RunPod launch, GPU choice, interruption/termination, and optional mirroring.
- Synthetic padding regressions for zones, distributions, and T-F-diff.
- Real tiny T5/LongT5 CPU generation, instrumentation flags, topology, and hook cleanup.
- A real tiny T5 with the smoke config's collector and both enabled feature processors.

The sweep tests are `test_sweep_regressions.py`, `test_second_sweep.py`,
`test_third_sweep.py`, and `test_fourth_sweep.py` under
[neuralsignal/tests](../neuralsignal/tests/). Saved dataset fixtures are
`malt_simple_112756.json` and `malt_transcript_111846.json` in its `fixtures/` folder.
The batch-invariance fixture config is [here](../configs/tests/batch_invariance.yaml).

## GPU acceptance

There is no established `pytest -m gpu` workflow. Build/publish the current image,
then use [the remote smoke launch](../configs/remote/malt_smoke.yaml) with a fresh
run ID. Verify actual rows/shards, finite features, both enabled prefixes, and
worker logs. The configured target is eight normalized examples. A returned
bundle or a completed manifest alone is insufficient.

Local tests do not establish pretrained quantization support, full GPU batch
invariance, or detector accuracy. See [verification status](../docs/current-state.md).
