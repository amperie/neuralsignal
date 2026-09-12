# Padding and batching

[generate_from_batch](../neuralsignal/core/modules/model_instrumentation.py)
pads batches, applies a positive configured truncation length consistently to
single and batched input, and retains each example's attention mask. Zero
truncation length means no explicit truncation. Inputs move to the model device.

The mask is passed to generation and attached to the scan. Hook cleanup runs
in a `finally` block across tokenization/generation/decoding/finalization failures.
An intentional collector abort produces `ABORT` per example without indexing a
fixed-size token list.

## Feature masking

- Zones filter matching token positions before sequence means.
- Layer distributions filter matching token positions before histograms.
- T-F-diff filters token positions before choosing the last valid token.
- `inference/masking.py` supplies masked mean/variance/std helpers.

`apply_attention_mask` only filters when mask length equals tensor length;
otherwise it returns the tensor unchanged. There is no semantic tracking of
encoder versus decoder/global positions. A source-token mask is not a validated
decoder or LongT5 global-block mask. Logit-lens remains outside the verified
padding-aware feature path and is disabled in examples.

## Evidence and limits

Synthetic padding fixtures test supported aggregations. Tiny T5/LongT5 tests
exercise real generation and hook collection on CPU. The smoke-settings T5 test
checks finite features and consistent column keys across a padded batch; it does
not prove equality of all decoder features across independent runs/batch sizes.
There is no automatic length bucketing, OOM batch-size retry, or universal
pretrained GPU invariance guarantee.

Run `uv run pytest neuralsignal/tests -q`; see [test coverage](testing-strategy.md).
