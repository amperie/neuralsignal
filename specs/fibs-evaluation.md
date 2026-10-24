# FIBS evaluation

Status: implemented (CPU unit-tested; no GPU collection run yet).

## Dataset
- Hugging Face `AlignmentResearch/fibs-v1`, config `default` (arXiv 2610.12445, far.ai/fibs).
- **The dataset card declares no license.** Confirm terms before redistributing data or derived features.
- Splits: `honest_honest`, `honest_deceptive`, `deceptive_honest`, `deceptive_deceptive` (select via `split: all | <name> | [list]`).
- Columns used: `messages_json`, `completion`, `judge_verdict`, `quadrant`, `source`, `_v32_source_content_md5`, `pair_id`, `tools`.
  `judge_justification` and `an_original_completion` are deliberately not read (label leakage).

## Labels
- Target `deceptive` (`source: fibs_verdict`): `LIE`/`UNTRUTHFUL` -> 1, `TRUTHFUL`/`HONEST` -> 0, anything else excluded.
- Labels are **AI-judge generated** and scoped to the final response; expect label noise.
- Each split is effectively single-class (label = second half of quadrant), so per-split AUROC/AUPRC is only defined where both classes exist.

## Rendering and long transcripts
- The judge model stays a T5-family encoder-decoder (flan-t5 / LongT5). Input = ordered `role: content` lines (tool calls appended), output = `assistant: <completion>`.
- Example config uses `google/long-t5-tglobal-large` at 4096 tokens with the final response placed first so truncation cannot drop it.
- Optional `chunk_chars` / `chunk_overlap_chars` split long histories into overlapping chunks; chunk features are mean-pooled per `fibs_row_id` before training.

## Splitting
- Episodes = union-find over content md5 and `pair_id`; GroupShuffleSplit for test, then validation carved from train. No episode crosses partitions.
- `mode: cross_split`: episodes are partitioned first, then train keeps only `train_splits` rows and test only `test_splits` rows. Each side needs both classes, e.g. train `[honest_honest, deceptive_deceptive]`, test `[deceptive_honest, honest_deceptive]`.

## Metrics
- Existing S1 metrics unchanged.
- Recall at 1% and 5% FPR with thresholds chosen on validation negatives only; achieved test FPR is reported too.
- AUROC/AUPRC overall and per FIBS split.
- Baselines (logistic regression): `length` (log transcript chars), `step` (step position), `length_step`.

## Limitations
- No GPU collection or detector-quality claim yet; MLflow remote logging untested for new metrics.
- `max_examples_per_split` counts transcripts, but a global `max_examples` cap counts chunks.
