**Historical 512-token protocol.** Current collection disables truncation and rejects nonzero truncation. The constant-feature cache is only a diagnostic of the old failure; do not use it for new benchmarks. Use [the full-context run instructions](agent_full_context_status.md) instead.

# Agent behavior benchmark runs

This extension reuses `notebooks/agent_outcome_hpo.ipynb` without changing its cells or MALT artifacts. The evaluation wrapper loads its feature reader, metric functions, XGBoost fixed parameters, Optuna search space, and latency benchmark directly from the notebook.

## Protocol

- Terminal Wrench: 25% task holdout, seed 42, with all models and trajectory variants of a task assigned together. Three grouped inner folds; eight Optuna trials, maximizing normal-class F1 as in MALT. Original and stripped variants use identical task assignments.
- TRACE: stratified five-fold outer CV, with three stratified inner folds and eight trials per outer fold. The release has only `trajectory_id`, `conversation`, and `label`; this is **not task-held-out** evaluation.
- Binary reporting uses abnormal=1, normal=0. PR-AUC is average precision. Macro F1 is calculated on the binary label with both classes included.
- Operating-point targets are 95%, 97%, and 99% abnormal catch. Thresholds are selected from training-only out-of-fold predictions and then applied to held-out rows. Test curves use a fixed threshold grid and do not choose deployment thresholds.
- This corrects two limitations of the existing notebook addendum for the new runs: its inner folds are trajectory-stratified, and its operating-point selection uses test labels. The original notebook is preserved for reproducibility.
- The exact MALT extraction configuration is retained: FLAN-T5-large, int8, batch size 1, 512-token truncation, max 8 generated tokens, zones and layer-distribution features. The actual MALT artifact contains 7,128 columns, including 4,320 decoder feature columns; it is not encoder-only.
- Original trajectory text, tool calls, and observations are retained in order. The last assistant turn and any subsequent observations/messages form `output`; earlier turns form `input`. Terminal Wrench's published red-team elicitation excerpt is removed, following MALT's removal of experimental instructions. Source labels and judge explanations never enter the model text.

## Source revisions and populations

- Terminal Wrench: `d8a29613235a0ef56a8b70b3142626a533da28c2` from https://github.com/few-sh/terminal-wrench. Both original and stripped variants have 7,200 rows: 3,632 hacks, 2,352 baselines, and 1,216 legitimate attacker solves, covering 331 tasks. Excluded: 1,441 no-reward attempts. Model counts: Claude Opus 4.6 2,760; Gemini 3.1 Pro 2,517; GPT-5.4 1,923.
- TRACE: `31d87f06078eca3ab6eaf1e06e5ea6fe9f2b7a6d` from https://huggingface.co/datasets/PatronusAI/trace-dataset. 517 rows: 268 abnormal and 249 benign. `0` is benign; comma-separated category codes indicate hacks. Original codes are retained for per-category breakdowns. Access requires accepting the dataset conditions with the authenticated Hugging Face account.

## Rerun

Run from the repository root after installing the existing project dependencies. Input preparation and extraction refuse to replace completed outputs; use fresh names or preserve old outputs before rerunning.

```powershell
uv run python -m scripts.prepare_agent_benchmarks trace --revision 31d87f06078eca3ab6eaf1e06e5ea6fe9f2b7a6d --output runs/datasets/trace-full.jsonl
uv run python -m scripts.prepare_agent_benchmarks terminal-wrench --revision d8a29613235a0ef56a8b70b3142626a533da28c2 --output runs/datasets/terminal-wrench-full.jsonl
uv run python -m scripts.prepare_agent_benchmarks terminal-wrench --variant stripped --revision d8a29613235a0ef56a8b70b3142626a533da28c2 --output runs/datasets/terminal-wrench-stripped.jsonl
uv run python -m scripts.collect_agent_benchmark runs/datasets/trace-full.jsonl --run-id trace-full
```

Terminal Wrench's first 512 tokens are dominated by common agent setup instructions. Exact-input reuse is supported to avoid redundant inference: tokenize every full prompt, group by the complete encoded token IDs and attention mask, and extract representatives for each source variant, model, and outcome. Expansion is permitted only when every representative in a group produces **bit-identical features**. All 7,200 trajectories, their metadata, and their statistical weights are preserved. No class labels are used to create token groups; class/model fields select additional validation representatives only.

```powershell
uv run python -m scripts.cache_agent_inputs prepare --inputs runs/datasets/terminal-wrench-full.jsonl runs/datasets/terminal-wrench-stripped.jsonl --compact runs/datasets/terminal-wrench-token-inputs.jsonl
uv run python -m scripts.collect_agent_benchmark runs/datasets/terminal-wrench-token-inputs.jsonl --run-id terminal-wrench-token-inputs
uv run python -m scripts.cache_agent_inputs expand --compact runs/datasets/terminal-wrench-token-inputs.jsonl --features runs/remote/terminal-wrench-token-inputs
```

Alternatively, run the normal collector directly on each full normalized JSONL with run IDs `terminal-wrench-full` and `terminal-wrench-stripped`. This computes each trajectory independently using the same pipeline.

```powershell
uv run python notebooks/evaluate_agent_benchmark.py --dataset trace --features runs/remote/trace-full --output notebooks/runs/notebook-agent-outcome-hpo/trace-full
uv run python notebooks/evaluate_agent_benchmark.py --dataset terminal-wrench --features runs/remote/terminal-wrench-full --group task_id --output notebooks/runs/notebook-agent-outcome-hpo/terminal-wrench-full
uv run python notebooks/evaluate_agent_benchmark.py --dataset terminal-wrench-stripped --features runs/remote/terminal-wrench-stripped --group task_id --output notebooks/runs/notebook-agent-outcome-hpo/terminal-wrench-stripped
uv run python -m scripts.summarize_agent_benchmarks
uv run pytest neuralsignal/tests/test_agent_benchmarks.py -q
```

The collector uses the configured RunPod image and private S3 handoff from MALT. Normalized data is sent through a temporary signed URL and deleted from handoff storage when collection ends. No public dataset publication is involved. Supply `--ssh-key PATH` if the default local identity does not exist. No private SSH key is sent to the worker.

## Outputs

Each evaluation directory contains `metrics.json`, `predictions.csv`, `breakdown.csv`, `operating-points.csv`, `threshold-sweep.csv`, `split-assignments.csv`, per-fold HPO trials, training out-of-fold predictions, saved models, and `agent_outcome_xgb_latency.csv`. Source audits are retained beside normalized inputs and in each feature directory. Feature shard checksums and row counts are verified before evaluation.

Category breakdowns compare each hack category against all benign rows; multi-label hack rows contribute to multiple category slices. These are binary detection metrics, not the paper's category-classification Match Rate. The latency benchmark measures only XGBoost prediction on existing features, not end-to-end detection.

## Baseline interpretation

Terminal Wrench's published GPT-5.4 monitor reports original AUC 0.9679 and stripped AUC 0.9168 (sanitized AUC 0.9474). Its negative population includes only 2,352 baselines, whereas our requested main run additionally includes 1,216 legitimate attacker solves. `breakdown.csv` includes a hacks-versus-baselines-only slice on the held-out tasks. Even that slice differs from the published full-corpus evaluation.

TRACE's paper reports GPT-5.2 isolated-trajectory Detection Rate of 45%, where Detection Rate means binary macro F1: https://arxiv.org/html/2601.20103v1. Our supervised cross-validation protocol trains on labeled TRACE trajectories and uses truncated input; it is not a like-for-like zero-shot judge comparison.

Results are written after evaluation completes. No results should be inferred from this protocol description.

## Changed components

- New dataset adapters: `neuralsignal/datasets/sources/terminal_wrench.py`, `trace.py`, and shared MALT rendering adapter `agent_transcripts.py`.
- Normalization, collection, exact-input reuse, and reporting commands: `scripts/prepare_agent_benchmarks.py`, `collect_agent_benchmark.py`, `cache_agent_inputs.py`, and `summarize_agent_benchmarks.py`.
- Dataset-configurable evaluation: `notebooks/evaluate_agent_benchmark.py`.
- Focused validation: `neuralsignal/tests/test_agent_benchmarks.py`.
- Protocol and results: this document and `docs/agent_benchmark_results.md`.

Existing MALT notebook cells and artifacts were not modified. Pre-existing edits in S1 training/config/tests and the feature-schema specification were left untouched.
