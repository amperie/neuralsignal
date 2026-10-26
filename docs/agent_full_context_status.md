# Full-context agent benchmark rerun

Status: preparation and safeguards are complete; full-dataset extraction and replacement evaluation are blocked by dense T5 attention memory. No new benchmark metrics are available.

The user requires one complete trajectory per model pass. No windowing, truncation, or length-based exclusions were introduced. `configs/remote/agent_full_context_bs1.yaml` sets `generation.truncation_length: 0`; zero disables truncation in the existing tokenizer path. It retains the judge, prompt, quantization, batch size, feature definitions, and instrumentation. The remaining `512` zone sizes describe activation feature grouping, not a context cutoff. MALT configuration, notebook cells, and artifacts remain untouched.

## Full-input audit

| Dataset | Rows | Median tokens | Maximum tokens |
|---|---:|---:|---:|
| trace-full | 517 | 15,417.0 | 36,326 |
| terminal-wrench-full | 7,200 | 9,323.5 | 165,805 |
| terminal-wrench-stripped | 7,200 | 8,710.0 | 165,805 |

Token counts include the exact judge wrapper and EOS. All 14,917 rows were tokenized without truncation. Per-trajectory lengths and prompt checksums are in `runs/datasets/prompt-audit/*-lengths.csv`; six complete readable prompt examples and separate metadata files are in that directory. Examples are the first source row of each binary class, not cherry-picked by predictions. The exported text is what the existing renderer sends to tokenization.

## RunPod feasibility result

Selected TRACE's longest input by length only: `trajectory_0446`, 36,326 tokens. Ran the existing remote image with batch size 1 and truncation disabled on an NVIDIA H200 (139.80 GiB reported device capacity; advertised $4.59/hour, not total cost). Inference failed with CUDA out of memory: the process used 132.75 GiB and could not allocate another 9.83 GiB. Zero feature rows completed. This establishes failure for this configuration and device, not impossibility on every implementation or machine.

Pod `ziua6lyjyepsuj` was terminated and its absence verified. The local controller was stopped after the failed container kept restarting; the private input handoff was deleted and the signed URL revoked. The benchmark collector now detects the remote failed manifest during bundle polling, raises immediately, and lets the existing lifecycle terminate the pod. This prevents future failed-but-restarting jobs from waiting until the timeout.

Evidence:

- `runs/datasets/trace-full-context-preflight.remote.log`
- `runs/datasets/trace-full-context-preflight.provenance.json`
- `runs/datasets/trace-full-context-preflight.failed-manifest.json`
- `runs/datasets/trace-full-context-preflight.jsonl` and `.audit.json`

Terminal Wrench's maximum is 165,805 tokens. No full Terminal Wrench GPU job was launched after the shorter TRACE case failed. Continuing while retaining one complete context requires an inference change, such as memory-efficient global attention, or a different judge. Neither has been implemented or validated here; no replacement score should be inferred.

## Processed datasets and leakage repair

The original normalized JSONL files remain intact. These new copies add only a SHA-256 transcript identity to metadata; their `input` and `output` strings are unchanged:

- `runs/datasets/terminal-wrench-full-context.jsonl` (7,200 rows)
- `runs/datasets/terminal-wrench-stripped-context.jsonl` (7,200 rows)
- `runs/datasets/trace-full-context.jsonl` (517 rows)

Each row has input, output, labels, and dataset metadata. Labels and judge explanations are not inserted into the prompt. Terminal Wrench retains task and model identifiers; TRACE retains category labels. Each JSONL has a source audit and checksum beside it. The preparation command now writes transcript identities on future normalizations too.

TRACE trajectories `trajectory_0516` and `trajectory_0517` contain identical text and the same hack category, `1.3.2`. They landed in outer folds 1 and 4 in the historical run, causing duplicate leakage. New TRACE evaluation uses stratified five-fold CV grouped by transcript identity in both outer and inner splits. All 517 rows remain; this is still NOT task-held-out because no task/scenario field is released. Terminal Wrench retains task-held-out splitting and group-aware inner HPO. Evaluation now rejects entirely constant feature matrices. Old scores are marked superseded diagnostics; saved prediction and feature artifacts are retained.

## Rerun

Do not launch a full corpus expecting the current dense T5 implementation to fit. Resolve the inference-memory blocker and pass a longest-input preflight first. The preflight command below reproduces the H200 failure, not a successful run:

```powershell
uv run python -m scripts.collect_agent_benchmark runs/datasets/trace-full-context-preflight.jsonl --run-id trace-full-context-preflight-retry --gpu-id "NVIDIA H200" --timeout 1200
```

After full-context feasibility is established, collect each corresponding processed file into a fresh run ID, then evaluate with the existing wrapper:

```powershell
uv run python notebooks/evaluate_agent_benchmark.py --dataset trace --features runs/remote/trace-full-context --output notebooks/runs/notebook-agent-outcome-hpo/trace-full-context
uv run python notebooks/evaluate_agent_benchmark.py --dataset terminal-wrench --features runs/remote/terminal-wrench-full-context --group task_id --output notebooks/runs/notebook-agent-outcome-hpo/terminal-wrench-full-context
uv run python notebooks/evaluate_agent_benchmark.py --dataset terminal-wrench-stripped --features runs/remote/terminal-wrench-stripped-context --group task_id --output notebooks/runs/notebook-agent-outcome-hpo/terminal-wrench-stripped-context
```

These commands require successful feature runs, which do not exist yet. The historical exact-input cache must not be reused: full Terminal Wrench prompts are distinct. Eight-trial nested HPO and training-only operating point selection remain unchanged. No MALT examples enter these training or test sets.

Validation: 27 focused tests passed, including duplicate-group isolation in outer and inner folds and forwarding complete long prompts with `truncation_length=0`. The audit covers every normalized row. These checks do not substitute for successful GPU feature extraction.
