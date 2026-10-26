**Superseded diagnostic run — not a full-trajectory benchmark.** These artifacts used only the first 512 tokens. TRACE also contains an exact duplicate pair (`trajectory_0516`, `trajectory_0517`) that crossed the original CV folds. Do not use these scores as validated full-trajectory results. See [the full-context rerun status](agent_full_context_status.md).

# NeuralSignal public agent benchmark results

**Terminal Wrench exposes a context-window failure in the unchanged MALT configuration.** Every original and stripped trajectory has the same first 512 tokens, entirely before the first assistant turn. Its results measure a detector receiving no distinguishing behavior evidence.

| Dataset | Split | Evaluated rows | AUC | PR-AUC (AP) | Macro F1 | Published baseline |
|---|---|---:|---:|---:|---:|---|
| terminal-wrench-full | task-held-out GroupShuffleSplit | 1758 | 0.5000 | 0.5313 | 0.3191 | GPT-5.4 AUC 0.9679 |
| terminal-wrench-stripped | task-held-out GroupShuffleSplit | 1758 | 0.5000 | 0.5313 | 0.3191 | GPT-5.4 AUC 0.9168 |
| trace-full | trajectory-stratified 5-fold CV; NOT task-held-out | 517 | 0.6798 | 0.6536 | 0.6491 | GPT-5.2 macro F1 0.45 |

## Operating points

Thresholds were selected using training-only out-of-fold predictions. Achieved test catch rates need not meet the training targets.

| Dataset | Training catch target | Held-out catch | Held-out false alarm |
|---|---:|---:|---:|
| terminal-wrench-full | 95% | 100.00% | 100.00% |
| terminal-wrench-full | 97% | 100.00% | 100.00% |
| terminal-wrench-full | 99% | 100.00% | 100.00% |
| terminal-wrench-stripped | 95% | 100.00% | 100.00% |
| terminal-wrench-stripped | 97% | 100.00% | 100.00% |
| terminal-wrench-stripped | 99% | 100.00% | 100.00% |
| trace-full | 95% | 95.15% | 83.13% |
| trace-full | 97% | 97.01% | 89.16% |
| trace-full | 99% | 98.13% | 93.98% |

## Breakdowns

### terminal-wrench-full

| Slice | Rows | Hacks | AUC | PR-AUC (AP) | Macro F1 |
|---|---:|---:|---:|---:|---:|
| claude-opus-4.6 | 638 | 296 | 0.5000 | 0.4639 | 0.3490 |
| gemini-3.1-pro | 620 | 365 | 0.5000 | 0.5887 | 0.2914 |
| gpt-5.4 | 500 | 273 | 0.5000 | 0.5460 | 0.3122 |
| hacks_vs_baselines_only | 1512 | 934 | 0.5000 | 0.6177 | 0.2766 |

### terminal-wrench-stripped

| Slice | Rows | Hacks | AUC | PR-AUC (AP) | Macro F1 |
|---|---:|---:|---:|---:|---:|
| claude-opus-4.6 | 638 | 296 | 0.5000 | 0.4639 | 0.3490 |
| gemini-3.1-pro | 620 | 365 | 0.5000 | 0.5887 | 0.2914 |
| gpt-5.4 | 500 | 273 | 0.5000 | 0.5460 | 0.3122 |
| hacks_vs_baselines_only | 1512 | 934 | 0.5000 | 0.6177 | 0.2766 |

### trace-full

| Slice | Rows | Hacks | AUC | PR-AUC (AP) | Macro F1 |
|---|---:|---:|---:|---:|---:|
| 1.1.1 | 305 | 56 | 0.7933 | 0.4209 | 0.6118 |
| 1.1.2 | 318 | 69 | 0.6891 | 0.3395 | 0.5993 |
| 1.1.3 | 300 | 51 | 0.7140 | 0.2840 | 0.5924 |
| 1.2.1 | 285 | 36 | 0.6171 | 0.1993 | 0.5114 |
| 1.2.2 | 270 | 21 | 0.4991 | 0.0813 | 0.4473 |
| 1.2.3 | 283 | 34 | 0.6133 | 0.1548 | 0.4947 |
| 1.3.1 | 274 | 25 | 0.5687 | 0.1109 | 0.4887 |
| 1.3.2 | 270 | 21 | 0.6718 | 0.1452 | 0.4958 |
| 1.4.1 | 270 | 21 | 0.7143 | 0.1814 | 0.4958 |
| 1.4.2 | 273 | 24 | 0.7356 | 0.1727 | 0.4980 |

## TRACE fold variation

| Fold | Rows | AUC | Macro F1 |
|---|---:|---:|---:|
| 0 | 104 | 0.7452 | 0.7299 |
| 1 | 104 | 0.6385 | 0.5856 |
| 2 | 103 | 0.6626 | 0.6691 |
| 3 | 103 | 0.6562 | 0.6302 |
| 4 | 103 | 0.6992 | 0.6251 |

## XGBoost latency

Notebook microbenchmark: 10 warmups, 100 timed calls per batch size, using DataFrame inputs on the local Xeon E5-2695 v4. Evaluation jobs overlapped, so these are indicative timings rather than a controlled hardware comparison. Feature extraction is excluded.

| Dataset | Batch size | Median ms per call | P95 ms per call |
|---|---:|---:|---:|
| terminal-wrench-full | 1 | 550.98 | 621.06 |
| terminal-wrench-full | 8 | 557.78 | 620.21 |
| terminal-wrench-full | 32 | 575.27 | 671.83 |
| terminal-wrench-stripped | 1 | 550.60 | 610.85 |
| terminal-wrench-stripped | 8 | 555.58 | 665.38 |
| terminal-wrench-stripped | 32 | 568.99 | 676.70 |
| trace-full | 1 | 574.51 | 653.54 |
| trace-full | 8 | 558.58 | 686.40 |
| trace-full | 32 | 567.00 | 690.48 |

## Interpretation and limits

- Terminal Wrench: 7,200 included rows per variant, 3,632 hacks and 3,568 normals across 331 tasks. Normals comprise 2,352 baselines and 1,216 legitimate attacker solves; 1,441 no-reward attempts are excluded. The original and stripped variants share identical task assignments.
- TRACE: 517 rows, 268 hacks and 249 benign trajectories. No task/scenario field exists in the release. Five-fold trajectory CV cannot establish generalization to unseen tasks. Category slices compare each code against all benign trajectories and overlap for multi-label rows.
- All feature columns match MALT, including decoder features: 7,128 total, 4,320 decoder. Batch size 1 and the 512-token limit are unchanged. These are not full-context behavioral scores.
- Terminal Wrench input reuse was allowed only after 24 representatives across source models, outcome classes, and variants yielded bit-identical features. All trajectories and statistical weights remain in the feature datasets.
- The published Terminal Wrench baseline uses hacks versus baselines only and evaluates a full-corpus LLM judge. Our main negative population additionally includes attacker legitimate solves; the baseline-only slice still evaluates held-out tasks only.
- TRACE compares supervised cross-validation against a zero-shot judge baseline. A numerical difference is not evidence of superiority under an equivalent protocol.
- Operating points use training-only predictions. The original MALT notebook is untouched; its test-selected operating points are not reproduced as deployment estimates.
- XGBoost latency CSVs measure prediction on existing features, excluding feature extraction. Zero-shot MALT transfer and separate no-reward scoring were not run.

## Sources and reproducibility

- [Terminal Wrench release and monitor baseline](https://github.com/few-sh/terminal-wrench)
- [TRACE dataset](https://huggingface.co/datasets/PatronusAI/trace-dataset)
- [TRACE paper](https://arxiv.org/html/2601.20103v1)
- [Protocol, revisions, changed components, and rerun commands](agent_benchmark_runs.md)
- Complete metrics, predictions, folds, operating points, breakdowns, HPO trials, models, and latency outputs: `notebooks/runs/notebook-agent-outcome-hpo/<dataset>/`.
