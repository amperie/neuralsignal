# MALT monitoring: current experiment

The current adapter uses `source: malt` with `metr-evals/malt-public`, split
`public`, config `default`. It emits complete recorded API histories and candidate
next responses as ordered message lines. It does not turn them into a compact
narrative or reconstruct an independent live trajectory runner.

The default judge prompt asks about behavior evidenced by history and the next
response. Experimental scaffold instructions matching the known exact marker are
excluded from the prompt and preserved in metadata. Labels and run condition are
kept as metadata, not automatically inserted into the default prompt.

S1 supervision is at run level: completion means are averaged into sample means,
then run means; the target is membership in `run_labels`. The default training
config selects `sabotage`. There is no implemented custom taxonomy for credential
behavior, persistence, or lateral movement, and no automatic label mapping to
such categories.

## Evidence

Saved MALT fixtures and local pooling tests pass. The eight-example remote smoke
run returned zero feature rows, so it did not establish a GPU feature dataset or
MALT performance result. Current fixes and real small-model CPU tests are
summarized in [current state](current-state.md).

## Next experiment boundary

Rebuild the worker image, run a fresh eight-example smoke test, and inspect both
feature sets before full collection. Training needs complete runs and enough
positive/negative runs for stratification. Mean pooling is the implemented
baseline; no prefix evaluation, live intervention, early-warning effectiveness,
calibration, or benchmark quality claim has been validated.

[Adapter details](../specs/dataset-imports.md) ·
[Training](../specs/local-s1-mlflow.md) · [Evaluation](../specs/s1-evaluation.md)
