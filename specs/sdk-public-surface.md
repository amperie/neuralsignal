# Public SDK

The package exports `NeuralSignal`, `DetectionResult`, and `BatchDetectionResult`.
Implementation: [sdk/client.py](../neuralsignal/sdk/client.py) and
[sdk/results.py](../neuralsignal/sdk/results.py).

The evaluator is supplied by the caller:

```python
from neuralsignal import NeuralSignal

def evaluator(examples, detectors):
    # Demonstration only; replace with actual inference.
    return [{"scores": {name: 0.1 for name in detectors}} for _ in examples]

ns = NeuralSignal(detectors=["sabotage"], evaluator=evaluator, threshold=0.5)
result = ns.evaluate("question", "response")
print(result.to_json())
batch = ns.evaluate_batch([{"input": "question", "output": "response"}])
print(batch.to_dict())
```

The callback receives `(examples, detectors)` and must return exactly one result
row per input, in order. Mismatched result counts raise an error. An empty batch
returns no results without invoking it. Evaluating a nonempty batch without an
evaluator raises `RuntimeError`.

Rows contain `scores`, optional `flags`, and optional `metadata`. If flags are
omitted they are computed as score >= threshold; `flagged` is any true flag.
Individual results provide `to_dict()` and `to_json()`; batches provide
`to_dict()` and a `results` list.

Constructor fields `judge_model` and `device` are stored but do not load models.
Automatic judge/S1 inference, `from_config`, direct generation, dataset creation,
training, and orchestration are not public SDK capabilities. Use the CLI for the
implemented experiment workflows.
