# SDK Public Surface

## Goal

The public SDK should expose indirect-mode detection with minimal configuration.
Dataset creation, feature collection, training, MLflow, and RunPod orchestration
belong to internal modules and CLI commands.

## Public API

```python
from neuralsignal import NeuralSignal

ns = NeuralSignal(detectors=["sabotage"])
result = ns.evaluate(input_text, output_text)
```

Batch evaluation:

```python
results = ns.evaluate_batch([
    {"input": "question", "output": "answer"},
    {"input": "question", "output": "answer"},
])
```

## Public Classes

- `NeuralSignal`
- `DetectionResult`
- `BatchDetectionResult`

## Hidden/Internal Concepts

These should not be part of the primary public SDK:

- direct mode;
- raw scan persistence;
- dataset runners;
- S1 training;
- RunPod orchestration;
- backend plugin dispatch;
- global mutable config singleton.

## Configuration

Simple constructor:

```python
ns = NeuralSignal(
    detectors=["sabotage", "refusal"],
    judge_model="google/flan-t5-large",
    device="cuda",
)
```

Advanced config remains available but should be explicit:

```python
ns = NeuralSignal.from_config("configs/sdk.yaml")
```

## Results

```python
class DetectionResult:
    scores: dict[str, float]
    flagged: bool
    flags: dict[str, bool]
    metadata: dict
```

The result object should be easy to serialize:

```python
result.to_dict()
result.to_json()
```

## Removed Behavior

`generate()` and direct mode should be removed or kept private until implemented.
Public methods should not raise `NotImplementedError` for advertised features.
