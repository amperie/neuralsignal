# NeuralSignal — CLAUDE.md

## Project Overview

NeuralSignal is a Python SDK for detecting behavioral anomalies in LLM outputs through real-time model instrumentation. It wraps a "judge" LLM (default: `google/flan-t5-large`) with PyTorch hooks to capture internal activation tensors ("scans"), featurizes them, and runs trained S1 classifiers to score behaviors such as hallucination, toxicity, and identity attack.

## Repository Layout

```
neuralsignal/
├── neuralsignal/
│   ├── sdk/
│   │   ├── neuralsignal.py          # Primary public API (SDK + DetectionResults classes)
│   │   └── neuralsignal_sdk.yaml    # Default config: evaluation mode, detectors, backend
│   ├── core/
│   │   ├── modules/
│   │   │   ├── model_instrumentation.py  # load_model(), generate_from_batch()
│   │   │   ├── collector.py              # Hooks that capture activation tensors
│   │   │   ├── detector.py              # Detector + DetectionResults (internal)
│   │   │   ├── generation_instance.py   # GenerationInstance — scan data container
│   │   │   ├── neuralsignal_config.py   # NeuralSignalConfig singleton (sdk_config)
│   │   │   ├── ns_model.py              # NSModel — sklearn/xgb wrapper
│   │   │   ├── s1_model.py              # S1Model — trained anomaly classifier
│   │   │   ├── prompting.py             # wrap_with_prompt() template engine
│   │   │   ├── tensors.py               # subtract_scans() and tensor utilities
│   │   │   └── utils.py                 # generate_uuid() and misc helpers
│   │   ├── exceptions/
│   │   │   └── NSAbortLLM.py            # Custom exception to abort generation early
│   │   └── feature_sets/
│   │       ├── feature_processor.py
│   │       ├── feature_set_base.py
│   │       ├── feature_set_factory.py
│   │       ├── feature_set_layer_distribution.py
│   │       ├── feature_set_logit_lens.py
│   │       ├── feature_set_tuned_lens.py
│   │       ├── feature_set_t_f_diff.py
│   │       ├── feature_set_zones.py
│   │       └── feature_utils.py
│   ├── backend/
│   │   ├── ns_backend.py           # NSBackend facade (plugin dispatch)
│   │   ├── ns_be_impl_v1.py        # NeuralSignal v1 backend (MLflow + Mongo)
│   │   ├── mongo_backend.py        # MongoDB backend
│   │   ├── file_backend.py         # File-based backend
│   │   └── backend_util.py
│   ├── datasets/
│   │   ├── dataset.py
│   │   ├── dataset_creator.py
│   │   ├── dataset_definitions.py
│   │   ├── dataset_runner.py
│   │   └── s1_trainer.py           # Trains S1 classifiers
│   ├── automation/
│   │   ├── dataset_automation.py
│   │   └── dataset_automation_core.py
│   └── tests/
│       ├── sdk_tests.py
│       ├── dataset_run.py
│       └── test_latency.py
├── notebooks/                      # Jupyter notebooks for exploration
├── pyproject.toml                  # uv/hatchling build config
├── .python-version                 # 3.13
└── main.py                         # Stub entrypoint (not wired up)
```

## Architecture

### Evaluation Modes

| Mode | Description | Status |
|------|-------------|--------|
| `indirect` | Feeds LLM input+output through a judge model; captures activations from the judge | Implemented |
| `direct` | Instruments the application's own LLM in real time | Not yet implemented |

### Data Flow (Indirect Mode)

```
User input/output
      │
      ▼
wrap_with_prompt()          ← detector-specific prompt template
      │
      ▼
generate_from_batch()       ← runs judge model (flan-t5-large) with hooks
      │
      ▼
Collector hooks capture tensors per layer
      │
      ▼
GenerationInstance          ← holds scan data + detections
      │
      ├──► Detector.detect() ──► S1Model.predict_proba() ──► DetectionResults
      │
      └──► NSBackend.save_scan()  ← optional persistence
```

### Key Classes

| Class | File | Role |
|-------|------|------|
| `SDK` | `sdk/neuralsignal.py` | Public API entry point |
| `DetectionResults` (public) | `sdk/neuralsignal.py` | Returned to callers |
| `Detector` | `core/modules/detector.py` | Wraps S1 model + prompt config |
| `DetectionResults` (internal) | `core/modules/detector.py` | Per-detection score container |
| `GenerationInstance` | `core/modules/generation_instance.py` | Scan + metadata per generation |
| `NSBackend` | `backend/ns_backend.py` | Backend plugin facade |
| `NeuralSignalConfig` | `core/modules/neuralsignal_config.py` | Singleton config (yaml-backed) |
| `S1Model` | `core/modules/s1_model.py` | Trained behavior classifier |

## Configuration

Primary config file: `neuralsignal/sdk/neuralsignal_sdk.yaml`

Key sections:
- `evaluation_mode`: `indirect` or `direct`
- `indirect_config`: judge model name, device, quantization
- `indirect_instrumentation_config`: which layer types to instrument, zone sizes
- `backend_config`: backend type (`noop`, `mongo`, `file_backend`, `neuralsignal_v1`), Mongo URL, MLflow URI
- `detectors`: list of named detectors with prompts, S1 model paths, and enabled flags

Config can be overridden at `SDK()` initialization by passing a `config` dict.

## Backend Types

| Type | Description |
|------|-------------|
| `noop` | Discards all writes (useful for testing) |
| `mongo` | Direct MongoDB storage |
| `file_backend` | Local file-based storage |
| `neuralsignal_v1` | Full implementation: MongoDB + MLflow model registry |

## Detector Types

| Type | Description |
|------|-------------|
| `normal` | Single prompt → S1 classifier |
| `scan_delta` | Two prompts separated by `<<--separator-->>`, then `subtract_scans()` compares activations |

## Running Tests

```bash
uv run pytest neuralsignal/tests/
```

## Dependencies

Managed via `uv`. Key packages:
- `torch`, `transformers`, `bitsandbytes` — LLM inference + quantization
- `datasets`, `huggingface-hub` — dataset loading
- `scikit-learn`, `xgboost`, `hyperopt` — S1 classifier training
- `pymongo` — MongoDB backend
- `mlflow` — experiment tracking + model registry
- `pandas` — data manipulation

Install: `uv sync`

## Known Issues / TODOs

- **Batch padding bug**: tokenizer pads batches to the largest prompt size, causing S1 score variance by batch size (`_evaluate_batch_output`)
- **Data repeat on OOM retry**: when a smaller batch is retried after OOM, previously-processed items may be re-saved to the backend
- **Detector caching**: `evaluate_indirect()` reloads `Detector` objects on every call; should cache them
- **Config race condition**: `sdk_config` singleton loads from the default yaml at import time; if a different config path is passed later, both could coexist
- **Direct mode**: `__init_direct()` raises `NotImplementedError`
- `generate()` stub also raises `NotImplementedError`
