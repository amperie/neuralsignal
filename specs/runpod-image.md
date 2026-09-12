# RunPod Image

## Goal

The RunPod image provides a reproducible GPU environment for feature collection.
It does not train S1 models and does not require MLflow.

## Responsibilities

The image must be able to:

- install NeuralSignal v2;
- authenticate to Hugging Face;
- load the judge model;
- collect activations;
- compute configured feature sets;
- write parquet feature shards;
- upload shards and manifest files to S3-compatible storage.

## Entrypoint

```text
python -m neuralsignal.remote.job \
  --run-id <run_id> \
  --config <config_path> \
  --s3-output-uri <s3_uri>
```

## Required Environment Variables

```text
HF_TOKEN
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_DEFAULT_REGION
NEURALSIGNAL_S3_ENDPOINT_URL
```

## Optional Environment Variables

```text
HF_HOME
TRANSFORMERS_CACHE
NEURALSIGNAL_DATASET_CACHE
NEURALSIGNAL_RUN_WORKDIR
```

## Local Paths Inside Pod

```text
/workspace/neuralsignal
/workspace/neuralsignal-runs/{run_id}
/workspace/cache/huggingface
```

## Image Contract

- Python version matches the project.
- Dependencies are installed with `uv`.
- CUDA/PyTorch versions are pinned.
- The job process exits non-zero on unrecoverable failure.
- The job writes a final manifest state before exit when possible.
- The image never stores credentials in built layers.

## Caching

Judge model and dataset caches may be mounted as RunPod volumes. The job must
still work without a warm cache.
