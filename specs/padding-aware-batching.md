# Padding-Aware Batching

## Problem

Batch padding changes feature values when padded token positions are included in
activation aggregation. This makes the same prompt produce different S1 inputs
depending on which other prompts are in the batch.

The fix is to keep batching and make every token-position aggregation aware of
the tokenizer attention mask.

## Required Behavior

The same example should produce equivalent features when evaluated:

- alone;
- in a batch with same-length examples;
- in a batch with much longer examples.

Small floating-point differences are acceptable. Padding-driven feature drift is
not.

## Implementation Rule

Tokenize batches with padding and retain `attention_mask`:

```python
encoded = tokenizer(
    prompts,
    padding=True,
    truncation=True,
    return_tensors="pt",
)
```

Pass both tensors through generation:

```python
model.generate(
    input_ids=encoded["input_ids"],
    attention_mask=encoded["attention_mask"],
    pad_token_id=tokenizer.eos_token_id,
)
```

Attach the attention mask to each generation/activation container and pass it to
the feature pipeline.

## Masked Aggregation

Mean pooling:

```python
def masked_mean(x, attention_mask):
    mask = attention_mask.to(dtype=x.dtype, device=x.device).unsqueeze(-1)
    return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
```

Variance:

```python
def masked_var(x, attention_mask):
    mean = masked_mean(x, attention_mask).unsqueeze(1)
    mask = attention_mask.to(dtype=x.dtype, device=x.device).unsqueeze(-1)
    denom = mask.sum(dim=1).clamp_min(1)
    return (((x - mean) ** 2) * mask).sum(dim=1) / denom
```

Any feature set that reduces over sequence positions must use masked operations.

## Throughput Optimization

Length bucketing is still useful because it reduces wasted padded tokens:

```text
examples -> tokenize length estimate -> buckets -> padded batches -> features
```

Length bucketing improves speed and memory usage. It must not be required for
correctness.

## Regression Test

Create a test with a short prompt and a long prompt:

1. evaluate short prompt alone;
2. evaluate short prompt batched with long prompt;
3. assert feature vectors are close;
4. assert detector scores are close.

This test should run against deterministic generation settings.
