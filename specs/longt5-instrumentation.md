# T5 and LongT5 instrumentation

Name-based loading recognizes `long-t5`/`longt5` as sequence-to-sequence models
for unquantized, int8, and int4 settings. Already-loaded T5 and LongT5 objects
are recognized by `config.model_type`, so local checkpoint names need not contain
the architecture. Name-based loading still requires a recognized model name.

LongT5 captures encoder local/transient-global attention Q/K/V/O and layer norm,
decoder self/cross-attention Q/K/V/O and layer norm, feed-forward input/output
projections and activation, and the terminal output head as `decoder.output`.
Selection flags control encoder, decoder, attention, and feed-forward hooks;
the terminal head is always captured. Embeddings are not hooked by these paths.

Names include stack, block, layer class, and projection. Transient-global K/V
calls use separate collector identities (`global_k`, `global_v`) for summary
positions. Their sequence lengths are global-block counts, not token counts.
The reset pre-hooks and forward hooks are removed by `deinstrument_model`.

Collector empty per-layer maps fall back to the global zone size. Layer filters
work without per-layer overrides. Input-only topology uses inputs. Re-reduction
rejects noninteger zone ratios rather than rounding them.

Real tiny CPU tests cover both LongT5 attention modes, both feed-forward variants,
unchanged generated tokens, global shapes, collector finalization, flags and cleanup.
The T5 smoke-settings integration exercises both enabled feature processors.
Pretrained GPU/quantized execution and semantic global/decoder masking remain
separate checks. See [padding limitations](padding-aware-batching.md) and
[current verification](../docs/current-state.md).
