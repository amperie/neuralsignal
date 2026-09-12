# LongT5 activation collection

Model names containing `long-t5` or `longt5` use the sequence-to-sequence loader
in unquantized, 8-bit, and 4-bit modes. Already-loaded models are also recognized
by `config.model_type == "longt5"`, including locally named checkpoints.

`instrument_longt5` follows the T5 collector interface and selection flags:

- Encoder local or transient-global attention: Q/K/V/O projections and layer norm.
- Decoder self-attention and cross-attention: Q/K/V/O projections and layer norm.
- Feed-forward layers: `wi` or gated `wi_0`/`wi_1`, activation, and `wo`.
- Output head: always captured as `decoder.output`, matching T5's terminal hook.

LongT5 names include the stack, block index, layer class, and projection, for
example `encoder.LongT5Block.0.LongT5LayerLocalSelfAttention.LocalSelfAttention.q`.
As in the existing T5 implementation, embeddings are not hooked.

Transient-global attention reuses K/V modules for token and global-summary
positions. Global calls are recorded under separate collector identities named
`global_k` and `global_v`. The global input layer norm is also captured. Their
sequence dimension is the number of global blocks, not the number of tokens;
use appropriate global masks if later computing masked statistics over these
layers. Standard token attention masks must not be applied directly to them.
All forward and reset pre-hooks are removed by `deinstrument_model`.

Tests use tiny randomly initialized Hugging Face LongT5 models with both encoder
attention types and both feed-forward variants, padded batches, real generation,
and the real Collector. They verify unchanged generated tokens, distinct global
shapes, collector finalization, selection flags, loader routing, and hook cleanup.
Full pretrained GPU/quantized execution remains a separate integration check.

To use an existing collection config, set `model.model_name` to
`google/long-t5-tglobal-large` (or another LongT5 checkpoint). Input length, device,
and memory settings are unchanged by instrumentation support.
