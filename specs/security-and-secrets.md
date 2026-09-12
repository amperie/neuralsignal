# Security and sensitive outputs

Use ignored local environment/secrets files for credentials. Never put real
credentials in feature YAML, dataset metadata, committed documentation, or MLflow
extra params. The code does not universally scrub these fields before persistence.

## Implemented protections

- Launch dry-run recursively redacts keys containing token/secret/password/access-key/API-key patterns.
- RunPod API keys are removed from worker environment payloads.
- Local log configuration suppresses routine provider request chatter below WARNING.
- Shard path resolution rejects absolute/parent-traversal paths and resolved escapes.
- ZIP and shard checksums detect content mismatch.
- Terraform blocks public bucket access and scopes handoff object permissions.

These are not blanket redaction or authenticity guarantees. Provider warnings,
exceptions, arbitrary config fields, or dataset contents can still contain
sensitive data; checksum sidecars are not signatures. The old smoke log contains
provider request URLs from before noise suppression. Review artifacts before
sharing them.

Feature rows store raw input/output text plus metadata. Raw activation scans are
not written by the CLI, but this does not make feature bundles free of source text.
Remote manifests copy dataset/materialization config entries; avoid embedding
credentials in those mappings.

Terraform state contains managed IAM key material even when outputs are marked
sensitive. The currently configured backend is local; protect and back up state.
S3 versioning means deletion can leave noncurrent data until lifecycle expiry.
See [Terraform setup](../infra/terraform/s3-handoff/README.md) and
[credential routing](required-secrets.md).
