# Terraform resources

Source: [infra/terraform/s3-handoff](../infra/terraform/s3-handoff/).
See its [setup instructions](../infra/terraform/s3-handoff/README.md) for commands.
This describes configuration, not a fresh inspection of deployed AWS resources.

## Resources and defaults

- Private handoff bucket `neuralsignal-runpod-handoff-<account_id>` (override `bucket_name`).
- Optional state bucket `neuralsignal-terraform-state-<account_id>`, created by default.
- Optional platform artifact bucket `neuralsignal-platform-artifacts-<account_id>`, created by default.
- Optional scoped RunPod IAM user/key and platform IAM user/key, both created by default.
- AWS profile `qc`, region `us-east-1`, configurable via variables.

All buckets use AES256 server-side encryption, versioning, and full public-access
blocks. State bucket has `prevent_destroy`. Force-destroy flags default false.
The handoff lifecycle expires current and noncurrent prefix objects after
`run_retention_days` (30 by default), and aborts multipart uploads after seven days.
Platform/state buckets are not governed by that temporary-data expiry rule.

## Access

RunPod policy includes list/multipart operations and read/write/**delete** access
under `allowed_prefix` (default `feature-runs/`). Platform policy provides bucket
listing, handoff-prefix object operations, and platform-artifact object operations.
These are runtime credentials, separate from the profile used to apply Terraform.

## State and outputs

`versions.tf` configures a **local** backend at `terraform.tfstate`. Creating the
state bucket does not activate it as a backend. `terraform_backend_hcl` describes
S3 settings, but the backend type must be changed before those settings can be
used. The existing helper's `-Migrate` switch does not make that code change.

Outputs include bucket/region/prefix, RunPod and platform access-key identifiers,
sensitive secret keys, `runpod_secret_values`, `platform_secret_values`, and
`terraform_backend_hcl`. There is no `runpod_secrets_template` output.
The launcher reads RunPod bucket/key outputs when available; local boto3 credentials
must still be configured independently. No MinIO or MLflow service is provisioned.
