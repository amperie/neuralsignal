# S3 handoff infrastructure

This module creates private handoff storage and optional platform/state buckets
and IAM runtime users. The checked-in backend is **local**, at `terraform.tfstate`.
This document describes the module; it does not assert the deployed account state.

## Setup and apply

Run these commands from the repository root, using the configured AWS profile:

```bash
aws configure sso --profile qc
aws sso login --profile qc
terraform -chdir=infra/terraform/s3-handoff init
terraform -chdir=infra/terraform/s3-handoff plan
terraform -chdir=infra/terraform/s3-handoff apply
```

Only run `aws configure sso` if the profile needs setup. The module defaults to
`qc`/`us-east-1`; `zz-profile.auto.tfvars` also selects `qc`. Review the plan before
applying. See [PROFILE.md](PROFILE.md).

## Resources

The handoff bucket is encrypted, versioned and private. Objects under
`feature-runs/` expire after 30 days by default, including noncurrent versions;
incomplete multipart uploads expire after seven days. The RunPod IAM policy
allows scoped list, read, write, delete and multipart operations.

By default the stack also creates a separate platform-artifact bucket/platform
runtime IAM user, and a state bucket with `prevent_destroy`. All public access is
blocked. Force-destroy settings default false. Runtime policies do not grant
Terraform administration access. [Variable reference](variables.tf).

## Credentials

Inspect non-secret templates/identifiers with:

```bash
terraform -chdir=infra/terraform/s3-handoff output runpod_secret_values
terraform -chdir=infra/terraform/s3-handoff output platform_secret_values
```

Actual sensitive outputs are `runpod_secret_access_key` and
`platform_secret_access_key`; retrieve them only into a protected local setup.
Use ignored files for runtime credentials. Worker values include
`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`, and
`NEURALSIGNAL_S3_BUCKET`. Keep `RUNPOD_API_KEY` local.

The launcher reads Terraform RunPod outputs for the worker environment. Configure
its local AWS download/delete credentials separately. See
[credential routing](../../../specs/required-secrets.md).

## Backend status

The provisioned state bucket is not currently the active backend. Protect and back
up local `terraform.tfstate`, which includes IAM key material. Do not delete local
state on the assumption it is already stored remotely.

`terraform_backend_hcl` and
[scripts/terraform_write_backend_hcl.ps1](../../../scripts/terraform_write_backend_hcl.ps1)
can produce S3 backend parameters, but they do not change `backend "local"` in
[versions.tf](versions.tf). The helper's `-Migrate` mode is not a complete migration
workflow for the current configuration. Moving state to S3 requires a deliberate
backend configuration change and migration with suitable credentials; it has not
been performed by this documentation update.

[Resource summary](../../../specs/terraform-s3-handoff.md) ·
[Transport layout](../../../specs/s3-minio-storage.md)
