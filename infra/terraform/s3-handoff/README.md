# NeuralSignal S3 Handoff Terraform

This Terraform stack creates a private S3 bucket for RunPod feature handoff and,
by default, a scoped IAM user/access key for RunPod jobs.

It uses this local AWS CLI profile:

```text
qc
```

## Setup

If the profile does not exist yet, create it locally:

```powershell
aws configure sso --profile qc
```

Then authenticate:

`powershell
aws sso login --profile qc
`$block

## Apply

```powershell
terraform init
terraform plan
terraform apply
```


## Terraform State Bucket

This stack also creates a separate private S3 bucket for Terraform state by default:

```text
neuralsignal-terraform-state-{account_id}
```

That bucket is encrypted, versioned, blocks public access, and has `prevent_destroy` enabled. Keep it separate from the RunPod handoff bucket. The handoff bucket is temporary data transport; the state bucket is infrastructure history.

Terraform cannot use a backend bucket before the bucket exists. Bootstrap in two steps:

1. Apply this stack once with local state.
2. Copy the generated backend config and run `terraform init -migrate-state`.

```powershell
scripts/terraform_write_backend_hcl.ps1 -Migrate
```

Equivalent manual commands from this directory:

```powershell
terraform output -raw terraform_backend_hcl > backend.hcl
terraform init -backend-config=backend.hcl -migrate-state
```

After migration, use the same backend config for future runs. Do not commit `backend.hcl` if it contains account-specific state settings you do not want shared.

## Platform Runtime IAM User

This stack also creates a separate platform runtime IAM user by default:

```text
neuralsignal-platform
```

Use this for the local/platform process, not the AWS root credentials. It is scoped to the NeuralSignal S3 resources created here:

- read/write/delete under the RunPod handoff prefix
- read/write/delete in the platform artifacts bucket

After apply, retrieve values with:

```powershell
terraform output platform_secret_values
terraform output -raw platform_secret_access_key
```

The platform environment should include:

```text
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1
NEURALSIGNAL_S3_BUCKET=...
NEURALSIGNAL_PLATFORM_BUCKET=...
```
## RunPod Secrets

After apply, write the RunPod credentials into an ignored secrets file:

```powershell
terraform output runpod_secrets_template
terraform output -raw runpod_secret_access_key
```

The final `runpod.secrets` values should include:

```text
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1
NEURALSIGNAL_S3_BUCKET=...
```

Do not commit `runpod.secrets`.

## Notes

- The bucket blocks all public access.
- Feature-run objects expire after 30 days by default.
- The RunPod IAM user can read/write only under `feature-runs/`.
- `RUNPOD_API_KEY` should stay local and must not be forwarded into the pod.





