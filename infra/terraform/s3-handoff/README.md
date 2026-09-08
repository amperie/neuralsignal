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
terraform output -raw terraform_backend_hcl > backend.hcl
terraform init -backend-config=backend.hcl -migrate-state
```

After migration, use the same backend config for future runs. Do not commit `backend.hcl` if it contains account-specific state settings you do not want shared.
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



