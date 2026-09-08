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

```powershell
aws sso login --profile qc
```

## Apply

```powershell
terraform init
terraform plan
terraform apply
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

