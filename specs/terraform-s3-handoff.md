# Terraform S3 Handoff Infrastructure

## Goal

Create the minimal cloud infrastructure needed for RunPod-to-local feature
dataset handoff:

```text
RunPod feature collector -> S3 bucket -> local sync -> local MinIO -> local S1 training
```

The bucket is a transient handoff store for feature shards, manifests, and logs.
It is not the long-term source of truth after a dataset has been curated into
local MinIO.

## Security Position

Do not make the bucket public read/write.

Even if feature data is not sensitive, public write creates avoidable problems:

- anyone can upload junk or malicious data;
- anyone can delete or overwrite objects if write permissions are broad enough;
- anyone can run up storage and egress costs;
- training jobs can accidentally consume poisoned feature shards;
- public bucket policies are easy to forget after the project grows.

Use least-privilege IAM users or roles instead.

## Terraform Layout

```text
infra/
  terraform/
    s3-handoff/
      main.tf
      variables.tf
      outputs.tf
      versions.tf
      README.md
```

Keep this folder small. It should create only the handoff bucket and the IAM
permissions needed by RunPod and the local sync CLI.

## Resources

Terraform should create:

- one private S3 bucket;
- bucket versioning;
- server-side encryption;
- public access block;
- lifecycle cleanup for old runs;
- IAM policy for feature handoff access;
- IAM user or access key for RunPod when roles are not available;
- outputs for bucket name, region, and policy/user identifiers.

## Bucket Layout

```text
s3://{bucket}/
  feature-runs/{run_id}/
    manifest.json
    features/part-*.parquet
    logs/worker.log
```

## Permissions

RunPod needs scoped access to feature-run objects:

```text
s3:PutObject
s3:GetObject
s3:ListBucket
s3:AbortMultipartUpload
s3:ListBucketMultipartUploads
s3:ListMultipartUploadParts
```

Local sync needs:

```text
s3:GetObject
s3:ListBucket
```

If the local CLI also deletes failed or expired runs manually, put delete access
behind a separate admin profile instead of the RunPod key.

## Public Access Block

The bucket must enable:

```text
block_public_acls       = true
block_public_policy     = true
ignore_public_acls      = true
restrict_public_buckets = true
```

## Lifecycle Policy

Remote handoff data should expire automatically after local sync and MinIO
curation. Default lifecycle:

```text
expire feature-runs/* after 30 days
abort incomplete multipart uploads after 7 days
```

Keep the expiration configurable.

## Credentials

Terraform may output the access key id, but should treat the secret access key
as sensitive.

RunPod credentials should be stored outside git:

```text
runpod.secrets
.env.runpod
```

Expected env vars:

```text
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_DEFAULT_REGION
NEURALSIGNAL_S3_BUCKET
NEURALSIGNAL_S3_ENDPOINT_URL
```

For AWS S3, `NEURALSIGNAL_S3_ENDPOINT_URL` can be omitted. For S3-compatible
providers, it is required.

## Terraform Variables

```hcl
variable "bucket_name" {
  type = string
}

variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "run_retention_days" {
  type    = number
  default = 30
}

variable "create_runpod_user" {
  type    = bool
  default = true
}

variable "allowed_prefix" {
  type    = string
  default = "feature-runs/"
}
```

## Outputs

```hcl
output "bucket_name" {}
output "bucket_arn" {}
output "runpod_access_key_id" {}
output "runpod_secret_access_key" {
  sensitive = true
}
```

## Local Setup Flow

```text
cd infra/terraform/s3-handoff
terraform init
terraform apply
terraform output -raw runpod_secret_access_key
```

Then write the credentials into an ignored local secrets file used by the RunPod
launcher.

## Alternative: Presigned URLs

Presigned upload URLs are safer than public write, but they complicate resumable
multi-shard uploads. Start with scoped IAM credentials. Add presigned URLs later
only if credential distribution becomes a real problem.

## Non-Goals

- no public bucket;
- no MLflow infrastructure;
- no local MinIO provisioning;
- no VPC/network complexity;
- no permanent dataset registry in remote S3.
