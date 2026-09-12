# AWS profile

The module defaults to local AWS CLI profile `qc`; `zz-profile.auto.tfvars` also
sets `aws_profile = "qc"`. Region defaults to `us-east-1`.

From the repository root:

```bash
aws sso login --profile qc
terraform -chdir=infra/terraform/s3-handoff plan
```

Configure the SSO profile first if it does not exist. To use another profile,
change/override `aws_profile`; an empty value uses the normal provider credential
chain. This profile applies to Terraform administration, not automatic credential
forwarding into RunPod. [Setup and state](README.md).
