# AWS Profile

This Terraform stack should use the local AWS CLI profile:

```text
qc
```

`zz-profile.auto.tfvars` overrides the original profile value.

Authenticate before running Terraform:

```powershell
aws sso login --profile qc
```
