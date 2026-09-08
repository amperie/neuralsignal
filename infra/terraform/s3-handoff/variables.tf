variable "aws_profile" {
  description = "Local AWS CLI profile used to create the handoff bucket."
  type        = string
  default     = "qc"
}

variable "aws_region" {
  description = "AWS region for the handoff bucket."
  type        = string
  default     = "us-east-1"
}

variable "bucket_name" {
  description = "Globally unique S3 bucket name. If null, uses neuralsignal-runpod-handoff-{account_id}."
  type        = string
  default     = null
}

variable "allowed_prefix" {
  description = "Object prefix RunPod can access."
  type        = string
  default     = "feature-runs/"
}

variable "run_retention_days" {
  description = "Days to retain remote handoff run objects."
  type        = number
  default     = 30
}

variable "create_runpod_user" {
  description = "Create an IAM user/access key for RunPod feature upload/download."
  type        = bool
  default     = true
}

variable "force_destroy" {
  description = "Allow Terraform to delete a non-empty bucket. Keep false unless intentionally tearing down."
  type        = bool
  default     = false
}

