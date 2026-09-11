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


variable "create_state_bucket" {
  description = "Create a separate S3 bucket that can be used as the Terraform remote state backend."
  type        = bool
  default     = true
}

variable "state_bucket_name" {
  description = "Globally unique S3 bucket name for Terraform state. If null, uses neuralsignal-terraform-state-{account_id}."
  type        = string
  default     = null
}

variable "state_bucket_force_destroy" {
  description = "Allow Terraform to delete a non-empty Terraform state bucket. Keep false unless intentionally tearing down state storage."
  type        = bool
  default     = false
}

variable "create_platform_user" {
  description = "Create an IAM user/access key for the NeuralSignal platform runtime."
  type        = bool
  default     = true
}

variable "platform_user_name" {
  description = "IAM user name for the NeuralSignal platform runtime."
  type        = string
  default     = "neuralsignal-platform"
}

variable "create_platform_bucket" {
  description = "Create a private S3 bucket for durable platform artifacts."
  type        = bool
  default     = true
}

variable "platform_bucket_name" {
  description = "Globally unique S3 bucket name for platform artifacts. If null, uses neuralsignal-platform-artifacts-{account_id}."
  type        = string
  default     = null
}

variable "platform_bucket_force_destroy" {
  description = "Allow Terraform to delete a non-empty platform artifact bucket. Keep false unless intentionally tearing down."
  type        = bool
  default     = false
}
