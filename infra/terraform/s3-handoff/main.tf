provider "aws" {
  profile = var.aws_profile
  region  = var.aws_region
}

data "aws_caller_identity" "current" {}

locals {
  bucket_name       = var.bucket_name != null ? var.bucket_name : "neuralsignal-runpod-handoff-${data.aws_caller_identity.current.account_id}"
  state_bucket_name = var.state_bucket_name != null ? var.state_bucket_name : "neuralsignal-terraform-state-${data.aws_caller_identity.current.account_id}"
  allowed_prefix    = trimsuffix(var.allowed_prefix, "/")
}

resource "aws_s3_bucket" "handoff" {
  bucket        = local.bucket_name
  force_destroy = var.force_destroy

  tags = {
    Project = "neuralsignal"
    Purpose = "runpod-feature-handoff"
  }
}

resource "aws_s3_bucket_public_access_block" "handoff" {
  bucket = aws_s3_bucket.handoff.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "handoff" {
  bucket = aws_s3_bucket.handoff.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "handoff" {
  bucket = aws_s3_bucket.handoff.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "handoff" {
  bucket = aws_s3_bucket.handoff.id

  rule {
    id     = "expire-feature-runs"
    status = "Enabled"

    filter {
      prefix = "${local.allowed_prefix}/"
    }

    expiration {
      days = var.run_retention_days
    }

    noncurrent_version_expiration {
      noncurrent_days = var.run_retention_days
    }

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}

data "aws_iam_policy_document" "runpod_handoff" {
  statement {
    sid = "ListHandoffPrefix"

    actions = [
      "s3:ListBucket",
      "s3:ListBucketMultipartUploads"
    ]

    resources = [aws_s3_bucket.handoff.arn]

    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values   = ["${local.allowed_prefix}/*"]
    }
  }

  statement {
    sid = "ReadWriteHandoffObjects"

    actions = [
      "s3:AbortMultipartUpload",
      "s3:GetObject",
      "s3:ListMultipartUploadParts",
      "s3:PutObject"
    ]

    resources = ["${aws_s3_bucket.handoff.arn}/${local.allowed_prefix}/*"]
  }
}

resource "aws_iam_policy" "runpod_handoff" {
  count       = var.create_runpod_user ? 1 : 0
  name        = "neuralsignal-runpod-handoff-${data.aws_caller_identity.current.account_id}"
  description = "Scoped access for NeuralSignal RunPod feature handoff."
  policy      = data.aws_iam_policy_document.runpod_handoff.json
}

resource "aws_iam_user" "runpod_handoff" {
  count = var.create_runpod_user ? 1 : 0
  name  = "neuralsignal-runpod-handoff"

  tags = {
    Project = "neuralsignal"
    Purpose = "runpod-feature-handoff"
  }
}

resource "aws_iam_user_policy_attachment" "runpod_handoff" {
  count      = var.create_runpod_user ? 1 : 0
  user       = aws_iam_user.runpod_handoff[0].name
  policy_arn = aws_iam_policy.runpod_handoff[0].arn
}

resource "aws_iam_access_key" "runpod_handoff" {
  count = var.create_runpod_user ? 1 : 0
  user  = aws_iam_user.runpod_handoff[0].name
}


resource "aws_s3_bucket" "terraform_state" {
  count         = var.create_state_bucket ? 1 : 0
  bucket        = local.state_bucket_name
  force_destroy = var.state_bucket_force_destroy

  lifecycle {
    prevent_destroy = true
  }

  tags = {
    Project = "neuralsignal"
    Purpose = "terraform-state"
  }
}

resource "aws_s3_bucket_public_access_block" "terraform_state" {
  count  = var.create_state_bucket ? 1 : 0
  bucket = aws_s3_bucket.terraform_state[0].id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "terraform_state" {
  count  = var.create_state_bucket ? 1 : 0
  bucket = aws_s3_bucket.terraform_state[0].id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "terraform_state" {
  count  = var.create_state_bucket ? 1 : 0
  bucket = aws_s3_bucket.terraform_state[0].id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}
