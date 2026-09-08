output "bucket_name" {
  value = aws_s3_bucket.handoff.bucket
}

output "bucket_arn" {
  value = aws_s3_bucket.handoff.arn
}

output "aws_region" {
  value = var.aws_region
}

output "allowed_prefix" {
  value = "${local.allowed_prefix}/"
}

output "runpod_access_key_id" {
  value = var.create_runpod_user ? aws_iam_access_key.runpod_handoff[0].id : null
}

output "runpod_secret_access_key" {
  value     = var.create_runpod_user ? aws_iam_access_key.runpod_handoff[0].secret : null
  sensitive = true
}

output "runpod_secret_values" {
  value = var.create_runpod_user ? {
    AWS_ACCESS_KEY_ID      = aws_iam_access_key.runpod_handoff[0].id
    AWS_DEFAULT_REGION     = var.aws_region
    NEURALSIGNAL_S3_BUCKET = aws_s3_bucket.handoff.bucket
    AWS_SECRET_ACCESS_KEY  = "run terraform output -raw runpod_secret_access_key"
  } : null
}