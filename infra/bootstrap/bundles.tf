resource "aws_s3_bucket" "serving_bundles" {
  bucket        = var.serving_bundle_bucket_name
  force_destroy = var.serving_bundle_force_destroy

  tags = merge(local.tags, {
    Name    = var.serving_bundle_bucket_name
    Purpose = "versioned-serving-bundles"
  })
}

resource "aws_s3_bucket_versioning" "serving_bundles" {
  bucket = aws_s3_bucket.serving_bundles.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "serving_bundles" {
  bucket = aws_s3_bucket.serving_bundles.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "serving_bundles" {
  bucket = aws_s3_bucket.serving_bundles.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_ownership_controls" "serving_bundles" {
  bucket = aws_s3_bucket.serving_bundles.id

  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

data "aws_iam_policy_document" "serving_bundles" {
  statement {
    sid     = "DenyInsecureTransport"
    effect  = "Deny"
    actions = ["s3:*"]
    resources = [
      aws_s3_bucket.serving_bundles.arn,
      "${aws_s3_bucket.serving_bundles.arn}/*",
    ]

    principals {
      type        = "*"
      identifiers = ["*"]
    }

    condition {
      test     = "Bool"
      variable = "aws:SecureTransport"
      values   = ["false"]
    }
  }
}

resource "aws_s3_bucket_policy" "serving_bundles" {
  bucket = aws_s3_bucket.serving_bundles.id
  policy = data.aws_iam_policy_document.serving_bundles.json

  depends_on = [aws_s3_bucket_public_access_block.serving_bundles]
}
