data "aws_iam_policy_document" "ec2_assume_role" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "instance" {
  name_prefix        = "${local.name}-"
  description        = "Runtime role for ${local.name}"
  assume_role_policy = data.aws_iam_policy_document.ec2_assume_role.json

  tags = merge(local.tags, {
    Name = "${local.name}-instance"
  })
}

resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.instance.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

data "aws_iam_policy_document" "runtime" {
  statement {
    sid       = "PublishHostMetrics"
    effect    = "Allow"
    actions   = ["cloudwatch:PutMetricData"]
    resources = ["*"]

    condition {
      test     = "StringEquals"
      variable = "cloudwatch:namespace"
      values   = ["CWAgent"]
    }
  }

  statement {
    sid    = "WriteConfiguredLogStreams"
    effect = "Allow"
    actions = [
      "logs:CreateLogStream",
      "logs:PutLogEvents",
    ]
    resources = [for group in aws_cloudwatch_log_group.this : "${group.arn}:*"]
  }

  statement {
    sid       = "DescribeConfiguredLogStreams"
    effect    = "Allow"
    actions   = ["logs:DescribeLogStreams"]
    resources = [for group in aws_cloudwatch_log_group.this : group.arn]
  }

  dynamic "statement" {
    for_each = length(var.ecr_repository_arns) > 0 ? [1] : []

    content {
      sid       = "AuthenticateToEcr"
      effect    = "Allow"
      actions   = ["ecr:GetAuthorizationToken"]
      resources = ["*"]
    }
  }

  dynamic "statement" {
    for_each = length(var.ecr_repository_arns) > 0 ? [1] : []

    content {
      sid    = "PullDeploymentImages"
      effect = "Allow"
      actions = [
        "ecr:BatchCheckLayerAvailability",
        "ecr:BatchGetImage",
        "ecr:GetDownloadUrlForLayer",
      ]
      resources = var.ecr_repository_arns
    }
  }

  dynamic "statement" {
    for_each = var.serving_bundle_bucket_arn != "" ? [1] : []

    content {
      sid       = "ListServingBundles"
      effect    = "Allow"
      actions   = ["s3:ListBucket"]
      resources = [var.serving_bundle_bucket_arn]

      condition {
        test     = "StringLike"
        variable = "s3:prefix"
        values = flatten([
          for prefix in local.readable_s3_prefixes : [
            prefix,
            "${prefix}/*",
          ]
        ])
      }
    }
  }

  dynamic "statement" {
    for_each = var.serving_bundle_bucket_arn != "" ? [1] : []

    content {
      sid     = "ReadServingBundles"
      effect  = "Allow"
      actions = ["s3:GetObject"]
      resources = [
        for prefix in local.readable_s3_prefixes :
        "${var.serving_bundle_bucket_arn}/${prefix}/*"
      ]
    }
  }
}

resource "aws_iam_role_policy" "runtime" {
  name   = "${local.name}-runtime"
  role   = aws_iam_role.instance.id
  policy = data.aws_iam_policy_document.runtime.json
}

resource "aws_iam_instance_profile" "this" {
  name_prefix = "${local.name}-"
  role        = aws_iam_role.instance.name

  tags = merge(local.tags, {
    Name = "${local.name}-instance"
  })
}
