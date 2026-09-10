resource "aws_iam_openid_connect_provider" "github" {
  url             = "https://token.actions.githubusercontent.com"
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = var.github_oidc_thumbprints

  tags = merge(local.tags, {
    Name = "${var.project_name}-github-actions"
  })
}

data "aws_caller_identity" "current" {}

data "aws_region" "current" {}

data "aws_iam_policy_document" "github_ci_assume_role" {
  statement {
    sid     = "GitHubRepositoryBranchOnly"
    effect  = "Allow"
    actions = ["sts:AssumeRoleWithWebIdentity"]

    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.github.arn]
    }

    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }

    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:sub"
      values   = ["repo:${var.github_repository}:ref:refs/heads/${var.github_branch}"]
    }
  }
}

resource "aws_iam_role" "github_ci" {
  name                 = var.github_ci_role_name
  description          = "GitHub OIDC role scoped to state, ECR, S3 config, and SSM deployment"
  assume_role_policy   = data.aws_iam_policy_document.github_ci_assume_role.json
  max_session_duration = 3600

  tags = merge(local.tags, {
    Name = var.github_ci_role_name
  })
}

data "aws_iam_policy_document" "github_ci" {
  statement {
    sid       = "ListProjectState"
    effect    = "Allow"
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.terraform_state.arn]

    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values = [
        var.state_key_prefix,
        "${var.state_key_prefix}/*",
      ]
    }
  }

  statement {
    sid    = "ReadWriteProjectStateAndLocks"
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
    ]
    resources = ["${aws_s3_bucket.terraform_state.arn}/${var.state_key_prefix}/*"]
  }

  statement {
    sid       = "ReleaseProjectStateLocks"
    effect    = "Allow"
    actions   = ["s3:DeleteObject"]
    resources = ["${aws_s3_bucket.terraform_state.arn}/${var.state_key_prefix}/*.tflock"]
  }

  statement {
    sid       = "ListDeploymentConfiguration"
    effect    = "Allow"
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.serving_bundles.arn]

    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values = [
        var.deployment_config_prefix,
        "${var.deployment_config_prefix}/*",
      ]
    }
  }

  statement {
    sid     = "PublishDeploymentConfiguration"
    effect  = "Allow"
    actions = ["s3:PutObject"]
    resources = [
      "${aws_s3_bucket.serving_bundles.arn}/${var.deployment_config_prefix}/*",
    ]
  }

  statement {
    sid       = "AuthenticateToEcr"
    effect    = "Allow"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }

  statement {
    sid    = "PublishProjectImages"
    effect = "Allow"
    actions = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:BatchGetImage",
      "ecr:CompleteLayerUpload",
      "ecr:GetDownloadUrlForLayer",
      "ecr:InitiateLayerUpload",
      "ecr:PutImage",
      "ecr:UploadLayerPart",
    ]
    resources = [for repository in aws_ecr_repository.this : repository.arn]
  }

  statement {
    sid     = "RunDeploymentDocument"
    effect  = "Allow"
    actions = ["ssm:SendCommand"]
    resources = [
      "arn:aws:ssm:${data.aws_region.current.name}::document/AWS-RunShellScript",
    ]
  }

  statement {
    sid     = "DeployToProjectInstances"
    effect  = "Allow"
    actions = ["ssm:SendCommand"]
    resources = [
      "arn:aws:ec2:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:instance/*",
    ]

    condition {
      test     = "StringEquals"
      variable = "ssm:resourceTag/Project"
      values   = [var.project_name]
    }
  }

  statement {
    sid       = "ReadDeploymentCommandResult"
    effect    = "Allow"
    actions   = ["ssm:GetCommandInvocation"]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "github_ci" {
  name   = "${var.project_name}-delivery"
  role   = aws_iam_role.github_ci.id
  policy = data.aws_iam_policy_document.github_ci.json
}
