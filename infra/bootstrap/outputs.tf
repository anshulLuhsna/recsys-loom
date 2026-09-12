output "state_bucket_name" {
  description = "Versioned, encrypted Terraform state bucket."
  value       = aws_s3_bucket.terraform_state.id
}

output "serving_bundle_bucket_name" {
  description = "Private versioned bucket for release bundles."
  value       = aws_s3_bucket.serving_bundles.id
}

output "serving_bundle_bucket_arn" {
  description = "Bucket ARN passed to each tenant runtime role."
  value       = aws_s3_bucket.serving_bundles.arn
}

output "live_backend_configuration" {
  description = "Non-secret values to place in infra/live/backend.hcl."
  value = {
    bucket       = aws_s3_bucket.terraform_state.id
    region       = var.aws_region
    encrypt      = true
    use_lockfile = true
    key_pattern  = "${var.state_key_prefix}/TENANT/ENVIRONMENT/terraform.tfstate"
  }
}

output "github_ci_role_arn" {
  description = "OIDC role for state, image publishing, and tagged-instance deployment."
  value       = aws_iam_role.github_ci.arn
}

output "github_oidc_subject" {
  description = "Exact GitHub repository and environment subject trusted by the CI role."
  value       = "repo:${split("/", var.github_repository)[0]}@${var.github_repository_owner_id}/${split("/", var.github_repository)[1]}@${var.github_repository_id}:environment:${var.github_environment}"
}

output "ecr_repository_arns" {
  description = "Repository ARNs to pass to the live tenant module for pull-only access."
  value       = { for name, repository in aws_ecr_repository.this : name => repository.arn }
}

output "ecr_repository_urls" {
  description = "Repository URLs used by image build and deployment automation."
  value       = { for name, repository in aws_ecr_repository.this : name => repository.repository_url }
}
