variable "aws_region" {
  description = "AWS region for state and container repositories."
  type        = string
  default     = "ap-south-1"
}

variable "project_name" {
  description = "Project tag and resource-name component."
  type        = string
  default     = "recsys-loom"
}

variable "tenant_name" {
  description = "Tenant tag for shared bootstrap resources."
  type        = string
  default     = "shared"
}

variable "state_bucket_name" {
  description = "Globally unique S3 bucket name for Terraform state."
  type        = string

  validation {
    condition     = can(regex("^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$", var.state_bucket_name))
    error_message = "state_bucket_name must be a valid lowercase S3 bucket name."
  }
}

variable "state_key_prefix" {
  description = "Key prefix the GitHub CI role may use for state and lock files."
  type        = string
  default     = "recsys-loom"

  validation {
    condition     = var.state_key_prefix != "" && !startswith(var.state_key_prefix, "/") && !endswith(var.state_key_prefix, "/")
    error_message = "state_key_prefix must be non-empty and have no leading or trailing slash."
  }
}

variable "state_force_destroy" {
  description = "Allow deletion of the state bucket and versions. Keep false outside disposable accounts."
  type        = bool
  default     = false
}

variable "serving_bundle_bucket_name" {
  description = "Globally unique S3 bucket for versioned runtime bundles."
  type        = string

  validation {
    condition     = can(regex("^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$", var.serving_bundle_bucket_name))
    error_message = "serving_bundle_bucket_name must be a valid lowercase S3 bucket name."
  }
}

variable "serving_bundle_force_destroy" {
  description = "Allow deletion of serving bundles. Enable only for disposable labs."
  type        = bool
  default     = false
}

variable "deployment_config_prefix" {
  description = "Serving-bundle bucket prefix used for immutable deployment configuration."
  type        = string
  default     = "ops"

  validation {
    condition     = var.deployment_config_prefix != "" && !startswith(var.deployment_config_prefix, "/") && !endswith(var.deployment_config_prefix, "/")
    error_message = "deployment_config_prefix must have no leading or trailing slash."
  }
}

variable "github_repository" {
  description = "GitHub owner/repository allowed to assume the CI role."
  type        = string

  validation {
    condition     = can(regex("^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$", var.github_repository))
    error_message = "github_repository must have owner/repository form."
  }
}

variable "github_branch" {
  description = "Single GitHub branch allowed to assume the CI role."
  type        = string
  default     = "main"

  validation {
    condition     = var.github_branch != "" && !startswith(var.github_branch, "refs/")
    error_message = "github_branch must be a non-empty branch name without a refs/ prefix."
  }
}

variable "github_oidc_thumbprints" {
  description = "TLS certificate thumbprints accepted for GitHub's OIDC provider."
  type        = list(string)
  default     = ["6938fd4d98bab03faadb97b34396831e3780aea1"]

  validation {
    condition     = length(var.github_oidc_thumbprints) > 0 && alltrue([for value in var.github_oidc_thumbprints : can(regex("^[0-9a-fA-F]{40}$", value))])
    error_message = "Each GitHub OIDC thumbprint must be a 40-character SHA-1 hex string."
  }
}

variable "github_ci_role_name" {
  description = "IAM role name assumed by GitHub Actions."
  type        = string
  default     = "recsys-loom-github-ci"
}

variable "ecr_repository_names" {
  description = "ECR repositories created for deployable images."
  type        = set(string)
  default = [
    "recsys-loom-recommendation",
    "recsys-loom-search",
  ]

  validation {
    condition     = length(var.ecr_repository_names) > 0 && alltrue([for name in var.ecr_repository_names : can(regex("^[a-z0-9]+(?:[._/-][a-z0-9]+)*$", name))])
    error_message = "ECR names must use lowercase repository-name syntax."
  }
}

variable "ecr_max_image_count" {
  description = "Maximum tagged images retained per ECR repository."
  type        = number
  default     = 20

  validation {
    condition     = var.ecr_max_image_count >= 5
    error_message = "ecr_max_image_count must be at least 5."
  }
}

variable "extra_tags" {
  description = "Additional tags. Required tags cannot be overridden."
  type        = map(string)
  default     = {}
}
