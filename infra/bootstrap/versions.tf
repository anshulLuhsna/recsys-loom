terraform {
  required_version = ">= 1.10.0, < 2.0.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.80"
    }
  }

  # Create the state bucket with an initial local apply, then migrate with:
  # terraform init -migrate-state -backend-config=backend.hcl
  backend "s3" {}
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = var.project_name
      Tenant      = var.tenant_name
      Environment = "bootstrap"
      ManagedBy   = "Terraform"
    }
  }
}
