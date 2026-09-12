provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = var.project_name
      Tenant      = var.tenant_name
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}
