terraform {
  required_version = ">= 1.10.0, < 2.0.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.80"
    }
  }

  # Supply bucket, key, region, encrypt, and use_lockfile with:
  # terraform init -backend-config=backend.hcl
  backend "s3" {}
}
