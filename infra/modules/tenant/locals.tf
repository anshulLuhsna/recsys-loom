data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_ami" "ubuntu_x86" {
  most_recent = true
  owners      = ["099720109477"]

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*"]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }

  filter {
    name   = "root-device-type"
    values = ["ebs"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

locals {
  name              = "${var.project_name}-${var.tenant_name}-${var.environment}"
  availability_zone = coalesce(var.availability_zone, data.aws_availability_zones.available.names[0])
  ami_id            = coalesce(var.ami_id, data.aws_ami.ubuntu_x86.id)

  required_tags = {
    Project     = var.project_name
    Tenant      = var.tenant_name
    Environment = var.environment
    ManagedBy   = "Terraform"
  }

  tags = merge(var.extra_tags, local.required_tags)

  serving_bundle_prefix    = trim(var.serving_bundle_prefix, "/")
  deployment_config_prefix = trim(var.deployment_config_prefix, "/")
  readable_s3_prefixes     = [
    local.serving_bundle_prefix,
    local.deployment_config_prefix,
  ]

  log_group_names = {
    application = "/${local.name}/application"
    deployment  = "/${local.name}/deployment"
    docker      = "/${local.name}/docker"
    system      = "/${local.name}/system"
  }
}
