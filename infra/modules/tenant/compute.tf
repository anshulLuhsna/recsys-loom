data "aws_region" "current" {}

resource "aws_ebs_volume" "lab_fault" {
  count = var.enable_lab_fault_volume ? 1 : 0

  availability_zone = local.availability_zone
  encrypted         = true
  size              = var.lab_fault_volume_size_gib
  type              = "gp3"

  tags = merge(local.tags, {
    Name    = "${local.name}-lab-faults"
    Purpose = "controlled-fault-lab"
  })
}

locals {
  cloudwatch_agent_config = templatefile("${path.module}/../../../monitoring/cloudwatch-agent.json.tftpl", {
    application_log_group_name = aws_cloudwatch_log_group.this["application"].name
    deployment_log_group_name  = aws_cloudwatch_log_group.this["deployment"].name
    docker_log_group_name      = aws_cloudwatch_log_group.this["docker"].name
    system_log_group_name      = aws_cloudwatch_log_group.this["system"].name
  })

  user_data = templatefile("${path.module}/../../../monitoring/cloud-init.sh.tftpl", {
    aws_region                      = data.aws_region.current.name
    cloudwatch_agent_config_base64  = base64encode(local.cloudwatch_agent_config)
    candidate_ssh_public_key_base64 = base64encode(var.candidate_ssh_public_key)
    lab_fault_enabled               = tostring(var.enable_lab_fault_volume)
    lab_fault_volume_id             = try(aws_ebs_volume.lab_fault[0].id, "")
    lab_fault_device_name           = var.lab_fault_device_name
  })
}

resource "aws_instance" "this" {
  ami                         = local.ami_id
  instance_type               = var.instance_type
  availability_zone           = local.availability_zone
  subnet_id                   = aws_subnet.public.id
  vpc_security_group_ids      = [aws_security_group.instance.id]
  associate_public_ip_address = true
  key_name                    = var.key_name
  iam_instance_profile        = aws_iam_instance_profile.this.name
  monitoring                  = true
  ebs_optimized               = true
  user_data                   = local.user_data

  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 1
    instance_metadata_tags      = "disabled"
  }

  root_block_device {
    delete_on_termination = true
    encrypted             = true
    volume_size           = var.root_volume_size_gib
    volume_type           = "gp3"

    tags = merge(local.tags, {
      Name = "${local.name}-root"
    })
  }

  dynamic "credit_specification" {
    for_each = startswith(var.instance_type, "t3.") || startswith(var.instance_type, "t3a.") ? [1] : []

    content {
      cpu_credits = "standard"
    }
  }

  tags = merge(local.tags, {
    Name = local.name
  })

  lifecycle {
    precondition {
      condition = (
        length(var.ssh_allowed_cidrs) == 0
        || var.key_name != null
        || var.candidate_ssh_public_key != ""
      )
      error_message = "key_name or candidate_ssh_public_key must be set when SSH ingress is enabled."
    }
  }

  depends_on = [
    aws_iam_role_policy.runtime,
    aws_iam_role_policy_attachment.ssm,
    aws_route.internet,
  ]
}

resource "aws_volume_attachment" "lab_fault" {
  count = var.enable_lab_fault_volume ? 1 : 0

  device_name = var.lab_fault_device_name
  volume_id   = aws_ebs_volume.lab_fault[0].id
  instance_id = aws_instance.this.id
}

resource "aws_eip" "this" {
  domain   = "vpc"
  instance = aws_instance.this.id

  tags = merge(local.tags, {
    Name = "${local.name}-eip"
  })

  depends_on = [aws_internet_gateway.this]
}

resource "aws_route53_record" "this" {
  count = var.route53_record == null ? 0 : 1

  zone_id = var.route53_record != null ? var.route53_record.zone_id : ""
  name    = var.route53_record != null ? var.route53_record.name : ""
  type    = "A"
  ttl     = var.route53_record != null ? var.route53_record.ttl : 300
  records = [aws_eip.this.public_ip]
}
