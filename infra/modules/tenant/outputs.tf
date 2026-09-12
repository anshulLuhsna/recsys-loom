output "instance_id" {
  description = "EC2 instance ID."
  value       = aws_instance.this.id
}

output "public_ip" {
  description = "Stable Elastic IP address."
  value       = aws_eip.this.public_ip
}

output "public_dns_name" {
  description = "Optional Route53 record name, or null when DNS is disabled."
  value       = try(aws_route53_record.this[0].fqdn, null)
}

output "ssm_start_session_command" {
  description = "Session Manager command; no inbound SSH is required."
  value       = "aws ssm start-session --target ${aws_instance.this.id}"
}

output "instance_role_arn" {
  description = "Runtime IAM role ARN."
  value       = aws_iam_role.instance.arn
}

output "log_group_names" {
  description = "CloudWatch log groups populated by the agent."
  value       = { for key, group in aws_cloudwatch_log_group.this : key => group.name }
}

output "lab_fault_volume_id" {
  description = "Optional controlled-fault EBS volume ID."
  value       = try(aws_ebs_volume.lab_fault[0].id, null)
}

output "deployment_notes" {
  description = "Non-secret post-apply facts for deployment automation."
  value = {
    application_root = "/opt/recsys-loom"
    deployment_log   = "/opt/recsys-loom/logs/deploy.log"
    lab_fault_mount  = var.enable_lab_fault_volume ? "/mnt/lab-faults" : null
    imdsv2_required  = true
    ssh_enabled      = length(var.ssh_allowed_cidrs) > 0
    candidate_user   = var.candidate_ssh_public_key != "" ? "candidate" : null
  }
}
