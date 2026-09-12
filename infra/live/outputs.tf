output "instance_id" {
  description = "EC2 instance ID."
  value       = module.tenant.instance_id
}

output "public_ip" {
  description = "Stable Elastic IP address."
  value       = module.tenant.public_ip
}

output "public_dns_name" {
  description = "Optional Route53 record."
  value       = module.tenant.public_dns_name
}

output "ssm_start_session_command" {
  description = "Command for shell access without opening SSH."
  value       = module.tenant.ssm_start_session_command
}

output "log_group_names" {
  description = "CloudWatch log groups used by the instance agent."
  value       = module.tenant.log_group_names
}

output "deployment_notes" {
  description = "Non-secret deployment paths and security settings."
  value       = module.tenant.deployment_notes
}
