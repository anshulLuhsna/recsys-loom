variable "aws_region" {
  description = "AWS region for this deployment."
  type        = string
  default     = "ap-south-1"
}

variable "project_name" {
  description = "Project tag and resource-name component."
  type        = string
}

variable "tenant_name" {
  description = "Tenant tag and resource-name component."
  type        = string
}

variable "environment" {
  description = "Deployment environment."
  type        = string
}

variable "vpc_cidr" {
  description = "IPv4 CIDR for the tenant VPC."
  type        = string
  default     = "10.20.0.0/16"
}

variable "public_subnet_cidr" {
  description = "IPv4 CIDR for the public subnet."
  type        = string
  default     = "10.20.10.0/24"
}

variable "availability_zone" {
  description = "Optional Availability Zone override."
  type        = string
  default     = null
}

variable "ssh_allowed_cidrs" {
  description = "IPv4 CIDRs allowed to reach SSH. Empty keeps SSH closed."
  type        = list(string)
  default     = []
}

variable "key_name" {
  description = "Optional existing EC2 key-pair name."
  type        = string
  default     = null
}

variable "candidate_ssh_public_key" {
  description = "Optional temporary SSH public key for the disposable interview user."
  type        = string
  default     = ""
}

variable "ami_id" {
  description = "Optional x86_64 Ubuntu AMI override."
  type        = string
  default     = null
}

variable "instance_type" {
  description = "x86 EC2 instance type."
  type        = string
  default     = "t3.medium"
}

variable "root_volume_size_gib" {
  description = "Encrypted gp3 root volume size."
  type        = number
  default     = 35
}

variable "enable_lab_fault_volume" {
  description = "Attach and mount an encrypted lab-fault volume."
  type        = bool
  default     = false
}

variable "lab_fault_volume_size_gib" {
  description = "Optional lab-fault volume size."
  type        = number
  default     = 8
}

variable "ecr_repository_arns" {
  description = "ECR repositories from which the instance may pull."
  type        = list(string)
  default     = []
}

variable "serving_bundle_bucket_arn" {
  description = "S3 bucket ARN containing immutable serving bundles."
  type        = string
  default     = ""
}

variable "serving_bundle_prefix" {
  description = "Serving-bundle key prefix."
  type        = string
  default     = "serving/"
}

variable "deployment_config_prefix" {
  description = "Deployment-configuration archive key prefix."
  type        = string
  default     = "ops/"
}

variable "log_retention_days" {
  description = "CloudWatch Logs retention."
  type        = number
  default     = 14
}

variable "alarm_action_arns" {
  description = "SNS or other CloudWatch alarm action ARNs."
  type        = list(string)
  default     = []
}

variable "cpu_alarm_threshold_percent" {
  description = "Sustained CPU threshold."
  type        = number
  default     = 85
}

variable "cpu_credit_alarm_threshold" {
  description = "Low burstable CPU credit threshold."
  type        = number
  default     = 10
}

variable "memory_alarm_threshold_percent" {
  description = "Sustained memory threshold."
  type        = number
  default     = 90
}

variable "swap_alarm_threshold_percent" {
  description = "Sustained swap-use threshold."
  type        = number
  default     = 20
}

variable "disk_alarm_threshold_percent" {
  description = "Disk usage threshold."
  type        = number
  default     = 85
}

variable "inode_free_alarm_threshold" {
  description = "Minimum root-filesystem free inode count."
  type        = number
  default     = 10000
}

variable "route53_record" {
  description = "Optional Route53 A record."
  type = object({
    zone_id = string
    name    = string
    ttl     = optional(number, 300)
  })
  default = null
}

variable "extra_tags" {
  description = "Additional non-authoritative tags."
  type        = map(string)
  default     = {}
}
