variable "project_name" {
  description = "Project tag and resource-name component."
  type        = string

  validation {
    condition     = can(regex("^[a-z0-9][a-z0-9-]{1,30}[a-z0-9]$", var.project_name))
    error_message = "project_name must be 3-32 lowercase letters, numbers, or hyphens."
  }
}

variable "tenant_name" {
  description = "Tenant tag and resource-name component."
  type        = string

  validation {
    condition     = can(regex("^[a-z0-9][a-z0-9-]{0,20}[a-z0-9]$", var.tenant_name))
    error_message = "tenant_name must be 2-22 lowercase letters, numbers, or hyphens."
  }
}

variable "environment" {
  description = "Deployment environment tag and resource-name component."
  type        = string

  validation {
    condition     = contains(["development", "staging", "production"], var.environment)
    error_message = "environment must be development, staging, or production."
  }
}

variable "vpc_cidr" {
  description = "IPv4 CIDR for the tenant VPC."
  type        = string
  default     = "10.20.0.0/16"

  validation {
    condition     = can(cidrnetmask(var.vpc_cidr))
    error_message = "vpc_cidr must be a valid IPv4 CIDR."
  }
}

variable "public_subnet_cidr" {
  description = "IPv4 CIDR for the public subnet."
  type        = string
  default     = "10.20.10.0/24"

  validation {
    condition     = can(cidrnetmask(var.public_subnet_cidr))
    error_message = "public_subnet_cidr must be a valid IPv4 CIDR."
  }
}

variable "availability_zone" {
  description = "Availability Zone for EC2 and EBS. The first available AZ is used when null."
  type        = string
  default     = null
}

variable "ssh_allowed_cidrs" {
  description = "IPv4 CIDRs allowed to reach SSH. Empty by default; use SSM Session Manager."
  type        = list(string)
  default     = []

  validation {
    condition     = alltrue([for cidr in var.ssh_allowed_cidrs : can(cidrnetmask(cidr))])
    error_message = "Every ssh_allowed_cidrs entry must be a valid IPv4 CIDR."
  }
}

variable "key_name" {
  description = "Optional existing EC2 key-pair name. Required only when SSH ingress is enabled."
  type        = string
  default     = null
}

variable "candidate_ssh_public_key" {
  description = "Optional temporary SSH public key installed for the disposable interview user."
  type        = string
  default     = ""

  validation {
    condition     = var.candidate_ssh_public_key == "" || can(regex("^ssh-(ed25519|rsa) [A-Za-z0-9+/=]+(?: .*)?$", var.candidate_ssh_public_key))
    error_message = "candidate_ssh_public_key must be empty or an OpenSSH ed25519/RSA public key."
  }
}

variable "ami_id" {
  description = "Optional x86_64 Ubuntu AMI override. The latest Canonical Ubuntu 24.04 AMI is used when null."
  type        = string
  default     = null
}

variable "instance_type" {
  description = "x86 EC2 instance type."
  type        = string
  default     = "t3.medium"

  validation {
    condition     = can(regex("^(t3|t3a|m5|m5a|m6i|c5|c6i)\\.", var.instance_type))
    error_message = "instance_type must be an allowed x86 instance family."
  }
}

variable "root_volume_size_gib" {
  description = "Encrypted gp3 root volume size in GiB."
  type        = number
  default     = 35

  validation {
    condition     = var.root_volume_size_gib >= 30 && var.root_volume_size_gib <= 40
    error_message = "root_volume_size_gib must be between 30 and 40 GiB."
  }
}

variable "enable_lab_fault_volume" {
  description = "Attach a small encrypted EBS volume at /mnt/lab-faults."
  type        = bool
  default     = false
}

variable "lab_fault_volume_size_gib" {
  description = "Size of the optional encrypted lab-fault volume."
  type        = number
  default     = 8

  validation {
    condition     = var.lab_fault_volume_size_gib >= 1 && var.lab_fault_volume_size_gib <= 16
    error_message = "lab_fault_volume_size_gib must be between 1 and 16 GiB."
  }
}

variable "lab_fault_device_name" {
  description = "EC2 API attachment name. Nitro instances expose the volume as an NVMe device."
  type        = string
  default     = "/dev/sdf"
}

variable "ecr_repository_arns" {
  description = "ECR repositories from which the instance may pull images."
  type        = list(string)
  default     = []

  validation {
    condition     = alltrue([for arn in var.ecr_repository_arns : can(regex("^arn:[^:]+:ecr:[^:]+:[0-9]{12}:repository/.+$", arn))])
    error_message = "Every ecr_repository_arns entry must be an ECR repository ARN."
  }
}

variable "serving_bundle_bucket_arn" {
  description = "S3 bucket ARN containing immutable serving bundles. Empty disables S3 access."
  type        = string
  default     = ""

  validation {
    condition     = var.serving_bundle_bucket_arn == "" || can(regex("^arn:[^:]+:s3:::[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$", var.serving_bundle_bucket_arn))
    error_message = "serving_bundle_bucket_arn must be empty or a valid S3 bucket ARN."
  }
}

variable "serving_bundle_prefix" {
  description = "Object-key prefix the instance may list and read."
  type        = string
  default     = "serving/"

  validation {
    condition     = !startswith(var.serving_bundle_prefix, "/") && var.serving_bundle_prefix != ""
    error_message = "serving_bundle_prefix must be non-empty and must not start with a slash."
  }
}

variable "deployment_config_prefix" {
  description = "Object-key prefix containing immutable deployment configuration archives."
  type        = string
  default     = "ops/"

  validation {
    condition     = !startswith(var.deployment_config_prefix, "/") && var.deployment_config_prefix != ""
    error_message = "deployment_config_prefix must be non-empty and must not start with a slash."
  }
}

variable "log_retention_days" {
  description = "CloudWatch Logs retention."
  type        = number
  default     = 14

  validation {
    condition     = contains([1, 3, 5, 7, 14, 30, 60, 90, 120, 150, 180, 365, 400, 545, 731, 1096, 1827, 2192, 2557, 2922, 3288, 3653], var.log_retention_days)
    error_message = "log_retention_days must be a CloudWatch Logs-supported retention value."
  }
}

variable "alarm_action_arns" {
  description = "SNS or other action ARNs notified by CloudWatch alarms."
  type        = list(string)
  default     = []
}

variable "cpu_alarm_threshold_percent" {
  description = "Sustained EC2 CPU utilization threshold."
  type        = number
  default     = 85
}

variable "cpu_credit_alarm_threshold" {
  description = "Low CPU credit balance threshold for burstable instances."
  type        = number
  default     = 10
}

variable "memory_alarm_threshold_percent" {
  description = "CloudWatch agent memory-used threshold."
  type        = number
  default     = 90
}

variable "swap_alarm_threshold_percent" {
  description = "CloudWatch agent swap-used threshold."
  type        = number
  default     = 20
}

variable "disk_alarm_threshold_percent" {
  description = "CloudWatch agent disk-used threshold."
  type        = number
  default     = 85
}

variable "inode_free_alarm_threshold" {
  description = "Minimum free inode count before alarming."
  type        = number
  default     = 10000
}

variable "monitored_filesystem_type" {
  description = "Filesystem dimension emitted for the root and lab-fault disk metrics."
  type        = string
  default     = "ext4"
}

variable "route53_record" {
  description = "Optional Route53 A record pointing to the EIP."
  type = object({
    zone_id = string
    name    = string
    ttl     = optional(number, 300)
  })
  default = null
}

variable "extra_tags" {
  description = "Additional tags. Required project, tenant, and environment tags cannot be overridden."
  type        = map(string)
  default     = {}
}
