resource "aws_cloudwatch_log_group" "this" {
  for_each = local.log_group_names

  name              = each.value
  retention_in_days = var.log_retention_days

  tags = merge(local.tags, {
    Name = "${local.name}-${each.key}"
  })
}

locals {
  common_alarm_arguments = {
    actions_enabled = length(var.alarm_action_arns) > 0
    alarm_actions   = var.alarm_action_arns
    ok_actions      = var.alarm_action_arns
  }
}

resource "aws_cloudwatch_metric_alarm" "instance_status" {
  alarm_name          = "${local.name}-instance-status"
  alarm_description   = "EC2 instance or system status check failed."
  namespace           = "AWS/EC2"
  metric_name         = "StatusCheckFailed"
  statistic           = "Maximum"
  period              = 60
  evaluation_periods  = 2
  datapoints_to_alarm = 2
  threshold           = 0
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "breaching"
  actions_enabled     = local.common_alarm_arguments.actions_enabled
  alarm_actions       = local.common_alarm_arguments.alarm_actions
  ok_actions          = local.common_alarm_arguments.ok_actions

  dimensions = {
    InstanceId = aws_instance.this.id
  }

  tags = merge(local.tags, {
    Name = "${local.name}-instance-status"
  })
}

resource "aws_cloudwatch_metric_alarm" "cpu_high" {
  alarm_name          = "${local.name}-cpu-high"
  alarm_description   = "EC2 CPU utilization is sustained above the configured threshold."
  namespace           = "AWS/EC2"
  metric_name         = "CPUUtilization"
  statistic           = "Average"
  period              = 300
  evaluation_periods  = 3
  datapoints_to_alarm = 3
  threshold           = var.cpu_alarm_threshold_percent
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "missing"
  actions_enabled     = local.common_alarm_arguments.actions_enabled
  alarm_actions       = local.common_alarm_arguments.alarm_actions
  ok_actions          = local.common_alarm_arguments.ok_actions

  dimensions = {
    InstanceId = aws_instance.this.id
  }

  tags = merge(local.tags, {
    Name = "${local.name}-cpu-high"
  })
}

resource "aws_cloudwatch_metric_alarm" "cpu_credits_low" {
  count = startswith(var.instance_type, "t") ? 1 : 0

  alarm_name          = "${local.name}-cpu-credits-low"
  alarm_description   = "Burstable EC2 CPU credit balance is low."
  namespace           = "AWS/EC2"
  metric_name         = "CPUCreditBalance"
  statistic           = "Minimum"
  period              = 300
  evaluation_periods  = 2
  datapoints_to_alarm = 2
  threshold           = var.cpu_credit_alarm_threshold
  comparison_operator = "LessThanOrEqualToThreshold"
  treat_missing_data  = "missing"
  actions_enabled     = local.common_alarm_arguments.actions_enabled
  alarm_actions       = local.common_alarm_arguments.alarm_actions
  ok_actions          = local.common_alarm_arguments.ok_actions

  dimensions = {
    InstanceId = aws_instance.this.id
  }

  tags = merge(local.tags, {
    Name = "${local.name}-cpu-credits-low"
  })
}

resource "aws_cloudwatch_metric_alarm" "memory_high" {
  alarm_name          = "${local.name}-memory-high"
  alarm_description   = "CloudWatch agent memory use is sustained above the configured threshold."
  namespace           = "CWAgent"
  metric_name         = "mem_used_percent"
  statistic           = "Average"
  period              = 300
  evaluation_periods  = 3
  datapoints_to_alarm = 3
  threshold           = var.memory_alarm_threshold_percent
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "missing"
  actions_enabled     = local.common_alarm_arguments.actions_enabled
  alarm_actions       = local.common_alarm_arguments.alarm_actions
  ok_actions          = local.common_alarm_arguments.ok_actions

  dimensions = {
    InstanceId = aws_instance.this.id
  }

  tags = merge(local.tags, {
    Name = "${local.name}-memory-high"
  })
}

resource "aws_cloudwatch_metric_alarm" "swap_high" {
  alarm_name          = "${local.name}-swap-high"
  alarm_description   = "CloudWatch agent swap use is above the configured threshold."
  namespace           = "CWAgent"
  metric_name         = "swap_used_percent"
  statistic           = "Average"
  period              = 300
  evaluation_periods  = 2
  datapoints_to_alarm = 2
  threshold           = var.swap_alarm_threshold_percent
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  actions_enabled     = local.common_alarm_arguments.actions_enabled
  alarm_actions       = local.common_alarm_arguments.alarm_actions
  ok_actions          = local.common_alarm_arguments.ok_actions

  dimensions = {
    InstanceId = aws_instance.this.id
  }

  tags = merge(local.tags, {
    Name = "${local.name}-swap-high"
  })
}

resource "aws_cloudwatch_metric_alarm" "root_disk_high" {
  alarm_name          = "${local.name}-root-disk-high"
  alarm_description   = "CloudWatch agent root filesystem use is above the configured threshold."
  namespace           = "CWAgent"
  metric_name         = "disk_used_percent"
  statistic           = "Maximum"
  period              = 300
  evaluation_periods  = 2
  datapoints_to_alarm = 2
  threshold           = var.disk_alarm_threshold_percent
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "missing"
  actions_enabled     = local.common_alarm_arguments.actions_enabled
  alarm_actions       = local.common_alarm_arguments.alarm_actions
  ok_actions          = local.common_alarm_arguments.ok_actions

  dimensions = {
    InstanceId = aws_instance.this.id
    fstype     = var.monitored_filesystem_type
    path       = "/"
  }

  tags = merge(local.tags, {
    Name = "${local.name}-root-disk-high"
  })
}

resource "aws_cloudwatch_metric_alarm" "root_inodes_low" {
  alarm_name          = "${local.name}-root-inodes-low"
  alarm_description   = "Root filesystem is running out of free inodes."
  namespace           = "CWAgent"
  metric_name         = "disk_inodes_free"
  statistic           = "Minimum"
  period              = 300
  evaluation_periods  = 2
  datapoints_to_alarm = 2
  threshold           = var.inode_free_alarm_threshold
  comparison_operator = "LessThanThreshold"
  treat_missing_data  = "missing"
  actions_enabled     = local.common_alarm_arguments.actions_enabled
  alarm_actions       = local.common_alarm_arguments.alarm_actions
  ok_actions          = local.common_alarm_arguments.ok_actions

  dimensions = {
    InstanceId = aws_instance.this.id
    fstype     = var.monitored_filesystem_type
    path       = "/"
  }

  tags = merge(local.tags, {
    Name = "${local.name}-root-inodes-low"
  })
}

resource "aws_cloudwatch_metric_alarm" "lab_fault_disk_high" {
  count = var.enable_lab_fault_volume ? 1 : 0

  alarm_name          = "${local.name}-lab-fault-disk-high"
  alarm_description   = "CloudWatch agent lab-fault filesystem use is above the configured threshold."
  namespace           = "CWAgent"
  metric_name         = "disk_used_percent"
  statistic           = "Maximum"
  period              = 300
  evaluation_periods  = 2
  datapoints_to_alarm = 2
  threshold           = var.disk_alarm_threshold_percent
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "missing"
  actions_enabled     = local.common_alarm_arguments.actions_enabled
  alarm_actions       = local.common_alarm_arguments.alarm_actions
  ok_actions          = local.common_alarm_arguments.ok_actions

  dimensions = {
    InstanceId = aws_instance.this.id
    fstype     = var.monitored_filesystem_type
    path       = "/mnt/lab-faults"
  }

  tags = merge(local.tags, {
    Name = "${local.name}-lab-fault-disk-high"
  })
}

resource "aws_cloudwatch_log_metric_filter" "container_oom" {
  name           = "${local.name}-container-oom"
  pattern        = "{ $.Action = \"oom\" }"
  log_group_name = aws_cloudwatch_log_group.this["docker"].name

  metric_transformation {
    name          = "ContainerOOMEvents"
    namespace     = "RecSysLoom/${local.name}"
    value         = "1"
    default_value = "0"
    unit          = "Count"
  }
}

resource "aws_cloudwatch_metric_alarm" "container_oom" {
  alarm_name          = "${local.name}-container-oom"
  alarm_description   = "At least one Docker container emitted an OOM event."
  namespace           = "RecSysLoom/${local.name}"
  metric_name         = "ContainerOOMEvents"
  statistic           = "Sum"
  period              = 60
  evaluation_periods  = 1
  datapoints_to_alarm = 1
  threshold           = 0
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  actions_enabled     = local.common_alarm_arguments.actions_enabled
  alarm_actions       = local.common_alarm_arguments.alarm_actions
  ok_actions          = local.common_alarm_arguments.ok_actions

  tags = merge(local.tags, {
    Name = "${local.name}-container-oom"
  })
}

resource "aws_cloudwatch_log_metric_filter" "http_5xx" {
  name           = "${local.name}-http-5xx"
  pattern        = "{ $.status >= 500 }"
  log_group_name = aws_cloudwatch_log_group.this["application"].name

  metric_transformation {
    name          = "Http5xx"
    namespace     = "RecSysLoom/${local.name}"
    value         = "1"
    default_value = "0"
    unit          = "Count"
  }
}

resource "aws_cloudwatch_metric_alarm" "http_5xx" {
  alarm_name          = "${local.name}-http-5xx"
  alarm_description   = "Caddy emitted repeated HTTP 5xx responses."
  namespace           = "RecSysLoom/${local.name}"
  metric_name         = "Http5xx"
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  datapoints_to_alarm = 1
  threshold           = 5
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching"
  actions_enabled     = local.common_alarm_arguments.actions_enabled
  alarm_actions       = local.common_alarm_arguments.alarm_actions
  ok_actions          = local.common_alarm_arguments.ok_actions

  tags = merge(local.tags, {
    Name = "${local.name}-http-5xx"
  })
}
