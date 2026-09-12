#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

require_lab
command -v aws >/dev/null 2>&1 || lab_fail "aws is unavailable"

instance_id="${INSTANCE_ID:-}"
[[ "$instance_id" =~ ^i-[0-9a-f]{8,17}$ ]] ||
  lab_fail "set INSTANCE_ID to the lab EC2 instance ID"

aws ec2 create-tags \
  --resources "$instance_id" \
  --tags "Key=LabDrift,Value=1"

printf '%s\n' "$instance_id" > "$LAB_ROOT/terraform-drift.instance"

echo "Added harmless tag LabDrift=1 on $instance_id."
echo "Verify: terraform -chdir=infra/live plan"
echo "Decide whether Terraform or the console is authoritative, then either"
echo "  terraform apply  or  aws ec2 delete-tags --resources $instance_id --tags Key=LabDrift"
print_cleanup_recovery
