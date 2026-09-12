#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

require_lab
require_compose

base_compose config --services | grep -Fxq "recommendation-api" ||
  lab_fail "Compose configuration has no recommendation-api service"

missing_dir="$LAB_ROOT/missing-artifact"
override="$LAB_ROOT/missing-artifact.override.yml"
install -d -m 0755 "$missing_dir"
printf '{}\n' > "$missing_dir/manifest.json"

cat > "$override" <<EOF
services:
  recommendation-api:
    volumes:
      - type: bind
        source: $missing_dir
        target: /app/artifacts/serving
        read_only: true
EOF

docker compose \
  --env-file "$LAB_RUNTIME_ENV" \
  --env-file "$LAB_DEPLOY_ENV" \
  -f "$LAB_COMPOSE_FILE" \
  -f "$override" \
  up -d --no-deps --force-recreate recommendation-api

echo "Mounted an invalid lab-only serving-artifact manifest."
echo "Verify: curl -i https://your-api.example/health/recommendation returns 503"
print_cleanup_recovery
