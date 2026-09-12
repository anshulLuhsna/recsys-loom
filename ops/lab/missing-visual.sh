#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

require_lab
require_compose

override="$LAB_ROOT/missing-visual.override.yml"
cat > "$override" <<'EOF'
services:
  search-api:
    environment:
      SEARCH_VISUAL: "1"
EOF

docker compose \
  --env-file "$LAB_RUNTIME_ENV" \
  --env-file "$LAB_DEPLOY_ENV" \
  -f "$LAB_COMPOSE_FILE" \
  -f "$override" \
  up -d --no-deps --force-recreate search-api

echo "Enabled visual search without adding visual artifacts."
echo "Verify: curl -i https://your-api.example/health/search returns 503"
print_cleanup_recovery
