#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"
SOURCE_LAB_DIR="$SCRIPT_DIR/lab"
APP_DIR="${APP_DIR:-/opt/recsys-loom/app}"
BUNDLE_ROOT="${BUNDLE_ROOT:-/opt/recsys-loom/bundles}"
RUNTIME_ENV="$APP_DIR/.env.runtime"

fail() {
  echo "install-release: $*" >&2
  exit 1
}

[[ "${EUID:-$(id -u)}" -eq 0 ]] || fail "run as root"
[[ -f "$SCRIPT_DIR/deploy.sh" ]] || fail "missing source deploy.sh"
[[ -f "$SCRIPT_DIR/docker-events.sh" ]] || fail "missing docker-events.sh"
[[ -f "$SCRIPT_DIR/recsys-docker-events.service" ]] ||
  fail "missing Docker event systemd unit"
[[ -f "$SCRIPT_DIR/recsys-loom.logrotate" ]] || fail "missing logrotate policy"
[[ -f "$REPOSITORY_ROOT/docker-compose.prod.yml" ]] ||
  fail "missing production Compose file"
[[ -f "$REPOSITORY_ROOT/Caddyfile" ]] || fail "missing Caddyfile"

install -d -m 0755 "$APP_DIR" "$APP_DIR/lab"
install -d -m 0755 \
  "$BUNDLE_ROOT" \
  "$BUNDLE_ROOT/empty" \
  "$BUNDLE_ROOT/releases" \
  "$BUNDLE_ROOT/staging"
install -d -m 0700 "$APP_DIR/.deploy-state"
install -d -m 0755 /var/log/recsys-loom/caddy

if [[ -f "$APP_DIR/docker-compose.prod.yml" && -f "$APP_DIR/Caddyfile" ]]; then
  cp -p "$APP_DIR/docker-compose.prod.yml" \
    "$APP_DIR/.deploy-state/docker-compose.prod.yml.previous"
  cp -p "$APP_DIR/Caddyfile" "$APP_DIR/.deploy-state/Caddyfile.previous"
  printf 'present\n' > "$APP_DIR/.deploy-state/config.previous.state"
else
  printf 'absent\n' > "$APP_DIR/.deploy-state/config.previous.state"
fi

install -m 0755 "$SCRIPT_DIR/deploy.sh" "$APP_DIR/deploy.sh"
install -m 0755 "$SCRIPT_DIR/docker-events.sh" "$APP_DIR/docker-events.sh"
install -m 0644 \
  "$SCRIPT_DIR/recsys-docker-events.service" \
  /etc/systemd/system/recsys-docker-events.service
install -m 0644 "$SCRIPT_DIR/recsys-loom.logrotate" \
  /etc/logrotate.d/recsys-loom
install -m 0644 \
  "$REPOSITORY_ROOT/docker-compose.prod.yml" \
  "$APP_DIR/docker-compose.prod.yml"
install -m 0644 "$REPOSITORY_ROOT/Caddyfile" "$APP_DIR/Caddyfile"

for lab_script in "$SOURCE_LAB_DIR"/*.sh; do
  [[ -f "$lab_script" ]] || continue
  install -m 0755 "$lab_script" "$APP_DIR/lab/$(basename "$lab_script")"
done

if [[ ! -e "$RUNTIME_ENV" ]]; then
  : "${API_DOMAIN:?Set API_DOMAIN to the production API hostname}"
  : "${ALLOWED_ORIGINS:?Set ALLOWED_ORIGINS to the Vercel production origin}"
  [[ "$API_DOMAIN" =~ ^[A-Za-z0-9]([A-Za-z0-9.-]*[A-Za-z0-9])?$ ]] ||
    fail "API_DOMAIN must be a hostname"
  [[ "$ALLOWED_ORIGINS" =~ ^https://[^[:space:],]+(,https://[^[:space:],]+)*$ ]] ||
    fail "ALLOWED_ORIGINS must contain comma-separated HTTPS origins"
  if [[ -n "${CADDY_IMAGE:-}" ]]; then
    [[ "$CADDY_IMAGE" =~ ^[A-Za-z0-9./:@_-]+$ ]] ||
      fail "CADDY_IMAGE has an invalid format"
  fi

  umask 077
  runtime_tmp="$(mktemp "$APP_DIR/.env.runtime.XXXXXX")"
  {
    printf 'API_DOMAIN=%s\n' "$API_DOMAIN"
    printf 'ALLOWED_ORIGINS=%s\n' "$ALLOWED_ORIGINS"
    printf 'CADDY_IMAGE=%s\n' "${CADDY_IMAGE:-caddy:2.10-alpine}"
    printf 'LOG_DIR=/var/log/recsys-loom\n'
    printf 'SEARCH_WARM_ON_READINESS=1\n'
    printf 'SEARCH_SEMANTIC=0\n'
    printf 'SEARCH_VISUAL=0\n'
    printf 'SEARCH_LLM=0\n'
    printf 'RECOMMENDATION_MEMORY_LIMIT=768m\n'
    printf 'RECOMMENDATION_CPU_LIMIT=0.75\n'
    printf 'SEARCH_MEMORY_LIMIT=2g\n'
    printf 'SEARCH_CPU_LIMIT=1.5\n'
  } > "$runtime_tmp"
  mv -f "$runtime_tmp" "$RUNTIME_ENV"
  chmod 0600 "$RUNTIME_ENV"
  echo "Created $RUNTIME_ENV; add runtime secrets there if required."
else
  echo "Preserved existing $RUNTIME_ENV."
fi
chown root:docker "$RUNTIME_ENV"
chmod 0640 "$RUNTIME_ENV"

systemctl daemon-reload
systemctl enable --now recsys-docker-events.service

echo "Installed release files in $APP_DIR."
echo "Verify: docker compose --env-file $RUNTIME_ENV -f $APP_DIR/docker-compose.prod.yml config"
echo "Deployments can now run without a repository checkout."
