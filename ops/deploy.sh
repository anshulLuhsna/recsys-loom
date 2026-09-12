#!/usr/bin/env bash

set -Eeuo pipefail

APP_DIR="${APP_DIR:-$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
BUNDLE_ROOT="${BUNDLE_ROOT:-/opt/recsys-loom/bundles}"
COMPOSE_FILE="$APP_DIR/docker-compose.prod.yml"
RUNTIME_ENV="$APP_DIR/.env.runtime"
DEPLOY_ENV="$APP_DIR/.env.deploy"
STATE_DIR="$APP_DIR/.deploy-state"
CURRENT_LINK="$BUNDLE_ROOT/current"
EMPTY_BUNDLE="$BUNDLE_ROOT/empty"
PREVIOUS_ENV="$STATE_DIR/.env.deploy.previous"
PREVIOUS_ENV_STATE="$STATE_DIR/.env.deploy.previous.state"
PREVIOUS_LINK="$STATE_DIR/current.previous"
PREVIOUS_LINK_STATE="$STATE_DIR/current.previous.state"
PREVIOUS_COMPOSE="$STATE_DIR/docker-compose.prod.yml.previous"
PREVIOUS_CADDY="$STATE_DIR/Caddyfile.previous"
PREVIOUS_CONFIG_STATE="$STATE_DIR/config.previous.state"
STAGING_DIR=""
SNAPSHOT_TAKEN=0

DEPLOY_LOG="${DEPLOY_LOG:-/opt/recsys-loom/logs/deploy.log}"
if [[ -d "$(dirname -- "$DEPLOY_LOG")" ]]; then
  touch "$DEPLOY_LOG"
  chmod 0640 "$DEPLOY_LOG"
  exec > >(tee -a "$DEPLOY_LOG") 2>&1
fi

fail() {
  echo "deploy: $*" >&2
  return 1
}

cleanup_staging() {
  if [[ -n "$STAGING_DIR" ]]; then
    case "$STAGING_DIR" in
      "$BUNDLE_ROOT"/staging/*)
        if ! rm -rf -- "$STAGING_DIR"; then
          echo "deploy: could not clean staging directory: $STAGING_DIR" >&2
        fi
        ;;
      *)
        echo "deploy: refusing to clean unexpected staging path" >&2
        ;;
    esac
  fi
}

compose() {
  docker compose \
    --env-file "$RUNTIME_ENV" \
    --env-file "$DEPLOY_ENV" \
    -f "$COMPOSE_FILE" \
    "$@"
}

set_current_link() {
  local target="$1"
  local temporary="$BUNDLE_ROOT/.current.$$.tmp"

  rm -f -- "$temporary"
  ln -s -- "$target" "$temporary"
  mv -Tf -- "$temporary" "$CURRENT_LINK"
}

restore_previous_state() {
  local previous_target=""

  [[ "$SNAPSHOT_TAKEN" -eq 1 ]] || return 0
  echo "deploy: restoring previous deployment state" >&2

  if [[ "$(cat "$PREVIOUS_ENV_STATE" 2>/dev/null || true)" == "present" ]]; then
    cp -p -- "$PREVIOUS_ENV" "$DEPLOY_ENV"
  else
    rm -f -- "$DEPLOY_ENV"
  fi

  if [[ "$(cat "$PREVIOUS_LINK_STATE" 2>/dev/null || true)" == "symlink" ]]; then
    previous_target="$(cat "$PREVIOUS_LINK")"
    set_current_link "$previous_target"
  elif [[ -L "$CURRENT_LINK" ]]; then
    rm -f -- "$CURRENT_LINK"
  fi

  if [[ "$(cat "$PREVIOUS_CONFIG_STATE" 2>/dev/null || true)" == "present" ]]; then
    cp -p -- "$PREVIOUS_COMPOSE" "$COMPOSE_FILE"
    cp -p -- "$PREVIOUS_CADDY" "$APP_DIR/Caddyfile"
  fi

  if [[ -f "$DEPLOY_ENV" ]]; then
    if compose up -d --remove-orphans; then
      echo "deploy: previous Compose configuration restored" >&2
    else
      echo "deploy: state restored, but previous services need manual recovery" >&2
    fi
  fi
}

on_error() {
  local status="$1"
  local line="$2"

  trap - ERR
  set +e
  echo "deploy: failure at line $line; starting rollback" >&2
  restore_previous_state
  cleanup_staging
  exit "$status"
}

validate_manifest() {
  local manifest="$1"
  local expected_version="$2"

  [[ -s "$manifest" ]] || return 1
  python3 - "$manifest" "$expected_version" <<'PY'
import json
import pathlib
import sys

manifest_path = pathlib.Path(sys.argv[1])
expected_version = sys.argv[2]
payload = json.loads(manifest_path.read_text(encoding="utf-8"))
if not isinstance(payload, dict) or not payload:
    raise SystemExit("manifest must be a non-empty JSON object")
for key in ("bundle_type", "bundle_version"):
    if not isinstance(payload.get(key), str) or not payload[key]:
        raise SystemExit(f"manifest field {key} must be a non-empty string")
if payload["bundle_version"] != expected_version:
    raise SystemExit("manifest bundle_version does not match requested version")
PY
}

validate_release() {
  local release_dir="$1"
  local expected_version="$2"

  validate_manifest "$release_dir/release-manifest.json" "$expected_version" || return 1
  [[ -s "$release_dir/articles.csv" ]] || return 1
  [[ -s "$release_dir/artifacts/overnight/demo_customers.json" ]] || return 1
  [[ -s "$release_dir/artifacts/overnight/demo_recommendations.json" ]] || return 1
  [[ -s "$release_dir/artifacts/recent_popularity_7d/metrics.json" ]] || return 1
  [[ -d "$release_dir/images" ]] || return 1
}

extract_bundle() {
  local archive="$1"
  local destination="$2"

  python3 - "$archive" "$destination" <<'PY'
import pathlib
import sys
import tarfile

archive_path = pathlib.Path(sys.argv[1])
destination = pathlib.Path(sys.argv[2]).resolve()

with tarfile.open(archive_path, mode="r:*") as bundle:
    members = bundle.getmembers()
    if not members:
        raise SystemExit("serving bundle archive is empty")
    for member in members:
        target = (destination / member.name).resolve()
        try:
            target.relative_to(destination)
        except ValueError as error:
            raise SystemExit(f"unsafe archive path: {member.name}") from error
        if member.issym() or member.islnk():
            raise SystemExit(f"archive links are not allowed: {member.name}")
        if not (member.isfile() or member.isdir()):
            raise SystemExit(f"unsupported archive entry: {member.name}")
    bundle.extractall(destination, members=members)
PY
}

for required_name in \
  RECOMMENDATION_IMAGE \
  SEARCH_IMAGE \
  PUBLIC_BASE_URL \
  DEPLOYMENT_SHA; do
  [[ -n "${!required_name:-}" ]] || fail "required environment is empty: $required_name"
done

[[ "$RECOMMENDATION_IMAGE" =~ ^[A-Za-z0-9][A-Za-z0-9._:/-]*@sha256:[0-9a-f]{64}$ ]] ||
  fail "RECOMMENDATION_IMAGE must be an immutable digest reference"
[[ "$SEARCH_IMAGE" =~ ^[A-Za-z0-9][A-Za-z0-9._:/-]*@sha256:[0-9a-f]{64}$ ]] ||
  fail "SEARCH_IMAGE must be an immutable digest reference"
[[ "$PUBLIC_BASE_URL" =~ ^https://[^/[:space:]\'\"]+/?$ ]] ||
  fail "PUBLIC_BASE_URL must be an HTTPS origin without a path"
PUBLIC_BASE_URL="${PUBLIC_BASE_URL%/}"
[[ "$DEPLOYMENT_SHA" =~ ^[0-9a-f]{40}$ ]] ||
  fail "DEPLOYMENT_SHA must be a full commit SHA"

SERVING_BUNDLE_URI="${SERVING_BUNDLE_URI:-}"
SERVING_BUNDLE_VERSION="${SERVING_BUNDLE_VERSION:-}"
if [[ -n "$SERVING_BUNDLE_URI" || -n "$SERVING_BUNDLE_VERSION" ]]; then
  [[ -n "$SERVING_BUNDLE_URI" && -n "$SERVING_BUNDLE_VERSION" ]] ||
    fail "serving bundle URI and version must be provided together"
  [[ "$SERVING_BUNDLE_URI" =~ ^s3://[A-Za-z0-9.-]+/[A-Za-z0-9._/-]+$ ]] ||
    fail "SERVING_BUNDLE_URI must be an S3 object URI"
  [[ "$SERVING_BUNDLE_VERSION" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] ||
    fail "SERVING_BUNDLE_VERSION has an invalid format"
fi

[[ "${EUID:-$(id -u)}" -eq 0 ]] || fail "run as root"
[[ -f "$COMPOSE_FILE" ]] || fail "missing $COMPOSE_FILE"
[[ -f "$APP_DIR/Caddyfile" ]] || fail "missing $APP_DIR/Caddyfile"
[[ -f "$RUNTIME_ENV" ]] || fail "missing $RUNTIME_ENV"
[[ ! -e "$CURRENT_LINK" || -L "$CURRENT_LINK" ]] ||
  fail "$CURRENT_LINK must be absent or a symlink"

for command_name in aws curl docker flock mktemp python3; do
  command -v "$command_name" >/dev/null 2>&1 ||
    fail "required command is unavailable: $command_name"
done
docker compose version >/dev/null 2>&1 || fail "Docker Compose v2 is unavailable"

install -d -m 0755 \
  "$BUNDLE_ROOT" \
  "$EMPTY_BUNDLE" \
  "$BUNDLE_ROOT/releases" \
  "$BUNDLE_ROOT/staging"
install -d -m 0700 "$STATE_DIR"

exec 9>"$STATE_DIR/deploy.lock"
flock -n 9 || fail "another deployment is in progress"

if [[ -f "$DEPLOY_ENV" ]]; then
  cp -p -- "$DEPLOY_ENV" "$PREVIOUS_ENV"
  printf 'present\n' > "$PREVIOUS_ENV_STATE"
else
  rm -f -- "$PREVIOUS_ENV"
  printf 'absent\n' > "$PREVIOUS_ENV_STATE"
fi

if [[ -L "$CURRENT_LINK" ]]; then
  readlink "$CURRENT_LINK" > "$PREVIOUS_LINK"
  printf 'symlink\n' > "$PREVIOUS_LINK_STATE"
else
  rm -f -- "$PREVIOUS_LINK"
  printf 'absent\n' > "$PREVIOUS_LINK_STATE"
fi
SNAPSHOT_TAKEN=1
trap 'on_error "$?" "$LINENO"' ERR

bundle_mount="$EMPTY_BUNDLE"
if [[ -L "$CURRENT_LINK" ]]; then
  bundle_mount="$CURRENT_LINK"
fi

if [[ -n "$SERVING_BUNDLE_URI" ]]; then
  release_dir="$BUNDLE_ROOT/releases/$SERVING_BUNDLE_VERSION"
  if [[ -d "$release_dir" ]]; then
    validate_release "$release_dir" "$SERVING_BUNDLE_VERSION" ||
      fail "existing release is incomplete or invalid: $release_dir"
    [[ -f "$release_dir/.source-uri" ]] ||
      fail "existing release has no source record: $release_dir"
    [[ "$(cat "$release_dir/.source-uri")" == "$SERVING_BUNDLE_URI" ]] ||
      fail "bundle version already exists with a different S3 URI"
  else
    STAGING_DIR="$(mktemp -d "$BUNDLE_ROOT/staging/$SERVING_BUNDLE_VERSION.XXXXXX")"
    archive="$STAGING_DIR/bundle.tar"
    extracted="$STAGING_DIR/extracted"
    install -d -m 0755 "$extracted"

    echo "deploy: downloading serving bundle $SERVING_BUNDLE_VERSION"
    aws s3 cp --only-show-errors "$SERVING_BUNDLE_URI" "$archive"
    extract_bundle "$archive" "$extracted"
    validate_release "$extracted" "$SERVING_BUNDLE_VERSION" ||
      fail "serving bundle is missing required runtime assets"
    printf '%s\n' "$SERVING_BUNDLE_URI" > "$extracted/.source-uri"
    printf '%s\n' "$SERVING_BUNDLE_VERSION" > "$extracted/.bundle-version"
    mv -- "$extracted" "$release_dir"
  fi

  set_current_link "$release_dir"
  bundle_mount="$CURRENT_LINK"
fi

[[ "$bundle_mount" != "$EMPTY_BUNDLE" ]] ||
  fail "no serving bundle is active; provide an immutable S3 bundle"
recommendation_digest="${RECOMMENDATION_IMAGE##*@}"
search_digest="${SEARCH_IMAGE##*@}"
model_version="${SERVING_BUNDLE_VERSION:-$DEPLOYMENT_SHA}"
deploy_env_tmp="$(mktemp "$APP_DIR/.env.deploy.XXXXXX")"
{
  printf 'RECOMMENDATION_IMAGE=%s\n' "$RECOMMENDATION_IMAGE"
  printf 'SEARCH_IMAGE=%s\n' "$SEARCH_IMAGE"
  printf 'RECOMMENDATION_IMAGE_DIGEST=%s\n' "$recommendation_digest"
  printf 'SEARCH_IMAGE_DIGEST=%s\n' "$search_digest"
  printf 'MODEL_VERSION=%s\n' "$model_version"
  printf 'RELEASE_DIR=%s\n' "$bundle_mount"
  printf 'IMAGES_DIR=%s/images\n' "$bundle_mount"
  printf 'SERVING_BUNDLE_VERSION=%s\n' "$SERVING_BUNDLE_VERSION"
} > "$deploy_env_tmp"
chown root:docker "$deploy_env_tmp"
chmod 0640 "$deploy_env_tmp"
mv -f -- "$deploy_env_tmp" "$DEPLOY_ENV"

compose config --quiet
recommendation_registry="${RECOMMENDATION_IMAGE%%/*}"
search_registry="${SEARCH_IMAGE%%/*}"
[[ "$recommendation_registry" == "$search_registry" ]] ||
  fail "recommendation and search images must use the same ECR registry"
[[ "$recommendation_registry" =~ ^[0-9]{12}\.dkr\.ecr\.([a-z0-9-]+)\.amazonaws\.com(\.cn)?$ ]] ||
  fail "image registry is not a recognized private ECR registry"
ecr_region="${BASH_REMATCH[1]}"
aws ecr get-login-password --region "$ecr_region" |
  docker login --username AWS --password-stdin "$recommendation_registry"
compose pull
compose up -d --remove-orphans

running_services="$(compose ps --status running --services)"
grep -Fxq "recommendation-api" <<< "$running_services" ||
  fail "recommendation container is not running"
grep -Fxq "search-api" <<< "$running_services" ||
  fail "search container is not running"
grep -Fxq "caddy" <<< "$running_services" ||
  fail "Caddy container is not running"

for readiness_path in /health/recommendation /health/search; do
  curl \
    --fail \
    --silent \
    --show-error \
    --retry 18 \
    --retry-all-errors \
    --retry-delay 5 \
    --connect-timeout 10 \
    --max-time 20 \
    "$PUBLIC_BASE_URL$readiness_path" >/dev/null
done

trap - ERR
cleanup_staging
echo "deploy: success"
echo "Recommendation image: $RECOMMENDATION_IMAGE"
echo "Search image: $SEARCH_IMAGE"
if [[ -n "$SERVING_BUNDLE_VERSION" ]]; then
  echo "Serving bundle: $SERVING_BUNDLE_VERSION"
else
  echo "Serving bundle: unchanged or not configured"
fi
echo "Readiness: $PUBLIC_BASE_URL/health/recommendation and /health/search"
