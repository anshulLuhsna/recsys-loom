#!/usr/bin/env bash

set -Eeuo pipefail

[[ "${EUID:-$(id -u)}" -eq 0 ]] || {
  echo "install-host-prerequisites: run as root" >&2
  exit 1
}

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y ca-certificates curl gnupg unzip e2fsprogs git logrotate lsof

if ! command -v aws >/dev/null 2>&1; then
  machine="$(dpkg --print-architecture)"
  case "$machine" in
    amd64) aws_arch=x86_64 ;;
    arm64) aws_arch=aarch64 ;;
    *) echo "Unsupported architecture for AWS CLI: $machine" >&2; exit 1 ;;
  esac
  aws_tmp="$(mktemp -d)"
  trap 'rm -rf "$aws_tmp"' EXIT
  curl --fail --silent --show-error --location --retry 5 \
    "https://awscli.amazonaws.com/awscli-exe-linux-${aws_arch}.zip" \
    -o "$aws_tmp/awscliv2.zip"
  unzip -q "$aws_tmp/awscliv2.zip" -d "$aws_tmp"
  "$aws_tmp/aws/install" --update
fi

if ! command -v docker >/dev/null 2>&1; then
  install -m 0755 -d /etc/apt/keyrings
  curl --fail --silent --show-error --location --retry 5 \
    https://download.docker.com/linux/ubuntu/gpg \
    -o /etc/apt/keyrings/docker.asc
  chmod a+r /etc/apt/keyrings/docker.asc
  . /etc/os-release
  printf '%s\n' \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $VERSION_CODENAME stable" \
    > /etc/apt/sources.list.d/docker.list
  apt-get update
  apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi

systemctl enable --now docker
getent group docker >/dev/null

