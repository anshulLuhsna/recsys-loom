#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

require_lab
command -v git >/dev/null 2>&1 || lab_fail "git is unavailable"

repository="$LAB_ROOT/git-conflict"
[[ ! -L "$repository" ]] || lab_fail "refusing to replace a symlink"
rm -rf -- "$repository"
git init --initial-branch=main "$repository" >/dev/null
git -C "$repository" config user.name "RecSys Lab"
git -C "$repository" config user.email "lab@example.invalid"

printf 'memory_limit=512m\n' > "$repository/service.env"
git -C "$repository" add service.env
git -C "$repository" commit -m "baseline config" >/dev/null

git -C "$repository" switch -c feature >/dev/null
printf 'memory_limit=1024m\n' > "$repository/service.env"
git -C "$repository" commit -am "raise feature limit" >/dev/null

git -C "$repository" switch main >/dev/null
printf 'memory_limit=768m\n' > "$repository/service.env"
git -C "$repository" commit -am "tune production limit" >/dev/null

set +e
git -C "$repository" merge feature >/dev/null 2>&1
merge_status=$?
set -e
(( merge_status != 0 )) || lab_fail "expected a merge conflict but merge succeeded"

echo "Created a conflict in a disposable repository: $repository"
echo "Verify: git -C $repository status && cat $repository/service.env"
echo "Resolve without modifying the real project repository."
print_cleanup_recovery
