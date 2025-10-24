#!/usr/bin/env bash
set -euo pipefail

SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
FILE="theta_eval_hbatch_biquat_max.py"

# Try common locations:
CAND=(
  "./theta_bot_averaging/$FILE"
  "./$FILE"
)

found=""
for dst in "${CAND[@]}"; do
  if [[ -f "$dst" ]]; then
    found="$dst"
    break
  fi
done

if [[ -z "$found" ]]; then
  echo "Could not find target file in repo. Please copy manually:"
  echo "  cp "$SRC_DIR/$FILE" <your-repo-path>/theta_bot_averaging/$FILE"
  exit 1
fi

cp -f "$SRC_DIR/$FILE" "$found"
echo "Patched: $found"
