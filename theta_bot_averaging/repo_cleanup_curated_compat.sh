#!/usr/bin/env bash
set -euo pipefail

# Repo cleanup (curated) – kompatibilní s Bash 3.2 (macOS).
# Usage:
#   bash repo_cleanup_curated_compat.sh [--apply]
#
# Default: dry-run (jen vypíše akce). S --apply opravdu provede změny.

APPLY=false
if [ "${1:-}" = "--apply" ]; then
  APPLY=true
fi

say() { printf '>> %s\n' "$*"; }

# --- Kurátorované položky (uprav podle potřeby) ---

# Důležité adresáře držet:
KEEP_DIRS=("theta_bot_averaging/" "src/")

# Důležité soubory (globy jsou povolené – expandujeme přes git ls-files):
KEEP_FILES=(
  "theta_eval_hbatch_biquat_max.py"
  "robustness_suite_v3_oos.py"
  "robustness_suite_v2.py"
  "robustness_suite.py"
  "biquat_prepare_and_compare.py"
  "theta_eval_hstrategy.py"
  "theta_eval_hbatch.py"
  "theta_predictor.py"
  "theta_backtest.py"
  "theta_batch.py"
  "make_prices_csv.py"
  "oos_pnl.py"
  "README.md"
  "README*.md"
  "LICENSE"
  "pyproject.toml"
  "requirements*.txt"
  "setup.cfg"
  ".flake8"
  ".pre-commit-config.yaml"
  "Dockerfile"
  "docker-compose.yml"
)

# --- Příprava adresářů ---
mkdir -p regression/prices regression/eval regression/reports history/legacy_scripts history/legacy_data

# --- .gitignore (append nebo vytvořit) ---
if [ -f .gitignore ]; then
  say "Append ignore rules do existující .gitignore"
  if $APPLY; then
    printf "\n\n# === Auto-added rules (curated cleanup) ===\n" >> .gitignore
    cat .gitignore.new >> .gitignore
  fi
else
  say "Create .gitignore"
  if $APPLY; then
    mv .gitignore.new .gitignore
  fi
fi

# --- Legacy (bak/pycache/swp/log) -> history/ ---
# Použijeme newline-delimited (Bash 3.2 friendly)
LEGACY_PAT='(\.bak($|_)|__pycache__/|\.swp$|\.log$)'
git ls-files | egrep -E "$LEGACY_PAT" | while IFS= read -r f; do
  [ -z "$f" ] && continue
  tgt="history/legacy_scripts/${f//\//_}"
  say "move legacy: $f -> $tgt"
  if $APPLY; then
    mkdir -p history/legacy_scripts
    git mv -f "$f" "$tgt" || true
  fi
done

# --- Smazat velká data/reporty mimo regression/ ---
BULKY_EXT='(csv|parquet|feather|xlsx|json|pdf|png)$'
git ls-files | egrep -E "\.($BULKY_EXT)" | grep -v '^regression/' | while IFS= read -r f; do
  [ -z "$f" ] && continue
  say "rm bulky data: $f"
  if $APPLY; then
    git rm -f "$f" || true
  fi
done

# --- Keep dirs info (jen vypíšeme) ---
for d in "${KEEP_DIRS[@]}"; do
  if [ -d "$d" ]; then
    say "keep dir: $d"
  fi
done

# --- Sestavit množinu explicitně držených souborů (KEEP_SET) ---
KEEP_SET_FILE=".keep_set_tmp.txt"
: > "$KEEP_SET_FILE"
for pat in "${KEEP_FILES[@]}"; do
  git ls-files "$pat" 2>/dev/null || true
done | sort -u >> "$KEEP_SET_FILE"

is_in_keep_set() {
  # arg1: filename
  # návrat 0 pokud je v KEEP_SET_FILE
  grep -Fqx "$1" "$KEEP_SET_FILE"
}

# --- Přesunout ostatní .py (mimo keep dirs a mimo KEEP_SET) do history/ ---
git ls-files "*.py" | while IFS= read -r pyf; do
  [ -z "$pyf" ] && continue

  # V chráněném adresáři?
  in_keep_dir=false
  for kd in "${KEEP_DIRS[@]}"; do
    case "$pyf" in
      "$kd"*) in_keep_dir=true; break ;;
    esac
  done
  $in_keep_dir && continue

  # Je explicitně v KEEP_SET?
  if is_in_keep_set "$pyf"; then
    continue
  fi

  tgt="history/legacy_scripts/${pyf//\//_}"
  say "move non-key script: $pyf -> $tgt"
  if $APPLY; then
    mkdir -p history/legacy_scripts
    git mv -f "$pyf" "$tgt" || true
  fi
done

rm -f "$KEEP_SET_FILE" || true

say "Dry run complete. Spusť s --apply pro reálné změny."

