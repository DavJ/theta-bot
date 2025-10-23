#!/usr/bin/env bash
set -euo pipefail

# Repo cleanup (curated). Default is DRY RUN.
# Usage:
#   bash repo_cleanup_curated.sh [--apply]

APPLY=false
if [[ "{1:-}" == "--apply" ]]; then
  APPLY=true
fi

say() { echo ">> $*"; }

# Ensure folders exist
mkdir -p regression/prices regression/eval regression/reports history/legacy_scripts history/legacy_data

# 1) Install/append .gitignore
if [[ -f .gitignore ]]; then
  say "Append ignore rules to existing .gitignore"
  $APPLY && printf "\n\n# === Auto-added rules (curated cleanup) ===\n" >> .gitignore
  $APPLY && cat .gitignore.new >> .gitignore
else
  say "Create .gitignore"
  $APPLY && mv .gitignore.new .gitignore
fi

# 2) Move obvious backups / caches to history
mapfile -t LEGACY < <(git ls-files -z | tr -d '\n' | tr '\0' '\n' | grep -E '(\\.bak($|_)|__pycache__/|\\.swp$|\\.log$)')
for f in "${LEGACY[@]}"; do
  [[ -z "$f" ]] && continue
  tgt="history/legacy_scripts/${f//\//_}"
  say "move legacy: $f -> $tgt"
  $APPLY && git mv -f "$f" "$tgt" || true
done

# 3) Remove bulky data/reports outside regression/
mapfile -t BULKY < <(git ls-files -z | tr -d '\n' | tr '\0' '\n' | grep -E '\\.(csv|parquet|feather|xlsx|json|pdf|png)$' | grep -v '^regression/')
for f in "${BULKY[@]}"; do
  [[ -z "$f" ]] && continue
  say "rm bulky data: $f"
  $APPLY && git rm -f "$f" || true
done

# 4) Keep curated directories
KEEP_DIRS=(theta_bot_averaging/ src/)
for d in "${KEEP_DIRS[@]}"; do
  if [[ -d "$d" ]]; then
    say "keep dir: $d"
  fi
done

# 5) Keep curated files (globs supported via git ls-files)
KEEP_FILES=(theta_eval_hbatch_biquat_max.py robustness_suite_v3_oos.py robustness_suite_v2.py robustness_suite.py biquat_prepare_and_compare.py robustness_suite.py theta_eval_hstrategy.py theta_eval_hbatch.py theta_predictor.py theta_backtest.py theta_batch.py make_prices_csv.py oos_pnl.py README.md README*.md LICENSE pyproject.toml requirements*.txt setup.cfg .flake8 .pre-commit-config.yaml Dockerfile docker-compose.yml)
declare -A KEEP_SET
for pat in "${KEEP_FILES[@]}"; do
  while IFS= read -r -d '' f; do
    KEEP_SET["$f"]=1
  done < <(git ls-files -z "$pat" 2>/dev/null || true)
done

# 6) Move other .py scripts to history/ (if nejsou v curated setu nebo uvnitř keep dirů)
while IFS= read -r -d '' pyf; do
  # skip if in keep dirs
  in_keep_dir=false
  for kd in "${KEEP_DIRS[@]}"; do
    if [[ "$pyf" == $kd* ]]; then
      in_keep_dir=true; break
    fi
  done
  if $in_keep_dir; then
    continue
  fi
  # skip if explicit keep
  if [[ -n "${KEEP_SET[$pyf]:-}" ]]; then
    continue
  fi
  tgt="history/legacy_scripts/${pyf//\//_}"
  say "move non-key script: $pyf -> $tgt"
  $APPLY && mkdir -p history/legacy_scripts && git mv -f "$pyf" "$tgt" || true
done < <(git ls-files -z "*.py")

# 7) Summary hint
say "Dry run complete. Re-run with --apply to modify repo."