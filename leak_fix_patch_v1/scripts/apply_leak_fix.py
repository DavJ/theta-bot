#!/usr/bin/env python3
# apply_leak_fix.py — idempotent in-place patch for leakage in theta_eval_hbatch_biquat_max.py
import sys, re, os, shutil

def main():
    if len(sys.argv) != 2:
        print("Usage: python apply_leak_fix.py <path/to/theta_eval_hbatch_biquat_max.py>")
        sys.exit(1)
    path = sys.argv[1]
    if not os.path.isfile(path):
        print(f"ERROR: file not found: {path}")
        sys.exit(2)
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()

    if "hi_tr = compare_idx - horizon" in src:
        print("Already patched (found 'hi_tr = compare_idx - horizon'). Nothing to do.")
        return

    original = src

    # 1) Replace 'hi = compare_idx' with 'hi_tr = compare_idx - horizon' (allowing spaces)
    src, n1 = re.subn(r'(\b)hi\s*=\s*compare_idx(\b)', r'\1hi_tr = compare_idx - horizon\2', src)

    # 2) Replace training design matrix slice X_all[lo:hi, :] -> X_all[lo:hi_tr, :]
    src, n2 = re.subn(r'X_all\s*\[\s*lo\s*:\s*hi\s*,\s*:\s*\]', r'X_all[lo:hi_tr, :]', src)

    # 3) Replace target slice closes[lo+horizon:hi+horizon] -> closes[lo+horizon:hi_tr+horizon]
    src, n3 = re.subn(r'closes\s*\[\s*lo\s*\+\s*horizon\s*:\s*hi\s*\+\s*horizon\s*\]',
                      r'closes[lo+horizon:hi_tr+horizon]', src)

    # 4) Also replace paired closes[lo:hi] -> closes[lo:hi_tr] when used with the above target
    src, n4 = re.subn(r'closes\s*\[\s*lo\s*:\s*hi\s*\]', r'closes[lo:hi_tr]', src)

    # 5) Insert guard right after the new hi_tr line (first occurrence)
    guard = (
        "hi_tr = compare_idx - horizon\n"
        "    if hi_tr <= lo:\n"
        "        continue\n"
        "    assert (hi_tr + horizon) <= (entry_idx + 1), 'leak guard: target past entry_idx'\n"
    )
    # Insert only if we changed something AND guard not already present
    inserted = False
    if (n1 + n2 + n3) > 0 and "leak guard: target past entry_idx" not in src:
        src = src.replace("hi_tr = compare_idx - horizon\n", guard, 1)
        inserted = True

    if src == original:
        print("WARN: No recognizable pattern was modified. File left untouched.")
        sys.exit(3)

    # Backup
    backup = path + ".bak"
    shutil.copy2(path, backup)
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)

    print(f"Patched {path}")
    print(f" - Replaced 'hi = compare_idx' -> 'hi_tr = compare_idx - horizon': {n1} occurrence(s)")
    print(f" - Updated X slice [lo:hi] -> [lo:hi_tr]: {n2} occurrence(s)")
    print(f" - Updated y slice [lo+horizon:hi+horizon] -> [lo+horizon:hi_tr+horizon]: {n3} occurrence(s)")
    print(f" - Updated closes [lo:hi] -> [lo:hi_tr]: {n4} occurrence(s)")
    print(f" - Inserted leak guard: {inserted}")
    print(f"Backup saved to: {backup}")

if __name__ == "__main__":
    main()
