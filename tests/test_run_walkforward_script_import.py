import runpy
import sys
from pathlib import Path


def test_run_walkforward_script_sets_repo_root_in_sys_path():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_walkforward.py"

    original_sys_path = sys.path.copy()
    try:
        sanitized_sys_path = [
            p for p in original_sys_path if p not in {"", str(repo_root)}
        ]
        sys.path = [str(script_path.parent)] + sanitized_sys_path

        runpy.run_path(str(script_path), run_name="__test__")

        assert str(repo_root) in sys.path
    finally:
        sys.path = original_sys_path
