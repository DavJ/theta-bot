import sys
import os

# Make ubt_with_chronofactor/forensic_fingerprint importable as a top-level package
_UBT_ROOT = os.path.join(os.path.dirname(__file__), "ubt_with_chronofactor")
if _UBT_ROOT not in sys.path:
    sys.path.insert(0, _UBT_ROOT)
