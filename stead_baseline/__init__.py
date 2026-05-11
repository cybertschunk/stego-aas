"""STEAD baseline package.

Adds the vendored directory to sys.path so that vendored files (which use
bare imports like `from utils import ...` and `from config import ...`) can
resolve those modules against our reconstructed `vendor/utils.py` and
`vendor/config.py`.
"""

import os
import sys

_VENDOR_DIR = os.path.join(os.path.dirname(__file__), "vendor")
if _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)
