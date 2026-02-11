"""
Pytest conftest.py — global test configuration.

Applies the Windows DLL fix BEFORE any test module is collected.
This prevents access violations when pytest imports modules that
load native extensions (DuckDB, spaCy, torch).
"""

import sys
import os

# ═══ DLL Fix: Must run before ANY native extension import ═══
# This is the earliest hook pytest provides — module-level code in conftest.py
# runs during collection, before any test file is even parsed.
try:
    from app.core.dll_fix import apply_dll_fix
    apply_dll_fix()
except ImportError:
    pass
