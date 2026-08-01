"""Streamlit Cloud entry point.

Streamlit Community Cloud picks up `streamlit_app.py` at the repo root
automatically. This file delegates to the real UI in `apps/hospitality.py`
so the local `streamlit run apps/hospitality.py` command keeps working too.
"""

from __future__ import annotations

import importlib.util as _importlib_util
import sys as _sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent

# Make `maison_concierge` importable even when the repo hasn't been
# `pip install -e .`-installed (belt-and-braces alongside the `-e .` line
# in requirements.txt — Streamlit Cloud reruns cause races where the
# editable install can lose its .pth registration).
_SRC = _REPO_ROOT / "src"
if _SRC.exists() and str(_SRC) not in _sys.path:
    _sys.path.insert(0, str(_SRC))

# `apps/` is not a package (no __init__.py) — load the module by path so we
# don't force a project-layout change that would break existing scripts.
_APPS_DIR = _REPO_ROOT / "apps"
_HOSPITALITY_PATH = _APPS_DIR / "hospitality.py"

_spec = _importlib_util.spec_from_file_location(
    "maison_hospitality_app", _HOSPITALITY_PATH
)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Could not load {_HOSPITALITY_PATH}")

_module = _importlib_util.module_from_spec(_spec)
_sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)

_module.main()
