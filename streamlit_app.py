"""Streamlit Cloud entry point.

Streamlit Community Cloud picks up `streamlit_app.py` at the repo root
automatically. This file delegates to the real UI in `apps/hospitality.py`
so the local `streamlit run apps/hospitality.py` command keeps working too.
"""

from __future__ import annotations

from pathlib import Path

# `apps/` is not a package (no __init__.py) — load the module by path so we
# don't force a project-layout change that would break existing scripts.
import importlib.util as _importlib_util
import sys as _sys

_APPS_DIR = Path(__file__).resolve().parent / "apps"
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
