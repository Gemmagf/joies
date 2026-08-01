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

# `scripts/` is not a package — expose its parent (repo root) on the path so
# `from scripts.bootstrap_demo import ...` resolves.
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

# Bootstrap analytics + personas + KB on cold start. Idempotent — a warm
# boot with the artifacts already in `data/models/` no-ops in <100ms.
# Wrapped in a Streamlit cached call so parallel session boots don't retrain
# in a race.
import streamlit as _st  # noqa: E402 — must import after sys.path setup


@_st.cache_resource(show_spinner="Preparing the concierge (first boot may take 2-3 minutes)...")
def _bootstrap_once() -> bool:
    from scripts.bootstrap_demo import (  # noqa: PLC0415 — cold-path import
        _generate_personas_if_needed,
        _train_analytics_if_needed,
        _warm_kb,
    )
    _train_analytics_if_needed(force=False)
    _generate_personas_if_needed(force=False)
    _warm_kb(force=False)
    return True


_bootstrap_once()

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
