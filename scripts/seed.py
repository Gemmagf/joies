"""Seed both Chroma collections (catalog + heritage). Idempotent.

Usage:
    python scripts/seed.py [--force]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from maison_concierge.retrieval import CatalogRAG, HeritageRAG  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed vector stores.")
    parser.add_argument("--force", action="store_true", help="Re-index even if collections are populated.")
    args = parser.parse_args()

    catalog = CatalogRAG()
    n_catalog = catalog.index(force=args.force)
    print(f"Catalog: {n_catalog} pieces indexed.")

    heritage = HeritageRAG()
    n_heritage = heritage.index(force=args.force)
    print(f"Heritage: {n_heritage} documents indexed.")


if __name__ == "__main__":
    main()
