"""Locate immutable defaults in a checkout or an installed wheel."""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CHECKOUT_ROOT = PACKAGE_ROOT.parents[1]


def resource_path(relative: str) -> Path:
    """Prefer the checkout's canonical files; wheels carry byte-identical copies."""
    if (CHECKOUT_ROOT / "pyproject.toml").is_file():
        return CHECKOUT_ROOT / relative
    return PACKAGE_ROOT / "resources" / relative
