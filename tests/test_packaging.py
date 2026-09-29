from __future__ import annotations

import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_wheel_configuration_keeps_both_research_package_roots() -> None:
    """Offline primitives must ship with the installable research artifact."""

    config = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    packages = config["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]

    assert set(packages) >= {"src/quant_lab", "src/quant_trading"}
    assert all((PROJECT_ROOT / package).is_dir() for package in packages)
