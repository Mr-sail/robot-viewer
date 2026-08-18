"""Resolve ROS package share directories without requiring a ROS install."""

from __future__ import annotations

import os
from pathlib import Path


class PackageNotFoundError(KeyError):
    """Raised when a package is absent from the configured bundle."""


def _prefixes() -> list[Path]:
    values = [
        os.environ.get("ESCOPE_ROS_PREFIX_PATH", ""),
        os.environ.get("AMENT_PREFIX_PATH", ""),
    ]
    prefixes: list[Path] = []
    for value in values:
        for entry in value.split(os.pathsep):
            if entry.strip():
                prefix = Path(entry).expanduser()
                if prefix not in prefixes:
                    prefixes.append(prefix)
    return prefixes


def get_package_share_directory(package_name: str, print_warning: bool = True) -> str:
    """Return ``share/<package_name>`` from the bundled ROS prefixes."""
    if not package_name or "/" in package_name or "\\" in package_name:
        raise ValueError(f"invalid package name: {package_name!r}")

    for prefix in _prefixes():
        share = prefix / "share" / package_name
        if share.is_dir():
            return str(share)

    searched = ", ".join(str(prefix) for prefix in _prefixes()) or "<none>"
    raise PackageNotFoundError(
        f"package {package_name!r} was not found in bundled ROS prefixes: {searched}"
    )

