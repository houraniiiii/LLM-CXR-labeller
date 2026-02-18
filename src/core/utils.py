"""Shared utility helpers used across the runtime."""
from __future__ import annotations

import re


def slugify(value: str) -> str:
    """Convert a string into a filesystem-friendly slug."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    return slug.strip("-")


__all__ = ["slugify"]

