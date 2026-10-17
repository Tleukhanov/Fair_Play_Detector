"""Lichess username normalization and validation."""

import re
import unicodedata

_USERNAME_PATTERN = re.compile(r"^[a-z0-9_][a-z0-9_-]{0,29}$")


def normalize_username(username: str) -> str:
    """Normalize a public Lichess username or raise a clear validation error."""

    normalized = unicodedata.normalize("NFKC", username).strip().casefold()
    if not _USERNAME_PATTERN.fullmatch(normalized):
        raise ValueError("invalid Lichess username format")
    return normalized
