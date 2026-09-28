"""Explicit selection of original, zero-based spectral input positions."""
from __future__ import annotations

from numbers import Integral


def validate_band_indices(indices, total_bands: int) -> list[int]:
    values = list(indices)
    if not values or any(isinstance(i, bool) or not isinstance(i, Integral) for i in values):
        raise ValueError("Band indices must be a nonempty list of integers")
    values = [int(i) for i in values]
    if min(values) < 0 or max(values) >= total_bands:
        raise ValueError(f"Band indices must be between 0 and {total_bands - 1}")
    if values != sorted(set(values)):
        raise ValueError("Band indices must be unique and ascending, preserving spectral order")
    return values


def parse_bands(spec: str, total_bands: int) -> list[int]:
    """all, 0:104, 0:313:2, or 0:100,200:313. Slice stop is exclusive."""
    if total_bands < 1:
        raise ValueError("total_bands must be positive")
    if spec.strip().lower() == "all":
        return list(range(total_bands))
    indices = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            raise ValueError("Empty band selection item")
        if ":" not in item:
            indices.append(int(item))
            continue
        parts = item.split(":")
        if len(parts) not in (2, 3):
            raise ValueError("Use start:stop[:step], with stop excluded")
        start = int(parts[0]) if parts[0] else 0
        stop = int(parts[1]) if parts[1] else total_bands
        step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
        if not 0 <= start < stop <= total_bands or step < 1:
            raise ValueError(f"Invalid band range {item!r} for {total_bands} bands")
        indices.extend(range(start, stop, step))
    return validate_band_indices(indices, total_bands)


def select_bands(cube, indices=None, source_bands=None):
    if source_bands is not None and cube.shape[0] != source_bands:
        raise ValueError(f"Source cube has {cube.shape[0]} bands, expected {source_bands}")
    if indices is None:
        return cube
    return cube[validate_band_indices(indices, cube.shape[0])]
