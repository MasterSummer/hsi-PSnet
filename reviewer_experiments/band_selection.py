"""Fit-only ranking by absolute difference between group-mean plant spectra."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import load_hsi, resolve_path


def rank_training_bands(frame, data_root=None):
    if frame.empty or not {"plant_id", "label", "hsi_path"} <= set(frame.columns):
        raise ValueError("Band ranking requires nonempty fitting metadata with plant_id, label, hsi_path")
    if frame.groupby("plant_id").label.nunique().gt(1).any() or set(frame.label) != {0, 1}:
        raise ValueError("Band ranking requires consistent plant labels and both treatments")
    spectra = []
    total_bands = None
    for row in frame.itertuples(index=False):
        cube = load_hsi(resolve_path(row.hsi_path, data_root), getattr(row, "hsi_layout", "CHW"))
        if total_bands is None:
            total_bands = cube.shape[0]
        if cube.shape[0] != total_bands or not np.isfinite(cube).all():
            raise ValueError("Inconsistent band count or non-finite values in fitting spectra")
        spectra.append(cube.mean(axis=(1, 2), dtype=np.float64))
    bands = list(range(total_bands))
    values = pd.DataFrame(np.vstack(spectra), columns=bands)
    values["plant_id"] = frame.plant_id.to_numpy()
    values["label"] = frame.label.to_numpy()
    # Match the pooled task's plant-level aggregation; observations equal within plant.
    plants = values.groupby(["plant_id", "label"], as_index=False)[bands].mean()
    a = plants.loc[plants.label.eq(0), bands].mean().to_numpy(float)
    b = plants.loc[plants.label.eq(1), bands].mean().to_numpy(float)
    scores = np.abs(b - a)
    order = np.lexsort((np.arange(total_bands), -scores))
    ranks = np.empty(total_bands, dtype=int)
    ranks[order] = np.arange(1, total_bands + 1)
    return pd.DataFrame(dict(band=bands, mock_mean=a, inoculated_mean=b,
                             absolute_mean_difference=scores, rank=ranks))


def top_band_indices(ranking, top_k):
    if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= len(ranking):
        raise ValueError(f"top_k must be between 1 and {len(ranking)}")
    return sorted(ranking.nsmallest(top_k, "rank").band.astype(int).tolist())
