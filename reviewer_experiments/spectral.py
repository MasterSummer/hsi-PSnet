from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .core import read_metadata
from .data import load_hsi, resolve_path


def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return float("nan")
    pooled = np.sqrt(((n1 - 1) * np.var(a, ddof=1) + (n2 - 1) * np.var(b, ddof=1)) / (n1 + n2 - 2))
    if pooled == 0:
        return 0.0
    correction = 1 - 3 / (4 * (n1 + n2) - 9)
    return float(correction * (np.mean(b) - np.mean(a)) / pooled)


def bh_fdr(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    ranked = pvalues[order]
    adjusted = np.minimum.accumulate((ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.minimum(adjusted, 1.0)
    return output


def confidence_interval(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return float("nan"), float("nan")
    mean = np.mean(values)
    margin = stats.t.ppf(0.975, len(values) - 1) * stats.sem(values)
    return float(mean - margin), float(mean + margin)


def spectral_analysis(metadata: str | Path, output: str | Path, data_root: str | Path | None = None, wavelengths: str | Path | None = None) -> None:
    frame = read_metadata(metadata)
    sample_rows = []
    spectra = []
    for row in frame.itertuples(index=False):
        cube = load_hsi(resolve_path(row.hsi_path, data_root), getattr(row, "hsi_layout", "auto"))
        spectra.append(cube.mean(axis=(1, 2)))
        sample_rows.append({"plant_id": row.plant_id, "treatment": row.treatment, "dpi": row.dpi})
    spectrum = pd.DataFrame(np.vstack(spectra))
    identifiers = pd.DataFrame(sample_rows)
    plants = pd.concat([identifiers, spectrum], axis=1).groupby(["plant_id", "treatment", "dpi"], as_index=False).mean()
    band_columns = [column for column in plants.columns if isinstance(column, int)]
    wavelength_values = np.arange(len(band_columns), dtype=float)
    wavelength_label = "band_index"
    if wavelengths is not None:
        table = pd.read_csv(wavelengths)
        if not {"band", "wavelength_nm"} <= set(table.columns) or len(table) != len(band_columns):
            raise ValueError("wavelength CSV must have one band,wavelength_nm row per HSI band")
        wavelength_values = table.sort_values("band")["wavelength_nm"].to_numpy(float)
        wavelength_label = "wavelength_nm"
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    export_plants = plants.copy()
    export_plants.columns = [str(column) for column in export_plants.columns]
    export_plants.to_csv(output / "plant_mean_spectra.csv", index=False)
    rows = []
    for dpi, dpi_frame in plants.groupby("dpi"):
        mock = dpi_frame[dpi_frame.treatment == "mock"]
        infected = dpi_frame[dpi_frame.treatment == "infected"]
        pvalues = []
        local = []
        for index, band in enumerate(band_columns):
            a, b = mock[band].to_numpy(float), infected[band].to_numpy(float)
            mock_ci, infected_ci = confidence_interval(a), confidence_interval(b)
            pvalue = float(stats.ttest_ind(a, b, equal_var=False).pvalue)
            pvalues.append(pvalue)
            local.append({
                "dpi": int(dpi), "band": index, wavelength_label: wavelength_values[index],
                "mock_mean": float(np.mean(a)), "mock_ci_low": mock_ci[0], "mock_ci_high": mock_ci[1],
                "infected_mean": float(np.mean(b)), "infected_ci_low": infected_ci[0], "infected_ci_high": infected_ci[1],
                "hedges_g_infected_minus_mock": hedges_g(a, b), "welch_p": pvalue,
            })
        adjusted = bh_fdr(np.asarray(pvalues))
        for row, qvalue in zip(local, adjusted):
            row["fdr_q"] = float(qvalue)
            rows.append(row)
    pd.DataFrame(rows).to_csv(output / "within_dpi_band_statistics.csv", index=False)

    red_rows = []
    note = {"red_edge": "skipped: provide --wavelengths with physical wavelengths"}
    if wavelengths is not None:
        selected = np.where((wavelength_values >= 680) & (wavelength_values <= 750))[0]
        if len(selected) >= 3:
            for _, row in plants.iterrows():
                reflectance = row[band_columns].to_numpy(dtype=float)
                derivative = np.gradient(reflectance[selected], wavelength_values[selected])
                maximum = int(np.argmax(derivative))
                red_rows.append({"plant_id": row["plant_id"], "treatment": row["treatment"], "dpi": row["dpi"],
                                 "red_edge_position_nm": wavelength_values[selected][maximum], "red_edge_max_slope": derivative[maximum]})
            note = {"red_edge": "computed on 680-750 nm first derivative"}
    red_frame = pd.DataFrame(red_rows)
    red_frame.to_csv(output / "red_edge_plant_metrics.csv", index=False)
    red_statistics = []
    if not red_frame.empty:
        for dpi, dpi_frame in red_frame.groupby("dpi"):
            for metric in ("red_edge_position_nm", "red_edge_max_slope"):
                a = dpi_frame[dpi_frame.treatment == "mock"][metric].to_numpy(float)
                b = dpi_frame[dpi_frame.treatment == "infected"][metric].to_numpy(float)
                red_statistics.append({"dpi": int(dpi), "metric": metric,
                                       "mock_mean": float(np.mean(a)), "infected_mean": float(np.mean(b)),
                                       "hedges_g_infected_minus_mock": hedges_g(a, b),
                                       "welch_p": float(stats.ttest_ind(a, b, equal_var=False).pvalue)})
    pd.DataFrame(red_statistics).to_csv(output / "red_edge_statistics.csv", index=False)
    (output / "spectral_analysis_notes.json").write_text(json.dumps(note, indent=2), encoding="utf-8")
