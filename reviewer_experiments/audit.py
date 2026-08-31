from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .core import read_metadata


def metadata_audit(metadata: str | Path, output: str | Path) -> None:
    frame = read_metadata(metadata)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    counts = frame.groupby(["dpi", "treatment"], as_index=False).agg(
        samples=("sample_id", "nunique"), plants=("plant_id", "nunique"),
        leaves=("leaf_id", "nunique"), dates=("acquisition_date", "nunique"),
        sessions=("imaging_session", "nunique"),
    )
    counts.to_csv(output / "exact_sample_plant_counts.csv", index=False)
    pd.crosstab([frame.dpi, frame.acquisition_date, frame.imaging_session], frame.treatment).to_csv(
        output / "treatment_by_date_session.csv"
    )
    acquisition_columns = [column for column in ("exposure_ms", "exposure_us", "gain", "white_reference_id") if column in frame]
    if acquisition_columns:
        numeric = [column for column in acquisition_columns if column != "white_reference_id"]
        if numeric:
            frame.groupby(["dpi", "treatment"])[numeric].agg(["count", "mean", "std", "min", "max"]).to_csv(
                output / "exposure_gain_summary.csv"
            )
        frame[["sample_id", "plant_id", "dpi", "treatment", *acquisition_columns]].to_csv(
            output / "acquisition_settings_by_sample.csv", index=False
        )
    summary = {
        "samples": int(frame.sample_id.nunique()), "plants": int(frame.plant_id.nunique()),
        "leaves": int(frame.leaf_id.nunique()), "dates": int(frame.acquisition_date.nunique()),
        "sessions": int(frame.imaging_session.nunique()),
        "repeated_samples_per_plant": frame.groupby("plant_id").size().describe().to_dict(),
        "available_acquisition_columns": acquisition_columns,
        "missing_acquisition_columns": [
            name for name, present in (
                ("exposure_ms_or_us", "exposure_ms" in frame or "exposure_us" in frame),
                ("gain", "gain" in frame),
                ("white_reference_id", "white_reference_id" in frame),
            ) if not present
        ],
        "interpretation_limit": "Plant-disjoint folds estimate within-run generalization only; they are not an external experiment.",
    }
    (output / "metadata_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
