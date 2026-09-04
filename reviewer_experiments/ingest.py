from __future__ import annotations

import json
from pathlib import Path
import re

import pandas as pd


NAME_PATTERN = re.compile(
    r"^Plant(?P<plant>\d+)_(?P<treatment>Infected|Healthy)_Leaf(?P<leaf>\d+)_Day(?P<dpi>[246])$",
    re.IGNORECASE,
)


def _hsi_index(root: str | Path | None) -> dict[str, Path]:
    if root is None:
        return {}
    root = Path(root).expanduser().resolve()
    candidates: dict[str, Path] = {}
    duplicates: set[str] = set()
    for suffix in ("*.npy", "*.npz"):
        for path in root.rglob(suffix):
            key = path.stem.lower()
            if key in candidates:
                duplicates.add(key)
            candidates[key] = path.resolve()
    if duplicates:
        raise ValueError(f"duplicate HSI stems (first 10): {sorted(duplicates)[:10]}")
    return candidates


def build_run1_metadata(
    rgb_root: str | Path,
    output: str | Path,
    hsi_root: str | Path | None = None,
    run_id: str = "run1",
) -> dict[str, object]:
    rgb_root = Path(rgb_root).expanduser().resolve()
    if not rgb_root.is_dir():
        raise FileNotFoundError(f"RGB root does not exist: {rgb_root}")
    hsi_by_stem = _hsi_index(hsi_root)
    rows = []
    unmatched = []
    malformed = []
    for rgb_path in sorted(rgb_root.rglob("*.RGB888")):
        if rgb_path.stem.lower() == "dark":
            continue
        match = NAME_PATTERN.match(rgb_path.stem)
        if match is None:
            malformed.append(str(rgb_path))
            continue
        values = match.groupdict()
        dpi = int(values["dpi"])
        treatment = "infected" if values["treatment"].lower() == "infected" else "mock"
        source_plant_number = int(values["plant"])
        # Healthy files are numbered 1-24 (day 2), 25-48 (day 4), and
        # 49-72 (day 6), although the experiment contains the same 24 mock
        # plants measured repeatedly. Normalize them to biological plant IDs.
        plant_number = source_plant_number if treatment == "infected" else (source_plant_number - 1) % 24 + 1
        leaf_number = int(values["leaf"])
        hsi_path = hsi_by_stem.get(rgb_path.stem.lower())
        if hsi_path is None:
            unmatched.append(rgb_path.stem)
        rows.append({
            "sample_id": f"{run_id}_{treatment}_p{plant_number:03d}_l{leaf_number}_d{dpi}",
            "plant_id": f"{run_id}_{treatment}_p{plant_number:03d}",
            "source_plant_number": source_plant_number,
            "leaf_id": f"L{leaf_number}",
            "treatment": treatment,
            "dpi": dpi,
            "acquisition_date": f"{run_id}_day{dpi}",
            "imaging_session": f"{run_id}_day{dpi}",
            "rgb_path": str(rgb_path),
            "hsi_path": str(hsi_path) if hsi_path is not None else "",
            "hsi_layout": "auto",
            "symptom_status": "presymptomatic" if dpi in (2, 4) else "symptomatic",
        })
    if malformed:
        raise ValueError(f"unrecognized RGB names (first 10): {malformed[:10]}")
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"no Run-1 RGB files found beneath {rgb_root}")
    if frame.sample_id.duplicated().any():
        raise ValueError("parsed sample IDs are not unique")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    summary: dict[str, object] = {
        "samples": int(len(frame)),
        "plants": int(frame.plant_id.nunique()),
        "counts_by_dpi_treatment": frame.groupby(["dpi", "treatment"]).size().rename("samples").reset_index().to_dict("records"),
        "hsi_files_indexed": len(hsi_by_stem),
        "hsi_matches": int(frame.hsi_path.ne("").sum()),
        "hsi_missing": len(unmatched),
        "multimodal_ready": len(unmatched) == 0,
        "date_note": "acquisition_date currently uses run/day identifiers; replace with real calendar dates if available.",
        "mock_id_note": "Healthy source IDs 1-24/25-48/49-72 were normalized to the same 24 biological plants across days.",
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
