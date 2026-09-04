from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch

from .data import make_pt_bundle_uri


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
    rgb_files = sorted(
        path for path in rgb_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".rgb888", ".jpg", ".jpeg"}
    )
    for rgb_path in rgb_files:
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


def _load_bundle(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _rgb_by_stem(root: str | Path | None) -> dict[str, Path]:
    if root is None:
        return {}
    root = Path(root).expanduser().resolve()
    result: dict[str, Path] = {}
    duplicate: set[str] = set()
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".rgb888"}:
            key = path.stem.lower()
            if key in result:
                duplicate.add(key)
            result[key] = path.resolve()
    if duplicate:
        raise ValueError(f"duplicate RGB stems beneath {root} (first 10): {sorted(duplicate)[:10]}")
    return result


def build_split4re_metadata(
    split_dir: str | Path,
    output: str | Path,
    rgb_root: str | Path | None = None,
    run_id: str = "run1",
) -> dict[str, object]:
    split_dir = Path(split_dir).expanduser().resolve()
    bundle_paths = [split_dir / "trainval.pt", split_dir / "cleantest.pt"]
    missing_bundles = [str(path) for path in bundle_paths if not path.is_file()]
    if missing_bundles:
        raise FileNotFoundError(f"split_4re is missing required PT bundles: {missing_bundles}")
    rgb_index = _rgb_by_stem(rgb_root)
    rows = []
    missing_rgb = []
    for bundle_path in bundle_paths:
        source_split = "trainval" if bundle_path.name == "trainval.pt" else "cleantest"
        bundle = _load_bundle(bundle_path)
        for index, item in enumerate(bundle):
            if not isinstance(item, (tuple, list)) or len(item) < 3:
                raise ValueError(f"unexpected item {index} in {bundle_path}")
            original_rgb, hsi, stored_label = item[:3]
            original_rgb = Path(str(original_rgb)).expanduser()
            match = NAME_PATTERN.match(original_rgb.stem)
            if match is None:
                raise ValueError(f"cannot parse plant/leaf/day from PT RGB path: {original_rgb}")
            values = match.groupdict()
            dpi = int(values["dpi"])
            treatment = "infected" if values["treatment"].lower() == "infected" else "mock"
            expected_label = 0 if treatment == "mock" else {2: 1, 4: 2, 6: 3}[dpi]
            if int(stored_label) != expected_label:
                raise ValueError(
                    f"label mismatch in {bundle_path.name}[{index}]: stored={stored_label}, expected={expected_label}"
                )
            source_plant_number = int(values["plant"])
            plant_number = source_plant_number if treatment == "infected" else (source_plant_number - 1) % 24 + 1
            leaf_number = int(values["leaf"])
            rgb_path = original_rgb if original_rgb.is_file() else rgb_index.get(original_rgb.stem.lower())
            if rgb_path is None or not Path(rgb_path).is_file():
                missing_rgb.append(str(original_rgb))
                rgb_path = original_rgb
            if not isinstance(hsi, (torch.Tensor, np.ndarray)):
                raise ValueError(f"HSI item {index} in {bundle_path} is not a tensor/array")
            shape = tuple(hsi.shape)
            if len(shape) != 3:
                raise ValueError(f"HSI item {index} in {bundle_path} is not 3-D: {shape}")
            rows.append({
                "sample_id": f"{run_id}_{treatment}_p{plant_number:03d}_l{leaf_number}_d{dpi}",
                "plant_id": f"{run_id}_{treatment}_p{plant_number:03d}",
                "source_plant_number": source_plant_number,
                "leaf_id": f"L{leaf_number}",
                "treatment": treatment,
                "dpi": dpi,
                "acquisition_date": f"{run_id}_day{dpi}",
                "imaging_session": f"{run_id}_day{dpi}",
                "rgb_path": str(Path(rgb_path).resolve()),
                "hsi_path": make_pt_bundle_uri(bundle_path, index),
                "hsi_layout": "CHW",
                "symptom_status": "presymptomatic" if dpi in (2, 4) else "symptomatic",
                "source_split": source_split,
            })
    frame = pd.DataFrame(rows)
    if frame.sample_id.duplicated().any():
        duplicates = frame.loc[frame.sample_id.duplicated(False), "sample_id"].tolist()
        raise ValueError(f"duplicate biological sample IDs across PT bundles: {duplicates[:10]}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    summary: dict[str, object] = {
        "split_dir": str(split_dir),
        "samples": int(len(frame)),
        "plants": int(frame.plant_id.nunique()),
        "counts_by_dpi_treatment": frame.groupby(["dpi", "treatment"]).size().rename("samples").reset_index().to_dict("records"),
        "missing_rgb": len(missing_rgb),
        "pt_bundles": [str(path) for path in bundle_paths],
        "multimodal_ready": len(missing_rgb) == 0,
        "source_split_note": "Legacy train/test membership is recorded only for provenance; reviewer folds are reassigned by biological plant.",
    }
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
