from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold


REQUIRED_COLUMNS = {
    "sample_id",
    "plant_id",
    "leaf_id",
    "treatment",
    "dpi",
    "acquisition_date",
    "imaging_session",
    "rgb_path",
    "hsi_path",
    "symptom_status",
}
TASK_DPI = {
    "dpi_2": {2},
    "dpi_4": {4},
    "dpi_6": {6},
    "presymptomatic_2_4": {2, 4},
    "all_dpi": {2, 4, 6},
}
TREATMENT_TO_LABEL = {"mock": 0, "infected": 1}


def read_metadata(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        dtype={"sample_id": str, "plant_id": str, "leaf_id": str},
    )
    validate_metadata(frame)
    frame = frame.copy()
    frame["treatment"] = frame["treatment"].str.lower()
    frame["dpi"] = pd.to_numeric(frame["dpi"]).astype(int)
    frame["label"] = frame["treatment"].map(TREATMENT_TO_LABEL).astype(int)
    return frame


def validate_metadata(frame: pd.DataFrame, require_files: bool = False) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"metadata is missing columns: {missing}")
    if frame.empty:
        raise ValueError("metadata contains no samples")
    if frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
        raise ValueError("sample_id must be non-empty and unique")
    if frame["plant_id"].isna().any():
        raise ValueError("plant_id is required for every sample")

    treatment = frame["treatment"].astype(str).str.lower()
    unknown = sorted(set(treatment) - set(TREATMENT_TO_LABEL))
    if unknown:
        raise ValueError(f"unknown treatment values: {unknown}")
    dpi = pd.to_numeric(frame["dpi"], errors="coerce")
    if dpi.isna().any() or not set(dpi.astype(int)) <= {2, 4, 6}:
        raise ValueError("dpi must contain only 2, 4, and 6")

    normalized = frame.assign(treatment=treatment, dpi=dpi.astype(int))
    switched = normalized.groupby("plant_id")["treatment"].nunique()
    if (switched > 1).any():
        raise ValueError("a plant cannot switch treatment")
    for value in (2, 4, 6):
        present = set(normalized.loc[normalized["dpi"] == value, "treatment"])
        if present != {"mock", "infected"}:
            raise ValueError(
                f"dpi={value} requires date-matched mock and infected samples; found {sorted(present)}"
            )

    if require_files:
        metadata_root = Path(frame.attrs.get("metadata_root", "."))
        empty = [
            f"{column}:<empty>"
            for column in ("rgb_path", "hsi_path")
            for value in frame[column]
            if pd.isna(value) or not str(value).strip()
        ]
        if empty:
            raise ValueError(f"empty referenced paths (first 10): {empty[:10]}")
        absent: list[str] = []
        for column in ("rgb_path", "hsi_path"):
            for value in frame[column]:
                candidate = Path(str(value)).expanduser()
                if not candidate.is_absolute():
                    candidate = metadata_root / candidate
                if not candidate.exists():
                    absent.append(str(candidate))
        if absent:
            raise ValueError(f"missing referenced files (first 10): {absent[:10]}")


def task_frame(frame: pd.DataFrame, task: str) -> pd.DataFrame:
    if task not in TASK_DPI:
        raise ValueError(f"unknown task {task!r}; choose from {sorted(TASK_DPI)}")
    return frame[frame["dpi"].isin(TASK_DPI[task])].copy()


def assign_plant_folds(frame: pd.DataFrame, n_splits: int, seed: int) -> pd.DataFrame:
    plants = frame[["plant_id", "label", "treatment"]].drop_duplicates().reset_index(drop=True)
    if plants["plant_id"].duplicated().any():
        raise ValueError("each plant must have one label")
    counts = plants["label"].value_counts()
    if len(counts) != 2 or counts.min() < n_splits:
        raise ValueError(
            f"need at least {n_splits} plants in each treatment; found {counts.to_dict()}"
        )
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_by_plant: dict[str, int] = {}
    for fold, (_, validation) in enumerate(splitter.split(plants, plants["label"]), start=1):
        for plant_id in plants.iloc[validation]["plant_id"]:
            fold_by_plant[str(plant_id)] = fold
    output = frame.copy()
    output["fold"] = output["plant_id"].map(fold_by_plant).astype(int)
    if output.groupby("plant_id")["fold"].nunique().max() != 1:
        raise RuntimeError("plant leakage detected")
    return output


def prepare_tasks(
    metadata: str | Path,
    output: str | Path,
    n_splits: int = 5,
    seed: int = 157,
    require_files: bool = False,
) -> None:
    metadata = Path(metadata)
    raw = pd.read_csv(metadata, dtype={"sample_id": str, "plant_id": str, "leaf_id": str})
    raw.attrs["metadata_root"] = str(metadata.parent)
    validate_metadata(raw, require_files=require_files)
    frame = read_metadata(metadata)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {}
    for task in TASK_DPI:
        assigned = assign_plant_folds(task_frame(frame, task), n_splits, seed)
        assigned.to_csv(output / f"{task}.csv", index=False)
        plants = assigned[["plant_id", "treatment", "fold"]].drop_duplicates()
        summary[task] = {
            "samples": int(len(assigned)),
            "plants": int(plants["plant_id"].nunique()),
            "plants_by_treatment": plants["treatment"].value_counts().to_dict(),
            "plants_by_fold": plants["fold"].value_counts().sort_index().to_dict(),
        }
    (output / "split_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
