#!/usr/bin/env python3
"""Plant-level spectral statistics; no training, GPU, or image resizing required."""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import platform
import re

import numpy as np
import pandas as pd
import scipy
from scipy import stats
from .bands import parse_bands


KEYS = ["plant_id", "treatment", "dpi"]


def require(frame, columns):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if frame.empty or frame[list(columns)].isna().any().any():
        raise ValueError("Empty input or missing required values")
    for col in columns:
        if frame[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"Empty values in {col}")


def validate_identifiers(frame):
    require(frame, KEYS)
    frame = frame.copy()
    frame["plant_id"] = frame.plant_id.astype(str).str.strip()
    frame["treatment"] = frame.treatment.astype(str).str.strip().str.lower()
    if not set(frame.treatment) <= {"mock", "infected"}:
        raise ValueError("treatment must be mock or infected (inoculation groups)")
    dpi = pd.to_numeric(frame.dpi, errors="raise")
    if not np.isfinite(dpi).all() or (dpi < 0).any() or (dpi % 1 != 0).any():
        raise ValueError("dpi must be nonnegative integers")
    frame["dpi"] = dpi.astype(int)
    if frame.groupby("plant_id").treatment.nunique().gt(1).any():
        raise ValueError("A plant_id occurs in multiple treatments; use globally unique IDs")
    return frame


def bh_fdr(pvalues):
    """Undefined tests retain NaN and conservatively count in family size."""
    p = np.asarray(pvalues, dtype=float)
    valid = np.isfinite(p)
    if ((p[valid] < 0) | (p[valid] > 1)).any():
        raise ValueError("Invalid p value")
    q = np.full(p.shape, np.nan)
    indices = np.flatnonzero(valid)
    indices = indices[np.argsort(p[indices], kind="stable")]
    if indices.size:
        adjusted = p[indices] * p.size / np.arange(1, indices.size + 1)
        q[indices] = np.minimum(1, np.minimum.accumulate(adjusted[::-1])[::-1])
    return q


def band_statistics(mock, infected):
    """Effect direction is infected minus mock; t intervals are pointwise."""
    a, b = np.asarray(mock, float), np.asarray(infected, float)
    if min(len(a), len(b)) < 2:
        raise ValueError("At least two plants per treatment per dpi are required")
    na, nb = len(a), len(b)
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    da = stats.t.ppf(.975, na - 1) * np.sqrt(va / na)
    db = stats.t.ppf(.975, nb - 1) * np.sqrt(vb / nb)
    diff = mb - ma
    se2 = va / na + vb / nb
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    g = (1 - 3 / (4 * (na + nb) - 9)) * diff / pooled if pooled > 0 else np.nan
    if se2 > 0:
        df = se2**2 / ((va / na)**2 / (na - 1) + (vb / nb)**2 / (nb - 1))
        t = diff / np.sqrt(se2)
        p = 2 * stats.t.sf(abs(t), df)
        margin = stats.t.ppf(.975, df) * np.sqrt(se2)
        low, high, status = diff - margin, diff + margin, "ok"
    else:
        t = df = p = low = high = np.nan
        status = "undefined_zero_variance"
    return dict(n_mock=na, n_infected=nb, mock_mean=ma, infected_mean=mb,
                mock_sd=np.sqrt(va), infected_sd=np.sqrt(vb),
                mock_ci_low=ma-da, mock_ci_high=ma+da,
                infected_ci_low=mb-db, infected_ci_high=mb+db,
                mean_difference=diff, difference_ci_low=low, difference_ci_high=high,
                hedges_g_infected_minus_mock=g, welch_t=t, welch_df=df,
                welch_p=p, test_status=status)


def local_path(value, root):
    p = Path(str(value)).expanduser()
    return p if p.is_absolute() else root / p


def numpy_array(path):
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        with loaded:
            if len(loaded.files) != 1:
                raise ValueError(f"NPZ must have exactly one array: {path}")
            return loaded[loaded.files[0]]
    return loaded


@lru_cache(maxsize=2)
def pt_bundle(path):
    import torch  # Optional; only needed for existing local split_4re bundles.
    return torch.load(path, map_location="cpu", weights_only=False)


def cube_array(value, root, layout):
    if str(value).startswith("ptbundle:"):
        path, index = str(value)[len("ptbundle:"):].rsplit("::", 1)
        bundle = pt_bundle(str(local_path(path, root)))
        index = int(index)
        if not 0 <= index < len(bundle):
            raise ValueError("PT bundle index out of range")
        item = bundle[index]
        if not isinstance(item, (tuple, list)) or len(item) < 2:
            raise ValueError("Expected a split_4re record with HSI at item[1]")
        cube = item[1]
        if hasattr(cube, "detach"):
            cube = cube.detach().cpu().numpy()
    else:
        cube = numpy_array(local_path(value, root))
    cube = np.asarray(cube)
    if cube.ndim != 3:
        raise ValueError(f"Expected exactly 3 dimensions, got {cube.shape}")
    if layout == "HWC":
        cube = np.moveaxis(cube, -1, 0)
    elif layout != "CHW":
        raise ValueError("Specify hsi_layout=CHW or HWC; automatic axis guessing is disabled")
    return cube


def aggregate_samples(samples, bands):
    # Repeated acquisitions do not give one leaf greater biological weight.
    leaves = samples.groupby(KEYS + ["leaf_id"], as_index=False)[bands].mean()
    plants = leaves.groupby(KEYS, as_index=False)[bands].mean()
    counts = leaves.groupby(KEYS).size().rename("n_leaves").reset_index()
    return plants.merge(counts, on=KEYS, validate="one_to_one")


def metadata_from_pt(paths, layout="CHW", mask_root=None):
    """Read this project's (RGB path, HSI tensor, four-class label) records."""
    pattern = re.compile(r"^Plant(?P<plant>\d+)_(?P<treatment>Infected|Healthy)_Leaf(?P<leaf>\d+)_Day(?P<dpi>[246])$", re.I)
    rows = []
    resolved = [p.expanduser().resolve() for p in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("The same PT file was supplied more than once")
    for path in resolved:
        print(f"Reading {path}", flush=True)
        bundle = pt_bundle(str(path))
        if not isinstance(bundle, (list, tuple)) or not bundle:
            raise ValueError(f"{path}: expected a nonempty list of (rgb_path, hsi, label) records, not model weights")
        for index, item in enumerate(bundle):
            if not isinstance(item, (tuple, list)) or len(item) < 3:
                raise ValueError(f"{path}[{index}]: expected (rgb_path, hsi, label)")
            rgb_path, hsi, label = item[:3]
            # RGB files need not exist; only their stored names provide metadata.
            stem = Path(str(rgb_path).replace("\\", "/")).stem
            match = pattern.fullmatch(stem)
            if not match:
                raise ValueError(f"Cannot recover plant/leaf/dpi from {rgb_path!r}. Expected Plant7_Infected_Leaf3_Day2.*; class label alone cannot recover mock acquisition dates or biological IDs.")
            info = match.groupdict()
            treatment = "infected" if info["treatment"].lower() == "infected" else "mock"
            dpi, plant, leaf = int(info["dpi"]), int(info["plant"]), int(info["leaf"])
            if plant < 1 or leaf < 1:
                raise ValueError(f"Nonpositive plant/leaf ID: {stem}")
            expected = 0 if treatment == "mock" else {2: 1, 4: 2, 6: 3}[dpi]
            scalar = label.item() if hasattr(label, "item") else label
            if float(scalar) != expected:
                raise ValueError(f"{path}[{index}]: stored label {scalar} disagrees with filename (expected {expected})")
            if not isinstance(hsi, np.ndarray) and not hasattr(hsi, "detach"):
                raise ValueError(f"{path}[{index}]: HSI must be a tensor or ndarray")
            record = dict(sample_id=f"run1_{treatment}_p{plant:03d}_l{leaf}_d{dpi}",
                          plant_id=f"run1_{treatment}_p{plant:03d}", leaf_id=str(leaf),
                          treatment=treatment, dpi=dpi, source_plant_number=plant,
                          hsi_path=f"ptbundle:{path}::{index}", hsi_layout=layout,
                          original_rgb_path=str(rgb_path), source_bundle=str(path), source_index=index)
            if mask_root is not None:
                record["mask_path"] = str((mask_root.expanduser().resolve() / f"{stem}.npy"))
            rows.append(record)
        print(f"  {len(bundle)} observations", flush=True)
    frame = pd.DataFrame(rows)
    if frame.sample_id.duplicated().any():
        duplicates = frame.loc[frame.sample_id.duplicated(False), "sample_id"].tolist()
        raise ValueError(f"Duplicate biological observations across PT inputs: {duplicates[:10]}")
    return frame


def extract_spectra(args, frame=None):
    if frame is None:
        frame = pd.read_csv(args.metadata, dtype={"plant_id": str, "leaf_id": str, "sample_id": str})
    frame = validate_identifiers(frame)
    require(frame, ["sample_id", "leaf_id", "hsi_path"])
    if frame.sample_id.duplicated().any() or frame.hsi_path.duplicated().any():
        raise ValueError("Duplicate sample_id or hsi_path; check duplicated acquisitions")
    if args.region != "whole":
        require(frame, ["mask_path"])
    root = args.data_root or (args.metadata.resolve().parent if args.metadata else Path.cwd())
    rows, audit = [], []
    bands = [str(i) for i in range(args.expected_bands)]
    for record in frame.to_dict("records"):
        layout = args.layout or str(record.get("hsi_layout", "")).upper()
        cube = cube_array(record["hsi_path"], root, layout)
        if cube.shape[0] != args.expected_bands:
            raise ValueError(f"{record['sample_id']}: band count {cube.shape[0]} != {args.expected_bands}")
        mask = np.ones(cube.shape[1:], dtype=bool)
        if args.region != "whole":
            mask = np.asarray(numpy_array(local_path(record["mask_path"], root)))
            if mask.shape != cube.shape[1:] or not np.isin(mask, [0, 1]).all():
                raise ValueError("Mask must be a registered H x W binary 0/1 NPY or NPZ")
            mask = mask.astype(bool)
            if args.region == "background":
                mask = ~mask
        if not mask.any():
            raise ValueError(f"Empty selected region: {record['sample_id']}")
        pixels = np.asarray(cube[:, mask], dtype=np.float64)
        if not np.isfinite(pixels).all():
            raise ValueError(f"NaN/Inf in selected pixels: {record['sample_id']}; clean and document upstream")
        means = pixels.mean(axis=1)
        rows.append({**{k: record[k] for k in KEYS + ["sample_id", "leaf_id"]}, **dict(zip(bands, means))})
        audit.append({**record, "selected_pixels": int(mask.sum()), "total_pixels": int(mask.size),
                      "region": args.region, "min_value": float(pixels.min()), "max_value": float(pixels.max())})
    samples = pd.DataFrame(rows)
    return aggregate_samples(samples, bands), bands, samples, pd.DataFrame(audit)


def load_plants(path):
    plants = validate_identifiers(pd.read_csv(path, dtype={"plant_id": str}))
    if plants.duplicated(["plant_id", "dpi"]).any():
        raise ValueError("Plant CSV requires exactly one row per plant_id and dpi")
    bands = sorted([c for c in plants.columns if c.isdigit()], key=int)
    if not bands or len(set(map(int, bands))) != len(bands):
        raise ValueError("Band columns must be unique integer indices, e.g. 0,1,2")
    plants[bands] = plants[bands].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(plants[bands].to_numpy(float)).all():
        raise ValueError("Plant spectra contain NaN/Inf")
    return plants, bands


def analyze(plants, bands, wavelengths, alpha):
    rows = []
    for dpi, group in plants.groupby("dpi", sort=True):
        for band in bands:
            a = group.loc[group.treatment == "mock", band].to_numpy(float)
            b = group.loc[group.treatment == "infected", band].to_numpy(float)
            rows.append(dict(dpi=int(dpi), band=int(band), wavelength_nm=wavelengths.get(int(band), np.nan),
                             **band_statistics(a, b)))
    result = pd.DataFrame(rows)
    result = result.sort_values(["dpi", "band"]).reset_index(drop=True)
    result["absolute_mean_difference"] = result.mean_difference.abs()
    result["difference_rank"] = result.groupby("dpi").absolute_mean_difference.rank(method="first", ascending=False).astype(int)
    for k in (3, 10, 30):
        result[f"descriptive_top{k}"] = result.difference_rank.le(k)
    result["fdr_q"] = result.groupby("dpi").welch_p.transform(lambda p: bh_fdr(p.to_numpy()))
    result["fdr_q_all_dpi"] = bh_fdr(result.welch_p.to_numpy())
    summary = []
    for dpi, group in result.groupby("dpi"):
        summary.append(dict(dpi=int(dpi), n_mock=int(group.n_mock.iloc[0]), n_infected=int(group.n_infected.iloc[0]),
                            n_bands=len(group), n_valid_tests=int(group.welch_p.notna().sum()),
                            significant_bands_within_dpi=int(group.fdr_q.lt(alpha).sum()),
                            significant_bands_all_dpi=int(group.fdr_q_all_dpi.lt(alpha).sum()),
                            minimum_q=group.fdr_q.min(),
                            median_abs_hedges_g=group.hedges_g_infected_minus_mock.abs().median()))
    return result, pd.DataFrame(summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--plant-csv", type=Path)
    source.add_argument("--metadata", type=Path)
    source.add_argument("--split-dir", type=Path, help="Directory containing trainval.pt and cleantest.pt; no CSV input required")
    source.add_argument("--pt-files", nargs="+", type=Path, help="One or more split_4re-format PT files; no CSV input required")
    parser.add_argument("--output", type=Path, required=True, help="New or empty output directory")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--layout", choices=["CHW", "HWC"])
    parser.add_argument("--mask-root", type=Path, help="Direct PT mode: directory of <original RGB stem>.npy binary masks")
    parser.add_argument("--expected-bands", type=int, default=313)
    parser.add_argument("--bands", default="all", help="Original zero-based positions: all, 0:104, or 0:100,200:313; stop excluded")
    parser.add_argument("--region", choices=["whole", "leaf", "background"], default="whole")
    parser.add_argument("--wavelengths", type=Path, help="CSV: band,wavelength_nm; partial mapping allowed")
    parser.add_argument("--calibrated-only", action="store_true", help="Only analyze bands with supplied physical wavelengths")
    parser.add_argument("--alpha", type=float, default=.05)
    args = parser.parse_args()
    if not 0 < args.alpha < 1 or args.expected_bands < 1:
        parser.error("Require 0 < alpha < 1 and positive expected-bands")
    if args.plant_csv and (args.region != "whole" or args.layout or args.data_root):
        parser.error("Region extraction, layout and data-root require --metadata; CSV cannot reconstruct pixels")
    direct_pt = bool(args.split_dir or args.pt_files)
    if args.mask_root and not direct_pt:
        parser.error("--mask-root is for direct PT inputs; metadata mode uses mask_path")
    if direct_pt and args.region != "whole" and not args.mask_root:
        parser.error("Direct PT leaf/background extraction requires --mask-root")
    pt_paths = ([args.split_dir / "trainval.pt", args.split_dir / "cleantest.pt"]
                if args.split_dir else (args.pt_files or []))
    if args.calibrated_only and not args.wavelengths:
        parser.error("--calibrated-only requires a measured --wavelengths table")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory is not empty; choose a new directory")
    if direct_pt:
        frame = metadata_from_pt(pt_paths, layout=args.layout or "CHW", mask_root=args.mask_root)
        plants, bands, samples, audit = extract_spectra(args, frame)
    elif args.plant_csv:
        plants, bands = load_plants(args.plant_csv)
        samples = audit = None
    else:
        plants, bands, samples, audit = extract_spectra(args)
    wavelengths = {}
    original_bands = list(bands)
    if args.wavelengths:
        table = pd.read_csv(args.wavelengths)
        require(table, ["band", "wavelength_nm"])
        band_ids = pd.to_numeric(table.band, errors="raise")
        values = pd.to_numeric(table.wavelength_nm, errors="raise")
        if not np.isfinite(band_ids).all() or (band_ids % 1 != 0).any():
            raise ValueError("Wavelength band indices must be finite integers")
        if band_ids.duplicated().any() or not set(band_ids.astype(int)) <= set(map(int, bands)):
            raise ValueError("Wavelength mapping has duplicate or unknown band indices")
        if not np.isfinite(values).all() or (values <= 0).any() or values.duplicated().any():
            raise ValueError("Wavelengths must be positive, finite and unique")
        wavelengths = dict(zip(band_ids.astype(int), values.astype(float)))
        if args.calibrated_only:
            bands = [b for b in bands if int(b) in wavelengths]
    if args.bands != "all":
        selected = parse_bands(args.bands, max(map(int, original_bands)) + 1)
        if not set(selected) <= set(map(int, original_bands)):
            raise ValueError("Requested band indices are absent from input")
        bands = [b for b in bands if int(b) in selected]
    if not bands:
        raise ValueError("No bands remain after selection and calibration filtering")
    result, summary = analyze(plants, bands, wavelengths, args.alpha)
    args.output.mkdir(parents=True, exist_ok=True)
    identifiers = [c for c in plants.columns if c not in original_bands]
    plants[identifiers + bands].to_csv(args.output / "plant_mean_spectra.csv", index=False)
    result.to_csv(args.output / "within_dpi_band_statistics.csv", index=False)
    summary.to_csv(args.output / "summary_by_dpi.csv", index=False)
    if samples is not None:
        identifiers = [c for c in samples.columns if c not in original_bands]
        samples[identifiers + bands].to_csv(args.output / "sample_mean_spectra.csv", index=False)
        audit.to_csv(args.output / "sample_qc.csv", index=False)
    inputs = [p for p in [args.plant_csv, args.metadata, args.wavelengths] if p]
    notes = dict(arguments={k: ([str(p) for p in v] if isinstance(v, list) else str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                 input_table_sha256={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                 script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__),
                 statistical_unit="plant within dpi; acquisitions averaged within leaf, then leaves equally weighted",
                 effect_direction="infected (inoculated) minus mock", ci="pointwise 95% Student t; not simultaneous",
                 fdr="Primary BH per dpi; all-dpi BH also exported as sensitivity analysis",
                 selected_bands=list(map(int, bands)),
                 ranking_note="Descriptive absolute mean-difference ranks are within dpi over all supplied plants; do not use these ranks for selecting classifier inputs. Model runs rank fitting plants separately in each fold.",
                 pt_sources=[dict(path=str(p.expanduser().resolve()), bytes=p.expanduser().stat().st_size) for p in pt_paths],
                 pt_identity_note="Direct PT mode retains source plant numbers without modulo mapping. All tests and aggregation are within dpi; cross-date mock identity is not inferred. Use only files from the same Run 1 experiment.",
                 limitations=["No automatic wavelength interpolation or radiometric calibration",
                              "Plant CSV preserves upstream extraction; masks cannot be applied retroactively",
                              "No acquisition-confound adjustment; group differences are not causal attribution",
                              "No across-date hypothesis test; nonsignificance does not establish absence of differences",
                              "Masks must be independently defined, registered to the HSI spatial grid and checked",
                              "Input-table hashes do not hash the contents of referenced cubes or masks"])
    (args.output / "analysis_notes.json").write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"\nSaved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
