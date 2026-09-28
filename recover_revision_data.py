"""Export recovered MAT cubes, or merge preprocessed cubes after overlap validation.

No normalization is guessed. All outputs must be new directories. See RERUN_ZH.md.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from PIL import Image
from scipy.io import loadmat

from reviewer_experiments.core import prepare_tasks, read_metadata
from reviewer_experiments.data import load_hsi, resolve_path
from reviewer_experiments.ingest import NAME_PATTERN, _rgb_by_stem


def identify(stem):
    match = NAME_PATTERN.fullmatch(stem)
    if not match:
        raise ValueError(f"Unrecognized sample name: {stem}")
    v = match.groupdict()
    source = int(v['plant'])
    treatment = 'infected' if v['treatment'].lower() == 'infected' else 'mock'
    plant = source if treatment == 'infected' else (source - 1) % 24 + 1
    return dict(sample_id=f"run1_{treatment}_p{plant:03d}_l{v['leaf']}_d{v['dpi']}",
                plant_id=f"run1_{treatment}_p{plant:03d}", source_plant_number=source,
                leaf_id=f"L{v['leaf']}", treatment=treatment, dpi=int(v['dpi']))


def fresh_directory(path):
    path = Path(path).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def export_mat(archive, output):
    output = fresh_directory(output)
    rows = []
    seen = set()
    with zipfile.ZipFile(archive) as z:
        for name in sorted(z.namelist()):
            if not name.lower().endswith('.mat') or name.startswith('__MACOSX/'):
                continue
            stem = Path(name).stem
            record = identify(stem)
            if record['sample_id'] in seen:
                raise ValueError(f"Duplicate observation: {stem}")
            seen.add(record['sample_id'])
            raw = z.read(name)
            mat = loadmat(io.BytesIO(raw), simplify_cells=True)
            cube = np.asarray(mat['H'])
            waves = np.asarray(mat['parameters']['Wavelengths']).reshape(-1)
            if cube.ndim != 3 or cube.shape[-1] != len(waves) or not np.isfinite(cube).all():
                raise ValueError(f"Invalid HWC cube: {name}")
            if not np.isfinite(waves).all() or not (np.diff(waves) > 0).all():
                raise ValueError(f"Invalid nominal wavelength axis: {name}")
            np.save(output / f'{stem}.npy', cube.astype(np.float32), allow_pickle=False)
            np.savetxt(output / f'{stem}.wavelengths.csv', waves, delimiter=',', header='nominal_wavelength_nm', comments='')
            rows.append({**record, 'file': f'{stem}.npy', 'layout': 'HWC', 'shape': list(cube.shape),
                         'mat_sha256': hashlib.sha256(raw).hexdigest(),
                         'min': float(cube.min()), 'max': float(cube.max())})
    if not rows:
        raise ValueError('No named MAT cubes found')
    (output / 'export_audit.json').write_text(json.dumps(dict(samples=rows,
        note='Raw MAT values only. Nominal wavelengths are not a validated calibration or a mapping to historical PT bands. No masking, rescaling or normalization was inferred.'), indent=2) + '\n')
    print(f'Exported {len(rows)} cubes to {output}; not merged into training data.')


def merge(base, recovered, rgb_root, output, layout='HWC'):
    output = fresh_directory(output)
    frame = read_metadata(base)
    # Resolve paths before writing metadata into a different directory.
    for col in ['rgb_path', 'hsi_path']:
        frame[col] = [str(resolve_path(v, Path(base).resolve().parent)) for v in frame[col]]
    rgb = _rgb_by_stem(rgb_root)
    existing = frame.set_index('sample_id')
    checks, additions, seen = [], [], set()
    shape = None
    for path in sorted(Path(recovered).glob('*.npy')):
        record = identify(path.stem)
        sid = record['sample_id']
        if sid in seen:
            raise ValueError(f'Duplicate recovered observation: {sid}')
        seen.add(sid)
        cube = load_hsi(path, layout)
        if not np.isfinite(cube).all():
            raise ValueError(f'Nonfinite cube: {path}')
        shape = cube.shape if shape is None else shape
        if cube.shape != shape:
            raise ValueError('Recovered cube shapes differ')
        if sid in existing.index:
            old = existing.loc[sid]
            for key in ['plant_id', 'leaf_id', 'treatment', 'dpi']:
                if str(record[key]) != str(old[key]):
                    raise ValueError(f'Overlap metadata mismatch: {sid}, {key}')
            reference = load_hsi(old.hsi_path, old.get('hsi_layout', 'CHW'))
            same_shape = reference.shape == cube.shape
            matched = same_shape and np.allclose(reference, cube, rtol=1e-5, atol=1e-6)
            checks.append(dict(sample_id=sid, matched=bool(matched),
                max_abs_difference=float(np.max(np.abs(reference-cube))) if same_shape else None))
            continue
        image = rgb.get(path.stem.lower())
        if image is None:
            raise ValueError(f'Missing corresponding RGB: {path.stem}')
        with Image.open(image) as im:
            im.verify()
        additions.append({**record, 'label': int(record['treatment'] == 'infected'),
            'rgb_path': str(image), 'hsi_path': str(path.resolve()), 'hsi_layout': layout,
            'acquisition_date': f"run1_day{record['dpi']}", 'imaging_session': f"run1_day{record['dpi']}",
            'symptom_status': 'presymptomatic' if record['dpi'] in (2, 4) else 'symptomatic'})
    passed = bool(checks) and all(c['matched'] for c in checks)
    (output / 'overlap_audit.json').write_text(json.dumps(dict(overlaps=checks,
        candidate_additions=len(additions), passed=passed,
        tolerance=dict(rtol=1e-5, atol=1e-6),
        limitation='Overlap agreement is necessary, not proof of preprocessing consistency for every new sample.'), indent=2)+'\n')
    if not passed:
        raise ValueError('Overlap check failed or no overlap supplied. No merged metadata written. Recover the original preprocessing; do not rescale a cube merely to pass this check.')
    if not additions:
        raise ValueError('No new observations to merge')
    # Verify all old cubes have compatible dimensions before mixing sources.
    for row in frame.itertuples():
        if load_hsi(row.hsi_path, getattr(row, 'hsi_layout', 'CHW')).shape != shape:
            raise ValueError('Historical and recovered cube shapes differ')
    merged = pd.concat([frame, pd.DataFrame(additions)], ignore_index=True)
    # A new cohort must get fresh outer folds, never inherit old fold assignments.
    merged = merged.drop(columns=['fold', 'task'], errors='ignore')
    merged.to_csv(output / 'metadata.csv', index=False)
    prepare_tasks(output / 'metadata.csv', output / 'tasks', n_splits=5, seed=157, require_files=True)
    print(f'Added {len(additions)} observations; retained {len(checks)} existing overlaps without duplication. Total {len(merged)}.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    export = sub.add_parser('export')
    export.add_argument('--zip', required=True, type=Path)
    export.add_argument('--output', required=True, type=Path)
    m = sub.add_parser('merge')
    m.add_argument('--metadata', required=True, type=Path)
    m.add_argument('--recovered-dir', required=True, type=Path)
    m.add_argument('--rgb-root', required=True, type=Path)
    m.add_argument('--output', required=True, type=Path)
    m.add_argument('--layout', choices=['HWC', 'CHW'], default='HWC')
    a = p.parse_args()
    if a.command == 'export':
        export_mat(a.zip, a.output)
    else:
        merge(a.metadata, a.recovered_dir, a.rgb_root, a.output, a.layout)


if __name__ == '__main__':
    main()
