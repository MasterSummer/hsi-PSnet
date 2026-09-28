"""Read-only comparison of an archived PT observation and its recovered NPY cube.

Reports layout/masking hypotheses, never edits inputs or approves a merge.
Uses the base_metadata.csv and recovered_raw directory from a previous job.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

RTOL, ATOL = 1e-5, 1e-6


def stats(array):
    return dict(shape=list(array.shape), min=float(array.min()), max=float(array.max()),
                mean=float(array.mean(dtype=np.float64)), std=float(array.std(dtype=np.float64)),
                zero_fraction=float(np.mean(array == 0)),
                sha256_float32_chw=hashlib.sha256(np.ascontiguousarray(array, dtype=np.float32).tobytes()).hexdigest())


def compare(old, candidate):
    if candidate.shape != old.shape:
        return dict(same_shape=False, shape=list(candidate.shape), matched=False)
    delta = candidate.astype(np.float64) - old
    close = np.isclose(candidate, old, rtol=RTOL, atol=ATOL)
    retained = old != 0
    x, y = candidate.astype(np.float64).ravel(), old.astype(np.float64).ravel()
    x -= x.mean(); y -= y.mean()
    denom = np.sqrt(np.dot(x, x) * np.dot(y, y))
    return dict(same_shape=True, matched=bool(close.all()),
                max_abs_difference=float(np.abs(delta).max()),
                mean_abs_difference=float(np.abs(delta).mean()),
                rmse=float(np.sqrt(np.mean(delta * delta))),
                correlation=float(np.dot(x, y) / denom) if denom else None,
                matching_fraction=float(close.mean()),
                retained_nonzero_values=int(retained.sum()),
                matching_fraction_on_old_nonzero=float(close[retained].mean()) if retained.any() else None,
                max_difference_on_old_nonzero=float(np.abs(delta[retained]).max()) if retained.any() else None,
                unchanged_except_old_zeros=bool(retained.any() and close[retained].all()))


def layout_candidates(raw_hwc):
    chw = np.moveaxis(raw_hwc, -1, 0)
    yield 'MAT_HWC_to_CHW', chw
    # Reversals are diagnostics, not recommended corrections.
    for flags in itertools.product((False, True), repeat=3):
        if not any(flags):
            continue
        axes = [i for i, flip in enumerate(flags) if flip]
        yield 'reverse_CHW_axes_' + '_'.join(map(str, axes)), np.flip(chw, axis=tuple(axes))
    yield 'HWC_flat_C_reshaped_as_CHW', raw_hwc.reshape(chw.shape, order='C')
    yield 'HWC_flat_F_reshaped_as_CHW', raw_hwc.reshape(chw.shape, order='F')
    # The .hypercube text stores C,H,W-flat payload. Test accidental HWC reshape.
    yield 'hypercube_payload_reshaped_as_HWC', np.moveaxis(chw.ravel().reshape(raw_hwc.shape), -1, 0)


def analyze(old, raw_hwc):
    old = np.asarray(old, dtype=np.float32)
    raw_hwc = np.asarray(raw_hwc, dtype=np.float32)
    if old.ndim != 3 or raw_hwc.ndim != 3:
        raise ValueError('Both arrays must be three-dimensional')
    if not np.isfinite(old).all() or not np.isfinite(raw_hwc).all():
        raise ValueError('Nonfinite input values')
    raw = np.moveaxis(raw_hwc, -1, 0)
    candidates = []
    for name, candidate in layout_candidates(raw_hwc):
        candidates.append(dict(hypothesis=name, **compare(old, candidate)))
    all_zero_pixel = np.all(old == 0, axis=0)
    any_zero_pixel = np.any(old == 0, axis=0)
    return dict(old=stats(old), recovered=stats(raw),
                spatial_zero_pattern=dict(all_bands_zero_pixels=int(all_zero_pixel.sum()),
                    some_but_not_all_bands_zero_pixels=int((any_zero_pixel & ~all_zero_pixel).sum()),
                    total_pixels=int(all_zero_pixel.size)),
                tolerance=dict(rtol=RTOL, atol=ATOL), comparisons=candidates,
                exact_matching_hypotheses=[r['hypothesis'] for r in candidates if r['matched']],
                nonzero_preserving_hypotheses=[r['hypothesis'] for r in candidates
                    if r.get('unchanged_except_old_zeros')],
                merge_authorized=False,
                limitation='Layout and zero-pattern matches are clues, not proof of the original pipeline. '
                           'No fitted scaling, masking or correction is applied to new observations.')


def load_archived(row, root):
    value = str(row.hsi_path)
    if value.startswith('ptbundle:'):
        import torch
        path, index = value[len('ptbundle:'):].rsplit('::', 1)
        path = Path(path).expanduser()
        if not path.is_absolute():
            path = root / path
        print('Reading PT bundle (may take several minutes): ' + str(path), flush=True)
        try:
            bundle = torch.load(path, map_location='cpu', weights_only=False)
        except TypeError:
            bundle = torch.load(path, map_location='cpu')
        item = bundle[int(index)]
        result = item[1]
        if hasattr(result, 'detach'):
            result = result.detach().cpu().numpy()
        result = np.array(result, dtype=np.float32, copy=True)
        del item, bundle
        gc.collect()
    else:
        path = Path(value).expanduser()
        result = np.load(path if path.is_absolute() else root / path, allow_pickle=False)
    layout = str(row.get('hsi_layout', 'CHW')).upper()
    if layout == 'HWC':
        result = np.moveaxis(result, -1, 0)
    elif layout != 'CHW':
        raise ValueError('Unknown archived layout: ' + layout)
    return np.ascontiguousarray(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True, help='Previous job directory containing results/')
    parser.add_argument('--output', type=Path, required=True, help='New diagnosis directory')
    args = parser.parse_args()
    source = args.job.expanduser().resolve() / 'results'
    metadata = pd.read_csv(source / 'base_metadata.csv')
    sid = 'run1_infected_p069_l3_d2'
    matches = metadata[metadata.sample_id == sid]
    if len(matches) != 1:
        raise ValueError('Expected exactly one overlap row: ' + sid)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError('Use a new diagnosis output directory: ' + str(output))
    raw_path = source / 'recovered_raw/Plant69_Infected_Leaf3_Day2.npy'
    raw = np.load(raw_path, allow_pickle=False)
    old = load_archived(matches.iloc[0], source)
    print('Comparing layout and zero patterns; no inputs will be changed.', flush=True)
    report = analyze(old, raw)
    report.update(sample_id=sid, archived_reference=str(matches.iloc[0].hsi_path),
                  recovered_reference=str(raw_path), source_job=str(args.job.resolve()))
    output.mkdir(parents=True, exist_ok=False)
    (output / 'diagnosis.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    np.savez_compressed(output / 'overlap_pair.npz', archived_chw=old, recovered_hwc=raw)
    raw_chw = np.moveaxis(raw, -1, 0)
    if raw_chw.shape == old.shape:
        pd.DataFrame(dict(band=np.arange(old.shape[0]), old_mean=old.mean((1,2), dtype=np.float64),
                          recovered_mean=raw_chw.mean((1,2), dtype=np.float64),
                          old_zero_fraction=(old == 0).mean((1,2)))).to_csv(output / 'bands.csv', index=False)
    summary = dict(sample_id=sid, old=report['old'], recovered=report['recovered'],
                   spatial_zero_pattern=report['spatial_zero_pattern'],
                   direct_comparison=report['comparisons'][0],
                   exact_matching_hypotheses=report['exact_matching_hypotheses'],
                   nonzero_preserving_hypotheses=report['nonzero_preserving_hypotheses'],
                   limitation=report['limitation'])
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
    print('Saved report and one overlap pair to: ' + str(output), flush=True)


if __name__ == '__main__':
    main()
