"""One-shot server job: preflight, guarded cohort selection, training and summary.

Run in a dedicated code snapshot. Use a NEW output directory for each job.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def execute(script, *args):
    command = [sys.executable, '-u', str(ROOT / script), *map(str, args)]
    print('RUN:', command, flush=True)
    subprocess.run(command, check=True, cwd=ROOT)


def choose_cohort(metadata, recovered, rgb_root, output, policy, layout='HWC'):
    from recover_revision_data import merge
    from reviewer_experiments.core import prepare_tasks
    if policy != 'archived' and recovered is not None:
        try:
            if rgb_root is None:
                raise ValueError('Adding recovered observations requires --rgb-root')
            merge(metadata, recovered, rgb_root, output / 'recovered_cohort', layout)
        except (ValueError, OSError, KeyError) as error:
            if policy == 'require-recovered':
                raise
            reason = f'{type(error).__name__}: {error}'
            print('RECOVERY NOT USED; continuing with archived cohort:', reason, flush=True)
        else:
            return output / 'recovered_cohort/tasks', dict(cohort='recovered', recovered_included=True,
                reason='Full-cube overlap check and merge passed. Inspect retained processing provenance.')
    else:
        if policy == 'require-recovered':
            raise ValueError('Recovery source is required')
        reason = 'Archived cohort selected' if policy == 'archived' else 'No recovery source supplied'
    tasks = output / 'archived_tasks'
    prepare_tasks(metadata, tasks, n_splits=5, seed=157, require_files=True)
    return tasks, dict(cohort='archived', recovered_included=False, reason=reason)


def preflight(metadata, device):
    import numpy as np
    import torch
    from PIL import Image
    from reviewer_experiments.core import read_metadata
    from reviewer_experiments.data import load_hsi, _load_pt_bundle
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in this Python environment; GPU training was not started')
    # Also validates the requested GPU index and that a CUDA allocation succeeds.
    torch.empty(1, device=device)
    frame = read_metadata(metadata)
    for row in frame.itertuples():
        cube = load_hsi(row.hsi_path, row.hsi_layout)
        if cube.shape != (313, 132, 135) or not np.isfinite(cube).all():
            raise ValueError(f'Invalid historical HSI tensor: {row.sample_id}, shape={cube.shape}')
        with Image.open(row.rgb_path) as im:
            im.convert('RGB').load()
    del cube
    _load_pt_bundle.cache_clear()
    gc.collect()
    # Fetch or check the only pretrained backbone used by the controls preset now,
    # rather than discovering a missing cache/network connection after preprocessing.
    from torchvision.models import resnet34, ResNet34_Weights
    model = resnet34(weights=ResNet34_Weights.DEFAULT)
    del model
    gc.collect()
    return dict(observations=len(frame), plants=int(frame.plant_id.nunique()),
                gpu=torch.cuda.get_device_name(torch.device(device)))


def job(args, output, stage):
    from reviewer_experiments.ingest import build_split4re_metadata
    from reviewer_experiments.data import _load_pt_bundle
    from recover_revision_data import export_mat
    stage('metadata')
    metadata = output / 'base_metadata.csv'
    info = build_split4re_metadata(args.split_dir, metadata, args.rgb_root)
    if info['missing_rgb']:
        raise ValueError('Missing historical RGB files; supply the correct --rgb-root')
    stage('preflight')
    validation = preflight(metadata, args.device)
    write_json(output / 'preflight.json', validation)
    stage('recovery_audit')
    recovered = args.processed_dir
    export_error = None
    if args.zip:
        try:
            export_mat(args.zip, output / 'recovered_raw')
            recovered = output / 'recovered_raw'
        except (ValueError, OSError, KeyError) as error:
            if args.data_policy == 'require-recovered':
                raise
            export_error = f'{type(error).__name__}: {error}'
            print('RECOVERY EXPORT FAILED:', export_error, flush=True)
    tasks, decision = choose_cohort(metadata, recovered, args.rgb_root, output,
                                   args.data_policy, 'HWC' if args.zip else args.layout)
    if export_error:
        decision['reason'] = 'Recovery export failed: ' + export_error
    write_json(output / 'cohort_decision.json', decision)
    print(json.dumps(decision, ensure_ascii=False), flush=True)
    # Avoid retaining both PT bundles in this parent during each trainer subprocess.
    _load_pt_bundle.cache_clear()
    gc.collect()
    stage('input_hashes', **decision)
    import pandas as pd
    from reviewer_experiments.data import parse_pt_bundle_uri
    frame = pd.read_csv(tasks / 'all_dpi.csv')
    files = set()
    for column in ('hsi_path', 'rgb_path'):
        for value in frame[column]:
            files.add(parse_pt_bundle_uri(value)[0] if value.startswith('ptbundle:') else Path(value))
    if args.zip:
        files.add(args.zip)
    hashes = {str(p): digest(p) for p in sorted(files)}
    write_json(output / 'input_sha256.json', hashes)
    stage('training', **decision)
    execute('run_revision.py', '--preset', 'controls', '--task-dir', tasks,
            '--output', output / 'experiments', '--device', args.device, '--execute')
    stage('summary', **decision)
    execute('summarize_revision.py', '--runs-root', output / 'experiments',
            '--output', output / 'summary')
    # Recheck once so accidental input edits during a long job do not pass unnoticed.
    stage('verify_inputs', **decision)
    if any(digest(Path(p)) != value for p, value in hashes.items()):
        raise RuntimeError('Input content changed during this job; results must not be used')
    message = ('补发数据通过重叠检查并已纳入。' if decision['recovered_included'] else
               '补发数据未纳入，本次使用旧 PT 队列。原因：' + decision['reason'])
    (output / 'RESULTS.md').write_text('# 运行完成\n\n' + message + '\n\n'
        '完成 controls：5 个模型 × 5 折 × 1 个种子，共 25 次训练。\n\n'
        '指标：summary/summary.csv；逐种子指标：summary/metrics_by_seed.csv。\n'
        '单种子没有种子间 SD；该实验混合全部日期，不能作为症状前诊断证据。\n'
        '核查文件：cohort_decision.json、input_sha256.json、preflight.json 及各折输出。\n')
    return decision


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split-dir', type=Path, required=True)
    p.add_argument('--rgb-root', type=Path)
    source = p.add_mutually_exclusive_group()
    source.add_argument('--zip', type=Path)
    source.add_argument('--processed-dir', type=Path)
    p.add_argument('--layout', choices=['HWC', 'CHW'], default='HWC')
    p.add_argument('--data-policy', choices=['archived', 'prefer-recovered', 'require-recovered'], default='prefer-recovered')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    for key in ('split_dir','rgb_root','zip','processed_dir','output'):
        if getattr(args, key) is not None:
            setattr(args, key, getattr(args, key).expanduser().resolve())
    # Refuse reusing a job directory; training-level resume remains separately available.
    args.output.mkdir(parents=True, exist_ok=False)
    def stage(name, **extra):
        write_json(args.output / 'STATUS.json', dict(state='running', stage=name, time=time.time(), **extra))
        print('STAGE:', name, flush=True)
    write_json(args.output / 'job.json', {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    try:
        if not args.device.startswith('cuda'):
            raise ValueError('This unattended preset requires a CUDA device')
        for path in (args.zip, args.processed_dir):
            if path is not None and not path.exists():
                raise FileNotFoundError(path)
        decision = job(args, args.output, stage)
    except BaseException as error:
        write_json(args.output / 'STATUS.json', dict(state='failed', error=f'{type(error).__name__}: {error}', time=time.time()))
        (args.output / 'FAILED.txt').write_text(traceback.format_exc())
        raise
    write_json(args.output / 'STATUS.json', dict(state='completed', time=time.time(), **decision))
    (args.output / 'DONE.txt').write_text('Completed; read RESULTS.md and cohort_decision.json.\n')


if __name__ == '__main__':
    main()
