"""Explicit experiment presets; prints commands unless --execute is supplied."""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys

PRESETS = {
    'main': ('dpi_2,dpi_4,dpi_6,presymptomatic_2_4',
             'psnet_full,plain_multimodal,simple_multimodal,rgb_resnet18,rgb_resnet34,hsi_transformer,spectral_1d_cnn,compact_3d_cnn', [157,257,357]),
    'priority': ('dpi_2,dpi_4,presymptomatic_2_4', 'psnet_full,rgb_resnet18,plain_multimodal', [157]),
    'recovered': ('dpi_2,presymptomatic_2_4,all_dpi', 'psnet_full,rgb_resnet18,plain_multimodal', [157,257,357]),
    'controls': ('all_dpi', 'psnet_full,psnet_raw_spectrum_token,psnet_caf_self,psnet_raw_token_caf_self,psnet_mean_depth_patch', [157]),
    'full': ('dpi_2,dpi_4,dpi_6,presymptomatic_2_4',
             'psnet_full,psnet_no_caf,psnet_no_wavelet,psnet_no_3d_patch,psnet_learnable_cls,plain_multimodal,simple_multimodal,rgb_resnet18,rgb_resnet34,hsi_transformer,spectral_1d_cnn,compact_3d_cnn', [157,257,357]),
}


def commands(args):
    tasks, models, seeds = PRESETS[args.preset]
    if args.seeds:
        seeds = [int(v) for v in args.seeds.split(',')]
        if len(set(seeds)) != len(seeds) or any(s < 0 or s >= 2**32-5 for s in seeds):
            raise ValueError('Seeds must be unique and in range')
    source = ['--task-dir', str(args.task_dir)] if args.task_dir else ['--split-dir', str(args.split_dir)]
    if args.rgb_root:
        source += ['--rgb-root', str(args.rgb_root)]
    for seed in seeds:
        yield [sys.executable, str(Path(__file__).with_name('run_controlled_ablations.py')),
            *source, '--output', str(args.output / f'seed_{seed}'), '--tasks', tasks,
            '--models', models, '--seed', str(seed), '--device', args.device,
            '--epochs', str(args.epochs), '--patience', str(args.patience),
            '--batch-size', str(args.batch_size), '--workers', str(args.workers)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preset', choices=PRESETS, default='priority')
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--task-dir', type=Path)
    source.add_argument('--split-dir', type=Path)
    p.add_argument('--rgb-root', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seeds', help='Override preset training seeds; does not change outer folds')
    p.add_argument('--device', default='cuda')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--patience', type=int, default=12)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    for command in commands(a):
        print(shlex.join(command), flush=True)
        if a.execute:
            subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
