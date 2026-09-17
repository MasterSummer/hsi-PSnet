import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd
from PIL import Image
import torch


class BandRunnerTest(unittest.TestCase):
    def test_direct_pt_preparation_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rgb = root/'rgb'
            rgb.mkdir()
            records = []
            for dpi in (2, 4, 6):
                for treatment in ('Healthy', 'Infected'):
                    for plant in range(1, 11):
                        number = plant + (24*(dpi//2-1) if treatment == 'Healthy' else 0)
                        name = f'Plant{number}_{treatment}_Leaf3_Day{dpi}.png'
                        Image.new('RGB', (32, 32)).save(rgb/name)
                        records.append((f'/old/server/{name}', torch.ones(32, 8, 8),
                                        0 if treatment == 'Healthy' else dpi//2))
            torch.save(records[::2], root/'trainval.pt')
            torch.save(records[1::2], root/'cleantest.pt')
            out = root/'out'
            command = [sys.executable, 'run_band_experiments.py', '--split-dir', str(root),
                       '--rgb-root', str(rgb), '--output', str(out), '--prepare-only', '--device', 'cpu']
            for _ in range(2):
                run = subprocess.run(command, text=True, capture_output=True, timeout=180)
                self.assertEqual(run.returncode, 0, run.stdout+run.stderr)
                self.assertIn('20 fold trainings', run.stdout)
            manifest = json.loads((out/'band_experiment_manifest.json').read_text())
            frame = pd.read_csv(out/'tasks/all_dpi.csv')
            self.assertTrue(frame.groupby('plant_id').fold.nunique().eq(1).all())
            for name, k in [('full', None), ('top3', 3), ('top10', 10), ('top30', 30)]:
                child = json.loads((out/name/'experiment_manifest.json').read_text())
                self.assertEqual(child['task_sha256'], manifest['task_sha256'])
                self.assertEqual(child['identity']['top_k_bands'], k)
                counts = pd.read_csv(out/name/'parameter_counts.csv')
                self.assertEqual(counts.bands.iloc[0], k or 32)


if __name__ == '__main__':
    unittest.main()
