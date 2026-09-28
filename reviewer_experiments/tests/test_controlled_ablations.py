import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from PIL import Image
import torch

from reviewer_experiments.models import build_model, trainable_parameters
from reviewer_experiments.train import train_fold


class ControlledAblationsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_matched_counts_and_projection_shapes(self):
        names = ['psnet_full', 'psnet_raw_spectrum_token', 'psnet_caf_self', 'psnet_raw_token_caf_self']
        counts = []
        for name in names:
            model = build_model(name, 16, dim=8, depth=3, heads=2, dropout=0)
            counts.append(trainable_parameters(model))
            self.assertEqual(tuple(model.hsi_encoder.global_token.project[0].weight.shape), (8, 16))
        self.assertEqual(len(set(counts)), 1)

    def test_variants_forward_backward_and_roundtrip(self):
        rgb, hsi = torch.randn(2, 3, 32, 32), torch.randn(2, 16, 32, 32)
        for name in ['psnet_raw_spectrum_token', 'psnet_caf_self', 'psnet_raw_token_caf_self', 'psnet_mean_depth_patch']:
            with self.subTest(name=name):
                model = build_model(name, 16, dim=8, depth=3, heads=2, dropout=0).eval()
                output = model(rgb, hsi)
                self.assertEqual(tuple(output.shape), (2, 2))
                output.square().sum().backward()
                self.assertGreater(model.hsi_encoder.patch.weight.grad.abs().sum().item(), 0)
                self.assertGreater(model.hsi_encoder.transformer.fusion[0].weight.grad.abs().sum().item(), 0)
                copy = build_model(name, 16, dim=8, depth=3, heads=2, dropout=0).eval()
                copy.load_state_dict(model.state_dict(), strict=True)
                with torch.no_grad():
                    torch.testing.assert_close(copy(rgb, hsi), output)

    def test_depth_mean_retains_tokens_and_has_expected_restricted_kernel(self):
        full = build_model('psnet_full', 16, dim=8, depth=3, heads=2).hsi_encoder.eval()
        mean = build_model('psnet_mean_depth_patch', 16, dim=8, depth=3, heads=2).hsi_encoder.eval()
        # The replacement is the constrained subset where all 8 depth weights are tied.
        full.spectral_reduce.load_state_dict(mean.spectral_reduce.state_dict())
        with torch.no_grad():
            full.patch.weight.copy_(mean.patch.weight.expand(-1, -1, 8, -1, -1)/8)
            full.patch.bias.copy_(mean.patch.bias)
            hsi = torch.randn(1, 16, 132, 135)
            a, b = full.patch_tokens(hsi), mean.patch_tokens(hsi)
        self.assertEqual(tuple(b.shape), (1, 512, 8))
        torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)

    def test_one_epoch_training_and_checkpoint_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for label in (0, 1):
                for i in range(10):
                    name = f'{label}_{i}'
                    np.save(root/f'{name}.npy', np.random.default_rng(i+label*10).normal(size=(16, 32, 32)).astype(np.float32))
                    Image.new('RGB', (32, 32), (30+label*70, 60, 90)).save(root/f'{name}.png')
                    rows.append(dict(sample_id=name, plant_id=name, label=label, fold=i%5+1,
                                     rgb_path=str(root/f'{name}.png'), hsi_path=str(root/f'{name}.npy'), hsi_layout='CHW'))
            pd.DataFrame(rows).to_csv(root/'dpi_2.csv', index=False)
            path = train_fold(root/'dpi_2.csv', 'psnet_raw_token_caf_self', 157, 1, root/'runs',
                              epochs=1, patience=1, batch_size=4, device='cpu', dim=8, depth=3, heads=2,
                              dropout=0, rgb_pretrained=False)
            result = pd.read_csv(path)
            self.assertEqual(len(result), 4)
            self.assertTrue(result.prob_infected.between(0, 1).all())
            self.assertTrue(path.with_name('best.pt').exists())


if __name__ == '__main__':
    unittest.main()
