import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from PIL import Image
import torch

from reviewer_experiments.bands import parse_bands, select_bands
from reviewer_experiments.data import BandStats, PairedDataset, compute_band_stats
from reviewer_experiments.models import build_model
from reviewer_experiments.train import train_fold, run_matrix
from reviewer_experiments.attribution import band_occlusion
from run_band_experiments import band_sets, experiment_sets
from reviewer_experiments.band_selection import rank_training_bands, top_band_indices


class BandSelectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_parser_and_default_groups(self):
        self.assertEqual(parse_bands('0:4,7:10:2', 10), [0, 1, 2, 3, 7, 9])
        self.assertEqual(parse_bands('all', 3), [0, 1, 2])
        for invalid in ['3,1', '1,1', '-1', '0:314', '2:2', '0:10:0', '']:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_bands(invalid, 313)
        groups = experiment_sets(None, '3,10,30', 313)
        self.assertEqual({name: value['count'] for name, value in groups.items()},
                         {'full': 313, 'top3': 3, 'top10': 10, 'top30': 30})
        self.assertIsNone(groups['top3']['indices'])
        for invalid in ['0', '314', '3,3']:
            with self.assertRaises(ValueError):
                experiment_sets(None, invalid, 313)
        with self.assertRaises(ValueError):
            band_sets(['same=0:2', 'same=2:4'], 8)

    def test_ranking_uses_equal_plant_weight_and_stable_ties(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for plant, label, spectrum, repeats in [
                ('mock1', 0, [0, 0, 0, 0], 1), ('mock2', 0, [0, 0, 0, 0], 1),
                ('infected1', 1, [20, -6, 6, 2], 1), ('infected2', 1, [0, -6, 6, 2], 5)]:
                path = root/f'{plant}.npy'
                np.save(path, np.broadcast_to(np.array(spectrum)[:, None, None], (4, 2, 2)))
                rows.extend([dict(plant_id=plant, label=label, hsi_path=str(path), hsi_layout='CHW')]*repeats)
            ranking = rank_training_bands(pd.DataFrame(rows))
            np.testing.assert_allclose(ranking.absolute_mean_difference, [10, 6, 6, 2])
            self.assertEqual(top_band_indices(ranking, 2), [0, 1])

    def test_only_selected_fit_bands_normalized_and_source_size_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cube = np.arange(6*4*4, dtype=np.float32).reshape(6, 4, 4)
            np.save(root/'fit.npy', cube)
            Image.new('RGB', (8, 8)).save(root/'rgb.png')
            fit = pd.DataFrame([dict(sample_id='fit', plant_id='fit', label=0,
                                    rgb_path=str(root/'rgb.png'), hsi_path=str(root/'fit.npy'), hsi_layout='CHW')])
            stats = compute_band_stats(fit, band_indices=[1, 4])
            np.testing.assert_allclose(stats.mean, cube[[1, 4]].mean(axis=(1, 2)))
            held = cube + 100
            held[0] = 1e9
            np.save(root/'held.npy', held)
            test = fit.assign(hsi_path=str(root/'held.npy'), sample_id='held', plant_id='held')
            restored = BandStats.from_dict(stats.as_dict())
            _, actual, _, _, _ = PairedDataset(test, restored, rgb_size=8)[0]
            expected = (held[[1, 4]] - stats.mean[:, None, None])/stats.std[:, None, None]
            np.testing.assert_allclose(actual.numpy(), expected)
            with self.assertRaisesRegex(ValueError, 'Source cube'):
                select_bands(np.zeros((7, 4, 4)), restored.band_indices, restored.source_bands)
            self.assertIsNone(BandStats.from_dict({'mean': [0], 'std': [1]}).band_indices)

    def test_training_checkpoint_inference_and_occlusion_use_same_subset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for label in (0, 1):
                for i in range(10):
                    name = f'{label}_{i}'
                    cube = np.random.default_rng(i+label*10).normal(size=(8, 32, 32)).astype(np.float32)
                    np.save(root/f'{name}.npy', cube)
                    Image.new('RGB', (32, 32), (40+80*label, 60, 90)).save(root/f'{name}.png')
                    rows.append(dict(sample_id=name, plant_id=name, label=label, fold=i%5+1,
                                     rgb_path=str(root/f'{name}.png'), hsi_path=str(root/f'{name}.npy'), hsi_layout='CHW'))
            task = root/'all_dpi.csv'
            pd.DataFrame(rows).to_csv(task, index=False)
            prediction = train_fold(task, 'psnet_full', 157, 1, root/'runs', epochs=1, patience=1,
                batch_size=4, device='cpu', dim=8, depth=3, heads=2, dropout=0, band_indices=[1, 3, 7])
            saved = torch.load(prediction.with_name('best.pt'), map_location='cpu', weights_only=False)
            self.assertEqual(saved['config']['bands'], 3)
            self.assertEqual(saved['config']['band_indices'], [1, 3, 7])
            dataset = PairedDataset(pd.read_csv(task).query('fold == 1'), BandStats.from_dict(saved['band_stats']))
            model = build_model('psnet_full', 3, 8, 3, 2, 0).eval()
            model.load_state_dict(saved['model_state'])
            rgb, hsi, _, _, _ = dataset[0]
            with torch.no_grad():
                prob = model(rgb[None], hsi[None]).softmax(1)[0, 1].item()
            self.assertAlmostEqual(prob, pd.read_csv(prediction).prob_infected.iloc[0], places=6)
            band_occlusion(prediction.with_name('best.pt'), task, root/'occlusion', group_size=2, device='cpu', batch_size=4)
            occlusion = pd.read_csv(root/'occlusion/band_occlusion.csv')
            self.assertEqual([json.loads(v) for v in occlusion.original_band_indices], [[1, 3], [7]])
            with self.assertRaisesRegex(ValueError, 'different band selection'):
                run_matrix(root, root/'runs', ['all_dpi'], ['psnet_full'], [157], [1], band_indices=[0, 2, 4])
            adaptive = train_fold(task, 'psnet_full', 157, 1, root/'adaptive', epochs=1, patience=1,
                batch_size=4, device='cpu', dim=8, depth=3, heads=2, dropout=0, top_k_bands=3)
            fit = pd.read_csv(adaptive.with_name('fit_observations.csv'))
            expected_ranking = rank_training_bands(fit)
            saved_ranking = pd.read_csv(adaptive.with_name('band_ranking.csv'))
            np.testing.assert_allclose(saved_ranking.absolute_mean_difference, expected_ranking.absolute_mean_difference)
            selected = saved_ranking.loc[saved_ranking.selected, 'band'].tolist()
            self.assertEqual(selected, top_band_indices(expected_ranking, 3))
            config = json.loads(adaptive.with_name('run.json').read_text())
            self.assertEqual(config['band_indices'], selected)
            self.assertEqual(config['top_k_bands'], 3)
            self.assertEqual(run_matrix(root, root/'adaptive', ['all_dpi'], ['psnet_full'], [157], [1], top_k_bands=3), [adaptive])


if __name__ == '__main__':
    unittest.main()
