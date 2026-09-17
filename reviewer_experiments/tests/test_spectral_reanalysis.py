import argparse
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy import stats

from reviewer_experiments.spectral_reanalysis import aggregate_samples, bh_fdr, band_statistics, extract_spectra, load_plants, metadata_from_pt, pt_bundle


class SpectralTests(unittest.TestCase):
    def test_welch_matches_scipy_and_effect_direction(self):
        a, b = np.array([1., 2., 4., 8.]), np.array([4., 5., 7., 9., 12.])
        got = band_statistics(a, b)
        self.assertAlmostEqual(got['welch_p'], stats.ttest_ind(a, b, equal_var=False).pvalue)
        self.assertGreater(got['hedges_g_infected_minus_mock'], 0)
        self.assertLess(got['difference_ci_low'], got['mean_difference'])
        self.assertGreater(got['difference_ci_high'], got['mean_difference'])

    def test_fdr_reference_and_nan_does_not_contaminate_family(self):
        p = np.array([.04, .001, .2, .03, .008])
        np.testing.assert_allclose(bh_fdr(p), stats.false_discovery_control(p))
        np.testing.assert_allclose(bh_fdr([.01, np.nan, .04]), [.03, np.nan, .06], equal_nan=True)

    def test_degenerate_band_is_flagged(self):
        r = band_statistics(np.ones(3), np.ones(5)*2)
        self.assertTrue(np.isnan(r['welch_p']))
        self.assertTrue(np.isnan(r['hedges_g_infected_minus_mock']))
        self.assertEqual(r['test_status'], 'undefined_zero_variance')

    def test_repeated_capture_does_not_overweight_leaf(self):
        samples = pd.DataFrame(dict(plant_id=['p']*3, treatment=['mock']*3, dpi=[2]*3,
                                    leaf_id=['3', '3', '4'], **{'0': [1., 3., 10.]}))
        plant = aggregate_samples(samples, ['0'])
        self.assertEqual(plant['0'].iloc[0], 6.)
        self.assertEqual(plant.n_leaves.iloc[0], 2)

    def test_roi_layout_and_empty_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cube = np.array([[[2., 10.], [6., 20.]], [[4., 30.], [8., 40.]]])
            np.save(root/'cube.npy', np.moveaxis(cube, 0, -1))
            np.save(root/'mask.npy', [[1, 0], [1, 0]])
            pd.DataFrame([dict(sample_id='s', plant_id='p', leaf_id='3', treatment='mock', dpi=2,
                               hsi_path='cube.npy', hsi_layout='HWC', mask_path='mask.npy')]).to_csv(root/'meta.csv', index=False)
            args = argparse.Namespace(metadata=root/'meta.csv', data_root=None, layout=None,
                                      region='leaf', expected_bands=2)
            plants, _, _, _ = extract_spectra(args)
            np.testing.assert_allclose(plants[['0', '1']].iloc[0], [4., 6.])
            args.region = 'background'
            plants, _, _, _ = extract_spectra(args)
            np.testing.assert_allclose(plants[['0', '1']].iloc[0], [15., 35.])
            np.save(root/'mask.npy', np.ones((2, 2)))
            with self.assertRaisesRegex(ValueError, 'Empty selected region'):
                extract_spectra(args)

    def test_duplicate_plant_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'plants.csv'
            pd.DataFrame(dict(plant_id=['p', 'p'], treatment=['mock', 'mock'], dpi=[2, 2],
                              **{'0': [1, 2]})).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'exactly one row'):
                load_plants(path)

    def test_direct_pt_end_to_end_without_rgb_or_csv(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = []
            for dpi in (2, 4, 6):
                for treatment in ('Healthy', 'Infected'):
                    for plant in range(1, 4):
                        for leaf in (3, 4):
                            # Mock source numbers differ across dates, as in Run 1.
                            source = plant + 24 * ((dpi // 2)-1) if treatment == 'Healthy' else plant
                            filename = f'Plant{source}_{treatment}_Leaf{leaf}_Day{dpi}.RGB888'
                            cube = torch.full((2, 3, 4), float(plant+leaf+(treatment=='Infected')))
                            label = 0 if treatment == 'Healthy' else dpi//2
                            records.append((f'Z:\\missing_rgb\\{filename}', cube, label))
            torch.save(records[::2], root/'trainval.pt')
            torch.save(records[1::2], root/'cleantest.pt')
            command = [sys.executable, '-m', 'reviewer_experiments.spectral_reanalysis',
                       '--split-dir', str(root), '--expected-bands', '2', '--bands', '1', '--output', str(root/'out')]
            run = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            result = pd.read_csv(root/'out/summary_by_dpi.csv')
            self.assertEqual(result.n_mock.tolist(), [3, 3, 3])
            self.assertEqual(result.n_infected.tolist(), [3, 3, 3])
            plants = pd.read_csv(root/'out/plant_mean_spectra.csv')
            self.assertEqual(len(plants), 18)
            self.assertTrue(plants.n_leaves.eq(2).all())
            self.assertNotIn('0', plants.columns)
            self.assertIn('1', plants.columns)
            statistics = pd.read_csv(root/'out/within_dpi_band_statistics.csv')
            self.assertEqual(statistics.band.tolist(), [1, 1, 1])

    def test_pt_duplicate_and_label_checks(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            item = ('Plant1_Infected_Leaf3_Day2.png', torch.ones(2, 3, 4), 1)
            torch.save([item], root/'a.pt')
            torch.save([item], root/'b.pt')
            with self.assertRaisesRegex(ValueError, 'Duplicate biological'):
                metadata_from_pt([root/'a.pt', root/'b.pt'])
            torch.save([(item[0], item[1], 0)], root/'bad.pt')
            with self.assertRaisesRegex(ValueError, 'disagrees'):
                metadata_from_pt([root/'bad.pt'])
        pt_bundle.cache_clear()

    def test_pt_rejects_filenames_without_metadata(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'input.pt'
            torch.save([('Healthy_001.png', torch.ones(2, 3, 4), 0)], path)
            with self.assertRaisesRegex(ValueError, 'Cannot recover'):
                metadata_from_pt([path])
        pt_bundle.cache_clear()


if __name__ == '__main__':
    unittest.main()
