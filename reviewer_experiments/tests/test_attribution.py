from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from reviewer_experiments.attribution import aggregate_occlusion


class AttributionTest(unittest.TestCase):
    def test_occlusion_runs_are_aggregated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for fold, drop in ((1, 0.1), (2, 0.3)):
                rows.append({
                    "model": "psnet_full", "task": "dpi_2", "seed": 157,
                    "fold": fold, "band_start": 0, "band_end_exclusive": 10,
                    "balanced_accuracy_drop": drop, "auroc_drop": drop / 2,
                })
            input_path = root / "band_occlusion.csv"
            pd.DataFrame(rows).to_csv(input_path, index=False)
            aggregate_occlusion([input_path], root / "output")
            result = pd.read_csv(root / "output" / "band_occlusion_summary.csv")
            self.assertEqual(result.loc[0, "runs"], 2)
            self.assertAlmostEqual(result.loc[0, "balanced_accuracy_drop_mean"], 0.2)

    def test_different_source_subsets_not_merged_and_sparse_wavelengths_mapped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [dict(model='psnet_full', task='all_dpi', seed=157, fold=fold,
                         band_start=0, band_end_exclusive=2, original_band_indices=bands,
                         balanced_accuracy_drop=.1, auroc_drop=.2)
                    for fold, bands in [(1, '[1, 7]'), (2, '[2, 8]')]]
            pd.DataFrame(rows).to_csv(root/'input.csv', index=False)
            pd.DataFrame(dict(band=[1, 7], wavelength_nm=[450, 700])).to_csv(root/'wave.csv', index=False)
            aggregate_occlusion([root/'input.csv'], root/'out', root/'wave.csv')
            result = pd.read_csv(root/'out/band_occlusion_summary.csv')
            self.assertEqual(len(result), 2)
            self.assertTrue(result.runs.eq(1).all())
            self.assertEqual(result.wavelengths_nm.tolist(), ['[450, 700]', '[null, null]'])


if __name__ == "__main__":
    unittest.main()
