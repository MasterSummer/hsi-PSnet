import tempfile
import unittest
from pathlib import Path

import pandas as pd

from reviewer_experiments.attribution import aggregate_occlusion


class AttributionTests(unittest.TestCase):
    def test_aggregate_occlusion_across_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            columns = {
                "model": ["psnet_full", "psnet_full"],
                "task": ["dpi_2", "dpi_2"],
                "seed": [157, 257],
                "fold": [1, 1],
                "band_start": [0, 0],
                "band_end_exclusive": [10, 10],
                "balanced_accuracy_drop": [0.1, 0.3],
                "auroc_drop": [0.2, 0.4],
            }
            first = root / "first.csv"
            second = root / "second.csv"
            pd.DataFrame({name: [values[0]] for name, values in columns.items()}).to_csv(first, index=False)
            pd.DataFrame({name: [values[1]] for name, values in columns.items()}).to_csv(second, index=False)
            wavelengths = root / "wavelengths.csv"
            pd.DataFrame({"band": list(range(10)), "wavelength_nm": list(range(500, 510))}).to_csv(
                wavelengths, index=False
            )
            output = root / "output"
            aggregate_occlusion([first, second], output, wavelengths)
            result = pd.read_csv(output / "band_occlusion_summary.csv").iloc[0]
            self.assertEqual(result.runs, 2)
            self.assertAlmostEqual(result.balanced_accuracy_drop_mean, 0.2)
            self.assertAlmostEqual(result.auroc_drop_mean, 0.3)
            self.assertEqual(result.wavelength_start_nm, 500)
            self.assertEqual(result.wavelength_end_nm, 509)


if __name__ == "__main__":
    unittest.main()
