from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from reviewer_experiments.models import MODEL_NAMES, build_model
from reviewer_experiments.spectral import bh_fdr, spectral_analysis


class ModelsSpectralTest(unittest.TestCase):
    def test_all_models_forward(self):
        rgb = torch.randn(2, 3, 32, 32)
        hsi = torch.randn(2, 16, 32, 32)
        for name in MODEL_NAMES:
            with self.subTest(name=name):
                model = build_model(name, bands=16, dim=16, depth=2, heads=2)
                model.eval()
                with torch.no_grad():
                    self.assertEqual(tuple(model(rgb, hsi).shape), (2, 2))

    def test_bh_is_monotonic_by_pvalue(self):
        p = np.array([0.03, 0.001, 0.02])
        q = bh_fdr(p)
        order = np.argsort(p)
        self.assertTrue(np.all(np.diff(q[order]) >= -1e-12))

    def test_spectral_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            wavelengths = pd.DataFrame({"band": range(8), "wavelength_nm": np.linspace(670, 760, 8)})
            wavelengths.to_csv(root / "wavelengths.csv", index=False)
            for dpi in (2, 4, 6):
                for treatment, offset in (("mock", 0.0), ("infected", 0.2)):
                    for index in range(3):
                        sample = f"{dpi}_{treatment}_{index}"
                        np.save(root / f"{sample}.npy", np.ones((8, 4, 4), np.float32) * (index + offset))
                        rows.append({"sample_id": sample, "plant_id": sample, "leaf_id": "L1", "treatment": treatment,
                                     "dpi": dpi, "acquisition_date": f"2024-01-0{dpi}", "imaging_session": f"S{dpi}",
                                     "rgb_path": "unused.jpg", "hsi_path": f"{sample}.npy", "hsi_layout": "CHW",
                                     "symptom_status": "asymptomatic"})
            pd.DataFrame(rows).to_csv(root / "metadata.csv", index=False)
            spectral_analysis(root / "metadata.csv", root / "out", root, root / "wavelengths.csv")
            self.assertTrue((root / "out" / "within_dpi_band_statistics.csv").exists())
            self.assertTrue((root / "out" / "red_edge_statistics.csv").exists())


if __name__ == "__main__":
    unittest.main()
