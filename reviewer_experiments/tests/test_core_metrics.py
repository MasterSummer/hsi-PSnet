from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
import numpy as np
from PIL import Image

from reviewer_experiments.core import assign_plant_folds, prepare_tasks
from reviewer_experiments.metrics import classification_metrics, evaluate_predictions, plant_predictions
from reviewer_experiments.train import train_fold


class CoreMetricsTest(unittest.TestCase):
    def metadata(self) -> pd.DataFrame:
        rows = []
        for dpi in (2, 4, 6):
            for label, treatment in enumerate(("mock", "infected")):
                for index in range(6):
                    plant = f"{treatment}_{index}"
                    rows.append({"sample_id": f"{plant}_{dpi}", "plant_id": plant, "leaf_id": "L1",
                                 "treatment": treatment, "dpi": dpi, "acquisition_date": f"2024-01-0{dpi}",
                                 "imaging_session": f"S{dpi}", "rgb_path": "x.jpg", "hsi_path": "x.npy",
                                 "symptom_status": "asymptomatic", "label": label})
        return pd.DataFrame(rows)

    def test_plant_folds_never_leak(self):
        result = assign_plant_folds(self.metadata(), 3, 11)
        self.assertEqual(result.groupby("plant_id").fold.nunique().max(), 1)

    def test_prepare_all_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = self.metadata().drop(columns="label")
            frame.to_csv(root / "metadata.csv", index=False)
            prepare_tasks(root / "metadata.csv", root / "tasks", 3, 11)
            self.assertTrue((root / "tasks" / "presymptomatic_2_4.csv").exists())

    def test_empty_hsi_path_fails_file_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = self.metadata().drop(columns="label")
            frame["hsi_path"] = ""
            frame.to_csv(root / "metadata.csv", index=False)
            with self.assertRaisesRegex(ValueError, "hsi_path:<empty>"):
                prepare_tasks(root / "metadata.csv", root / "tasks", 3, 11, require_files=True)

    def test_plant_aggregation_and_metrics(self):
        predictions = pd.DataFrame([
            {"model": "psnet_full", "task": "dpi_2", "seed": 1, "plant_id": "a", "label": 0, "prob_infected": 0.1},
            {"model": "psnet_full", "task": "dpi_2", "seed": 1, "plant_id": "a", "label": 0, "prob_infected": 0.3},
            {"model": "psnet_full", "task": "dpi_2", "seed": 1, "plant_id": "b", "label": 1, "prob_infected": 0.9},
        ])
        plants = plant_predictions(predictions)
        self.assertEqual(len(plants), 2)
        self.assertEqual(classification_metrics(plants)["balanced_accuracy"], 1.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pred.csv"
            predictions.to_csv(path, index=False)
            evaluate_predictions([path], Path(directory) / "out", bootstrap=10)
            self.assertTrue((Path(directory) / "out" / "metrics_summary.csv").exists())

    def test_training_smoke_uses_outer_test_fold(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for label, treatment in enumerate(("mock", "infected")):
                for index in range(6):
                    sample = f"{treatment}_{index}"
                    Image.fromarray(np.full((32, 32, 3), 40 + label * 100, np.uint8)).save(root / f"{sample}.png")
                    np.save(root / f"{sample}.npy", np.full((16, 32, 32), label, np.float32))
                    rows.append({"sample_id": sample, "plant_id": sample, "label": label, "fold": index % 3 + 1,
                                 "rgb_path": f"{sample}.png", "hsi_path": f"{sample}.npy", "hsi_layout": "CHW"})
            task = root / "dpi_2.csv"
            pd.DataFrame(rows).to_csv(task, index=False)
            prediction_path = train_fold(task, "spectral_1d_cnn", 7, 1, root / "runs", root,
                                         epochs=1, patience=1, batch_size=4, device="cpu", dim=8)
            predictions = pd.read_csv(prediction_path)
            expected = {row["plant_id"] for row in rows if row["fold"] == 1}
            self.assertEqual(set(predictions.plant_id), expected)


if __name__ == "__main__":
    unittest.main()
