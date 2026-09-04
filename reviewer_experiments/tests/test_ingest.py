from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from reviewer_experiments.data import load_hsi
from reviewer_experiments.ingest import build_run1_metadata, build_split4re_metadata


class IngestTest(unittest.TestCase):
    def test_run1_name_parsing_and_hsi_matching(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rgb = root / "rgb" / "2dpi"
            hsi = root / "hsi"
            rgb.mkdir(parents=True)
            hsi.mkdir()
            name = "Plant7_Infected_Leaf3_Day2"
            (rgb / f"{name}.RGB888").write_bytes(b"jpeg-placeholder")
            np.save(hsi / f"{name}.npy", np.zeros((2, 2, 3), dtype=np.float32))
            summary = build_run1_metadata(root / "rgb", root / "metadata.csv", hsi)
            frame = pd.read_csv(root / "metadata.csv")
            self.assertEqual(frame.loc[0, "plant_id"], "run1_infected_p007")
            self.assertEqual(frame.loc[0, "dpi"], 2)
            self.assertTrue(summary["multimodal_ready"])

    def test_healthy_source_numbers_map_to_repeated_biological_plant(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for day, source in ((2, 1), (4, 25), (6, 49)):
                folder = root / "rgb" / "Healthy" / f"Day {day}"
                folder.mkdir(parents=True)
                (folder / f"Plant{source}_Healthy_Leaf3_Day{day}.RGB888").write_bytes(b"jpeg-placeholder")
            summary = build_run1_metadata(root / "rgb", root / "metadata.csv")
            frame = pd.read_csv(root / "metadata.csv")
            self.assertEqual(frame.plant_id.nunique(), 1)
            self.assertEqual(set(frame.source_plant_number), {1, 25, 49})
            self.assertEqual(summary["plants"], 1)

    def test_jpg_extension_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rgb = root / "rgb" / "2dpi"
            rgb.mkdir(parents=True)
            (rgb / "Plant1_Infected_Leaf3_Day2.jpg").write_bytes(b"jpeg-placeholder")
            summary = build_run1_metadata(root / "rgb", root / "metadata.csv")
            self.assertEqual(summary["samples"], 1)

    def test_split4re_pt_bundles_are_used_directly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split = root / "split_4re"
            rgb = root / "rgb"
            split.mkdir()
            rgb.mkdir()
            infected_name = "Plant1_Infected_Leaf3_Day2"
            mock_name = "Plant25_Healthy_Leaf3_Day4"
            infected_rgb = rgb / f"{infected_name}.jpg"
            mock_rgb = rgb / f"{mock_name}.jpg"
            infected_rgb.write_bytes(b"jpeg-placeholder")
            mock_rgb.write_bytes(b"jpeg-placeholder")
            infected_hsi = torch.ones(3, 2, 2)
            mock_hsi = torch.zeros(3, 2, 2)
            torch.save([(str(infected_rgb), infected_hsi, 1)], split / "trainval.pt")
            torch.save([(str(mock_rgb), mock_hsi, 0)], split / "cleantest.pt")
            summary = build_split4re_metadata(split, split / "metadata.csv")
            frame = pd.read_csv(split / "metadata.csv")
            self.assertEqual(summary["samples"], 2)
            self.assertEqual(frame.loc[frame.treatment == "mock", "plant_id"].iloc[0], "run1_mock_p001")
            loaded = load_hsi(frame.loc[frame.treatment == "infected", "hsi_path"].iloc[0], "CHW")
            np.testing.assert_array_equal(loaded, infected_hsi.numpy())


if __name__ == "__main__":
    unittest.main()
