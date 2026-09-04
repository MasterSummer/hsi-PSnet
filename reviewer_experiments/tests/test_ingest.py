from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from reviewer_experiments.ingest import build_run1_metadata


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


if __name__ == "__main__":
    unittest.main()
