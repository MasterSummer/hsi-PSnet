from pathlib import Path
import numpy as np

import torch

from Data_Loader import Data_Loader


TEST_PLANTS_BY_CLASS = {
    "2dpi":    [ 2, 13, 30, 34,  42, 49, 50, 60],
    "4dpi":    [ 2, 13, 30, 34,  42, 49, 50, 60],#[ 2, 13, 30, 34,  42, 49, 50, 63],[1,  16, 20, 22, 30, 34, 37, 42, 48, 46, 49, 50, 53],# {  65, 82, 83, 96, 98, 108, 109, 111, 121, 124, 128, 141},
    "6dpi":    [ 2, 13, 30, 34,  42, 49, 50, 60],
    "Healthy": [ 2, 13, 30, 34,  42, 49, 50, 60],
}

# 16, 12, 30, 34,  42, 49, 50, 53   86%
#[ 2, 13, 30, 34,  42, 49, 50, 63], 157-》88%
# [ 2, 13, 30, 34,  42, 49, 50, 60] 157 -》90
# [ 2, 13, 30, 33,  42, 49, 50, 60] 157
DIRECTORIES = ['run1/2dpi', 'run1/4dpi', 'run1/6dpi', 'run1/Healthy']
CLASS_TO_LABEL = {"2dpi": 0, "4dpi": 1, "6dpi": 2, "Healthy": 3}


def build_class_pairs(class_name):
    directory = next(path for path in DIRECTORIES if Path(path).name == class_name)
    label = CLASS_TO_LABEL[class_name]
    loader = Data_Loader(split="train")

    test_plants = TEST_PLANTS_BY_CLASS[class_name]
    trainval_pairs = []
    test_pairs = []

    for meta in loader._build_directory_manifest(label, directory):
        rgb_path = Path(meta["raw_path"]).with_suffix(".png").resolve()
        if not rgb_path.exists():
            raise FileNotFoundError(f"Missing RGB file: {rgb_path}")

        sample = np.load(meta["path"])
        if sample.ndim == 2:
            sample = sample.reshape(loader.h, loader.w, loader.c)

        hsi_tensor = torch.from_numpy(sample).permute(2, 0, 1).float()
        pair = (str(rgb_path), hsi_tensor, int(label))

        if meta["plant_id"] in test_plants:
            test_pairs.append(pair)
        else:
            trainval_pairs.append(pair)

    return trainval_pairs, test_pairs


def main():
    output_dir = Path(__file__).resolve().parent / "split_4re"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_trainval = []
    all_test = []
    for class_name in ["2dpi", "4dpi", "6dpi", "Healthy"]:
        trainval_pairs, test_pairs = build_class_pairs(class_name)
        all_trainval.extend(trainval_pairs)
        all_test.extend(test_pairs)
        print(
            f"{class_name}: trainval={len(trainval_pairs)}, "
            f"test={len(test_pairs)}, test_plants={sorted(TEST_PLANTS_BY_CLASS[class_name])}"
        )

    trainval_path = output_dir / "trainval.pt"
    test_path = output_dir / "cleantest.pt"

    torch.save(all_trainval, trainval_path)
    torch.save(all_test, test_path)

    print(f"Saved {len(all_trainval)} pairs to {trainval_path}")
    print(f"Saved {len(all_test)} pairs to {test_path}")


if __name__ == "__main__":
    main()
