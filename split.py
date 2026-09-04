from pathlib import Path

import torch

from Data_Loader import Data_Loader


def build_pairs(split_name, split_seed=35):
    loader = Data_Loader(split=split_name, split_seed=split_seed)
    loader.load_data()
    x_data, y_data = loader.get_data()
    metadata = loader.get_sample_metadata()

    pairs = []
    for sample, label, meta in zip(x_data, y_data, metadata):
        rgb_path = Path(meta["raw_path"]).with_suffix(".png").resolve()
        if not rgb_path.exists():
            raise FileNotFoundError(f"Missing RGB file for split {split_name}: {rgb_path}")

        hsi_tensor = torch.from_numpy(sample).permute(2, 0, 1).float()
        pairs.append((str(rgb_path), hsi_tensor, int(label)))

    return pairs


def main():
    output_dir = Path(__file__).resolve().parent / "datasplits35"
    output_dir.mkdir(exist_ok=True)

    train_pairs = build_pairs("train")
    val_pairs = build_pairs("val")
    test_pairs = build_pairs("test")

    trainval_pairs = train_pairs + val_pairs

    torch.save(trainval_pairs, output_dir / "trainval.pt")
    torch.save(test_pairs, output_dir / "cleantest.pt")

    print(f"Saved {len(trainval_pairs)} pairs to {output_dir / 'trainval.pt'}")
    print(f"Saved {len(test_pairs)} pairs to {output_dir / 'cleantest.pt'}")


if __name__ == "__main__":
    main()
