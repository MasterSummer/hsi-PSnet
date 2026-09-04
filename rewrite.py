import argparse
import re
from pathlib import Path

import torch


NAME_PATTERN = re.compile(
    r"Plant(?P<plant>\d+)_(?P<status>[A-Za-z]+)_Leaf(?P<leaf>\d+)_Day(?P<day>\d+)$"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rewrite one class subset between trainval.pt and test.pt using Plant-style file names."
    )
    parser.add_argument("--trainval", required=True, help="Path to trainval.pt")
    parser.add_argument("--test", required=True, help="Path to test.pt / cleantest.pt")
    parser.add_argument("--class-name", required=True, choices=["2dpi", "4dpi", "6dpi", "Healthy"])
    parser.add_argument(
        "--test-plants",
        required=True,
        help="Comma-separated plant ids to keep in test for the target class.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for rewritten trainval.pt and test.pt",
    )
    return parser.parse_args()


def parse_target(text):
    return sorted({int(item.strip()) for item in text.split(",") if item.strip()})


def parse_sample(pair):
    rgb_path = Path(pair[0])
    stem = rgb_path.stem
    match = NAME_PATTERN.match(stem)
    if not match:
        raise ValueError(f"Unsupported filename format: {rgb_path}")

    class_name = rgb_path.parent.name
    if class_name == "Day 2" or class_name == "Day 4" or class_name == "Day 6":
        class_name = "Healthy"

    return {
        "pair": pair,
        "class_name": class_name,
        "plant_id": int(match.group("plant")),
        "leaf_id": int(match.group("leaf")),
        "day": int(match.group("day")),
        "status": match.group("status"),
        "rgb_path": str(rgb_path),
    }


def split_pairs(pairs, target_class):
    target = []
    other = []
    for pair in pairs:
        sample = parse_sample(pair)
        if sample["class_name"] == target_class:
            target.append(sample)
        else:
            other.append(pair)
    return target, other


def sort_key(sample):
    return (sample["plant_id"], sample["leaf_id"], sample["day"], Path(sample["rgb_path"]).name)


def main():
    args = parse_args()
    desired_test_plants = parse_target(args.test_plants)

    trainval_pairs = torch.load(args.trainval)
    test_pairs = torch.load(args.test)

    trainval_target, trainval_other = split_pairs(trainval_pairs, args.class_name)
    test_target, test_other = split_pairs(test_pairs, args.class_name)

    combined_target = {}
    for sample in trainval_target + test_target:
        key = (sample["plant_id"], sample["leaf_id"], sample["day"], sample["rgb_path"])
        if key in combined_target:
            raise ValueError(f"Duplicate sample detected: {sample['rgb_path']}")
        combined_target[key] = sample

    available_test_plants = sorted({sample["plant_id"] for sample in combined_target.values()})
    missing = [plant for plant in desired_test_plants if plant not in available_test_plants]
    if missing:
        raise ValueError(f"Requested plants not found in class {args.class_name}: {missing}")

    new_test_target = [
        sample["pair"]
        for sample in sorted(combined_target.values(), key=sort_key)
        if sample["plant_id"] in desired_test_plants
    ]
    new_trainval_target = [
        sample["pair"]
        for sample in sorted(combined_target.values(), key=sort_key)
        if sample["plant_id"] not in desired_test_plants
    ]

    new_trainval = trainval_other + new_trainval_target
    new_test = test_other + new_test_target

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    trainval_out = output_dir / "trainval.pt"
    test_out = output_dir / Path(args.test).name

    torch.save(new_trainval, trainval_out)
    torch.save(new_test, test_out)

    print(f"Saved rewritten trainval to {trainval_out}")
    print(f"Saved rewritten test to {test_out}")
    print(f"{args.class_name} plants in test: {desired_test_plants}")
    print(f"trainval size: {len(new_trainval)}")
    print(f"test size: {len(new_test)}")


if __name__ == "__main__":
    main()
