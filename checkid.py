import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description="Export file names stored in a .pt split file.")
    parser.add_argument("--pt_path", help="Path to .pt file, e.g. splits/cleantest.pt")
    parser.add_argument("--save", help="Optional output txt/csv path")
    args = parser.parse_args()

    pairs = torch.load(args.pt_path)
    output_lines = []

    for idx, (rgb_path, _hsi, label) in enumerate(pairs):
        path_obj = Path(rgb_path)
        output_lines.append(
            f"index={idx}, filename={path_obj.name}, stem={path_obj.stem}, label={int(label)}, path={rgb_path}"
        )

    if args.save:
        output_path = Path(args.save)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(output_lines), encoding="utf-8")
        print(f"Saved {len(output_lines)} lines to {output_path}")
    else:
        for line in output_lines:
            print(line)


if __name__ == "__main__":
    main()
