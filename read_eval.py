import argparse
import csv
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


EVAL_DIR_PATTERN = re.compile(r"^eval_(.+)$")
PLANT_PATTERN = re.compile(r"Plant(?P<plant>\d+)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Read the latest 5 eval folders under results and summarize highest-error plants."
    )
    parser.add_argument("results_dir", help="Path to the results folder")
    parser.add_argument("--topk", type=int, default=20, help="How many plants to show")
    parser.add_argument("--latest", type=int, default=5, help="How many latest eval folders to use")
    parser.add_argument("--save-csv", type=str, default=None, help="Optional CSV output path")
    return parser.parse_args()


def parse_eval_time(folder_name):
    match = EVAL_DIR_PATTERN.match(folder_name)
    if not match:
        return None

    time_part = match.group(1)
    digits = re.sub(r"[^0-9]", "", time_part)

    for size, fmt in (
        (14, "%Y%m%d%H%M%S"),
        (12, "%Y%m%d%H%M"),
        (10, "%Y%m%d%H"),
        (8, "%Y%m%d"),
    ):
        if len(digits) >= size:
            try:
                return datetime.strptime(digits[:size], fmt)
            except ValueError:
                pass
    return None


def find_latest_eval_dirs(results_dir, latest_n):
    candidates = []
    for path in Path(results_dir).iterdir():
        if not path.is_dir():
            continue
        if not EVAL_DIR_PATTERN.match(path.name):
            continue

        parsed_time = parse_eval_time(path.name)
        sort_time = parsed_time.timestamp() if parsed_time else path.stat().st_mtime
        candidates.append((sort_time, path))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return [path for _, path in candidates[:latest_n]]


def extract_plants_from_file(file_path):
    plants = []
    for line in file_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = PLANT_PATTERN.search(line)
        if match:
            plants.append(int(match.group("plant")))
    return plants


def summarize(eval_dirs):
    total_counts = Counter()
    folder_presence = Counter()
    folder_details = defaultdict(Counter)

    used_dirs = []
    for eval_dir in eval_dirs:
        wrong_file = eval_dir / "always_wrong_samples.txt"
        if not wrong_file.exists():
            continue

        used_dirs.append(eval_dir)
        plants = extract_plants_from_file(wrong_file)
        current_counts = Counter(plants)

        for plant, count in current_counts.items():
            total_counts[plant] += count
            folder_presence[plant] += 1
            folder_details[plant][eval_dir.name] = count

    return used_dirs, total_counts, folder_presence, folder_details


def save_csv(csv_path, rows):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "plant_id",
                "total_errors",
                "folders_hit",
                "folder_hit_rate",
                "per_folder_counts",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    latest_eval_dirs = find_latest_eval_dirs(results_dir, args.latest)
    used_dirs, total_counts, folder_presence, folder_details = summarize(latest_eval_dirs)

    if not used_dirs:
        print("No usable eval folders with always_wrong_samples.txt were found.")
        return

    print("Used eval folders:")
    for path in used_dirs:
        print(f"- {path}")

    print("\nHighest-error plants:")
    rows = []
    for plant, total_errors in total_counts.most_common(args.topk):
        folders_hit = folder_presence[plant]
        folder_hit_rate = folders_hit / len(used_dirs)
        per_folder_counts = ", ".join(
            f"{folder}:{count}" for folder, count in sorted(folder_details[plant].items())
        )
        print(
            f"Plant{plant}: total_errors={total_errors}, "
            f"folders_hit={folders_hit}/{len(used_dirs)}, "
            f"folder_hit_rate={folder_hit_rate:.2%}"
        )
        rows.append(
            {
                "plant_id": f"Plant{plant}",
                "total_errors": total_errors,
                "folders_hit": f"{folders_hit}/{len(used_dirs)}",
                "folder_hit_rate": f"{folder_hit_rate:.4f}",
                "per_folder_counts": per_folder_counts,
            }
        )

    if args.save_csv:
        save_csv(args.save_csv, rows)
        print(f"\nSaved CSV to {args.save_csv}")


if __name__ == "__main__":
    main()
