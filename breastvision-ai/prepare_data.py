import argparse
from pathlib import Path
from src.data import discover_images, make_splits

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate BUSI folders and create reproducible splits.")
    parser.add_argument("--data-dir", default="data/BUSI", help="Folder containing benign/, malignant/, normal/")
    parser.add_argument("--output", default="artifacts/splits.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    args = parser.parse_args()
    if args.test_size <= 0 or args.val_size <= 0 or args.test_size + args.val_size >= 1:
        parser.error("test-size and val-size must be > 0 and sum to < 1")
    df = discover_images(args.data_dir, deduplicate=True)
    print("Usable class counts:\n", df.label.value_counts())
    make_splits(df, args.output, seed=args.seed, test_size=args.test_size, val_size=args.val_size)
