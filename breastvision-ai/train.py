import argparse
from src.engine import train_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BreastVision AI with EfficientNet-B0.")
    parser.add_argument("--manifest", default="artifacts/splits.csv")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0, help="Use 0 on Windows for simplest compatibility")
    args = parser.parse_args()
    train_model(args.manifest, args.output_dir, args.epochs, args.batch_size,
                args.image_size, args.seed, args.patience, args.warmup_epochs, args.num_workers)
