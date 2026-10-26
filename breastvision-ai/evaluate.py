import argparse
from src.evaluate import evaluate_checkpoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the best checkpoint on the held-out test split.")
    parser.add_argument("--manifest", default="artifacts/splits.csv")
    parser.add_argument("--checkpoint", default="artifacts/best_model.pth")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    evaluate_checkpoint(args.manifest, args.checkpoint, args.output_dir, args.batch_size, args.num_workers)
