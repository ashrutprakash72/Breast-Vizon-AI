from pathlib import Path

paths = [
    Path("data/BUSI/benign/benign (433).png"),
    Path("data/BUSI/malignant/malignant (145).png"),
]

report = Path("artifacts/excluded_conflicts.txt")
report.parent.mkdir(parents=True, exist_ok=True)

with report.open("w", encoding="utf-8") as f:
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        f.write(f"{path.resolve()}\n")

print(f"Recorded {len(paths)} conflicting images in {report}")
print("Original dataset files were not modified.")
