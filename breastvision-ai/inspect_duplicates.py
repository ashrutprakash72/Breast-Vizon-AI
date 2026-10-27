from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from PIL import Image

ROOT = Path("data/BUSI")
CLASSES = ("benign", "malignant", "normal")
groups = defaultdict(list)

for label in CLASSES:
    folder = ROOT / label

    for path in folder.iterdir():
        if not path.is_file() or "mask" in path.stem.lower():
            continue

        try:
            with Image.open(path) as image:
                image.load()
                digest = sha256(image.convert("RGB").tobytes()).hexdigest()
                groups[digest].append((label, path))
        except Exception as exc:
            print(f"Unreadable: {path} — {exc}")

conflicts = [
    entries for entries in groups.values()
    if len({label for label, _ in entries}) > 1
]

print(f"Unique pixel groups: {len(groups)}")
print(f"Cross-class duplicate groups: {len(conflicts)}")

for index, entries in enumerate(conflicts, start=1):
    print(f"\nConflict {index}:")
    for label, path in entries:
        print(f"  {label}: {path}")
