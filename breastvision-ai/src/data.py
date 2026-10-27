
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

CLASSES = ["benign", "malignant", "normal"]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def image_sha256(path: Path) -> str:
    """Hash decoded pixels so PNG metadata differences do not defeat deduplication."""
    with Image.open(path) as im:
        rgb = im.convert("RGB")
        payload = f"{rgb.size}".encode() + rgb.tobytes()
    return hashlib.sha256(payload).hexdigest()


def load_excluded_paths(
    data_dir: Path,
    exclusion_file: str | Path | None = None,
) -> set[str]:
    """Load image paths explicitly excluded from the experiment."""
    if exclusion_file is None:
        exclusion_file = data_dir.parent.parent / "artifacts" / "excluded_conflicts.txt"

    exclusion_file = Path(exclusion_file)
    if not exclusion_file.exists():
        return set()

    excluded = set()
    for line in exclusion_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            excluded.add(str(Path(line).resolve()).casefold())

    print(f"Loaded {len(excluded)} excluded image path(s) from {exclusion_file}")
    return excluded


def discover_images(
    data_dir: str | Path,
    deduplicate: bool = True,
    exclusion_file: str | Path | None = None,
) -> pd.DataFrame:
    root = Path(data_dir).resolve()

    if not root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    excluded_paths = load_excluded_paths(root, exclusion_file)
    rows = []

    for label in CLASSES:
        class_dir = root / label

        if not class_dir.is_dir():
            raise FileNotFoundError(f"Expected class folder: {class_dir}")

        for path in sorted(class_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            # Never train on BUSI lesion masks.
            if "mask" in path.stem.lower():
                continue

            # Exclude only the paths explicitly recorded in the exclusion file.
            if str(path.resolve()).casefold() in excluded_paths:
                print(f"Excluding conflicting image: {path}")
                continue

            try:
                with Image.open(path) as im:
                    im.verify()
                digest = image_sha256(path)
            except Exception as exc:
                print(f"Skipping unreadable image {path}: {exc}")
                continue

            rows.append({
                "path": str(path.resolve()),
                "label": label,
                "sha256": digest,
            })

    if not rows:
        raise RuntimeError(f"No usable images found under {root}")

    df = pd.DataFrame(rows)

    # Stop if identical pixels still appear under different labels.
    conflicts = df.groupby("sha256")["label"].nunique()
    bad_hashes = set(conflicts[conflicts > 1].index)

    if bad_hashes:
        examples = df[df.sha256.isin(bad_hashes)][
            ["path", "label", "sha256"]
        ].head(10)

        raise ValueError(
            "Exact duplicate image pixels were found under different class labels. "
            "Review the exclusion file or labels before training:\n"
            + examples.to_string(index=False)
        )

    if deduplicate:
        before = len(df)
        df = df.drop_duplicates("sha256", keep="first").reset_index(drop=True)
        print(f"Removed {before - len(df)} exact duplicate image(s).")

    missing = set(CLASSES) - set(df["label"])
    if missing:
        raise ValueError(f"No usable images found for class(es): {sorted(missing)}")

    print("\nUsable images by class:")
    print(df["label"].value_counts().reindex(CLASSES, fill_value=0))

    return df


def make_splits(
    df: pd.DataFrame,
    output_csv: str | Path,
    seed: int = 42,
    test_size: float = 0.15,
    val_size: float = 0.15,
) -> pd.DataFrame:
    """Create stratified splits after exclusions and exact-pixel deduplication.

    Patient-level separation cannot be guaranteed because the dataset does not
    provide a reliable patient ID field. Related or near-duplicate images may
    also remain, so evaluation results should be interpreted cautiously.
    """
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["label"],
    )

    val_fraction_of_remaining = val_size / (1.0 - test_size)

    train, val = train_test_split(
        train_val,
        test_size=val_fraction_of_remaining,
        random_state=seed,
        stratify=train_val["label"],
    )

    train = train.copy()
    val = val.copy()
    test = test.copy()

    train["split"] = "train"
    val["split"] = "val"
    test["split"] = "test"

    result = pd.concat([train, val, test], ignore_index=True)

    if result.groupby("sha256")["split"].nunique().max() > 1:
        raise AssertionError("Duplicate leakage detected across splits.")

    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)

    print("\nSplit counts:")
    print(pd.crosstab(result["split"], result["label"]))
    print(f"\nSaved split manifest: {out.resolve()}")

    return result
