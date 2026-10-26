from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Iterable
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

CLASSES = ["benign", "malignant", "normal"]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def image_sha256(path: Path) -> str:
    """Hash decoded pixels so byte-level PNG metadata differences don't defeat deduplication."""
    with Image.open(path) as im:
        rgb = im.convert("RGB")
        payload = f"{rgb.size}".encode() + rgb.tobytes()
    return hashlib.sha256(payload).hexdigest()


def discover_images(data_dir: str | Path, deduplicate: bool = True) -> pd.DataFrame:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {root.resolve()}")
    rows = []
    for label in CLASSES:
        class_dir = root / label
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Expected class folder: {class_dir}")
        for path in sorted(class_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            # BUSI provides lesion masks beside source images. Never train on masks.
            if "mask" in path.stem.lower():
                continue
            try:
                with Image.open(path) as im:
                    im.verify()
                digest = image_sha256(path)
            except Exception as exc:
                print(f"Skipping unreadable image {path}: {exc}")
                continue
            rows.append({"path": str(path.resolve()), "label": label, "sha256": digest})
    if not rows:
        raise RuntimeError(f"No usable images found under {root.resolve()}")
    df = pd.DataFrame(rows)
    # Exact duplicate pixels with conflicting folder labels are ambiguous; stop rather than
    # silently allowing the same image to leak into multiple labels/splits.
    conflicts = df.groupby("sha256")["label"].nunique()
    bad_hashes = set(conflicts[conflicts > 1].index)
    if bad_hashes:
        examples = df[df.sha256.isin(bad_hashes)][["path", "label", "sha256"]].head(10)
        raise ValueError(
            "Exact duplicate image pixels were found under different class labels. "
            "Review these labels manually before training; examples:\n" + examples.to_string(index=False)
        )
    if deduplicate:
        before = len(df)
        df = df.drop_duplicates("sha256", keep="first").reset_index(drop=True)
        print(f"Removed {before - len(df)} exact duplicate image(s).")
    missing = set(CLASSES) - set(df.label)
    if missing:
        raise ValueError(f"No usable images found for class(es): {sorted(missing)}")
    return df


def make_splits(df: pd.DataFrame, output_csv: str | Path, seed: int = 42,
                test_size: float = 0.15, val_size: float = 0.15) -> pd.DataFrame:
    """Create stratified train/val/test splits after exact-duplicate removal.

    BUSI's public image files do not provide a reliable patient ID field. This split therefore
    prevents exact-pixel duplicates crossing splits, but cannot guarantee patient-level separation
    or detect every near-duplicate/related image. Document this limitation in any report.
    """
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["label"]
    )
    val_fraction_of_remaining = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_fraction_of_remaining,
        random_state=seed, stratify=train_val["label"]
    )
    train, val, test = train.copy(), val.copy(), test.copy()
    train["split"] = "train"
    val["split"] = "val"
    test["split"] = "test"
    result = pd.concat([train, val, test], ignore_index=True)
    if result.groupby("sha256")["split"].nunique().max() > 1:
        raise AssertionError("Duplicate leakage detected across splits.")
    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    print("\nSplit counts:\n", pd.crosstab(result["split"], result["label"]))
    print(f"Saved split manifest: {out.resolve()}")
    return result
