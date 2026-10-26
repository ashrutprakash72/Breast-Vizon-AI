from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             balanced_accuracy_score, f1_score, roc_auc_score, roc_curve, auc)
from sklearn.preprocessing import label_binarize
from .data import CLASSES
from .engine import make_loader
from .model import build_model

@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    ys, probs = [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        ys.extend(labels.numpy().tolist())
        probs.extend(torch.softmax(logits, dim=1).cpu().numpy().tolist())
    return np.asarray(ys), np.asarray(probs)

def evaluate_checkpoint(manifest, checkpoint_path, output_dir="artifacts", batch_size=8, num_workers=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    class_names = ckpt.get("class_names", CLASSES)
    model = build_model(len(class_names), pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    frame = pd.read_csv(manifest)
    test_df = frame[frame.split == "test"]
    if test_df.empty:
        raise ValueError("No test rows in manifest.")
    loader = make_loader(test_df, batch_size, ckpt.get("image_size", 224), False, num_workers)
    y_true, probs = collect_predictions(model, loader, device)
    y_pred = probs.argmax(axis=1)
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    report = classification_report(y_true, y_pred, labels=list(range(len(class_names))),
                                   target_names=class_names, output_dict=True, zero_division=0)
    metrics = {"accuracy": float(accuracy_score(y_true, y_pred)),
               "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
               "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
               "classification_report": report,
               "note": "Image-level evaluation only; BUSI patient identifiers are not reliably available in the folder release."}
    try:
        y_bin = label_binarize(y_true, classes=list(range(len(class_names))))
        metrics["roc_auc_ovr_macro"] = float(roc_auc_score(y_bin, probs, average="macro", multi_class="ovr"))
    except ValueError as exc:
        metrics["roc_auc_ovr_macro"] = None
        metrics["roc_auc_note"] = str(exc)
    (out / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame({"path": test_df.path.to_list(), "true_label": [class_names[i] for i in y_true],
                  "predicted_label": [class_names[i] for i in y_pred],
                  **{f"prob_{name}": probs[:, i] for i, name in enumerate(class_names)}}).to_csv(out / "test_predictions.csv", index=False)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    plt.figure(figsize=(7, 6)); sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted label"); plt.ylabel("True label"); plt.title("BUSI test confusion matrix")
    plt.tight_layout(); plt.savefig(out / "confusion_matrix.png", dpi=180); plt.close()
    print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2))
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    print(f"Saved metrics and plots to {out.resolve()}")
    return metrics
