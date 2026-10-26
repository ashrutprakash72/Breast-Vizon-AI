from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.utils.class_weight import compute_class_weight
from .data import CLASSES
from .model import build_model, freeze_backbone, unfreeze_all

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

class BUSIDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_size: int = 224, training: bool = False):
        self.frame = frame.reset_index(drop=True)
        self.image_size = image_size
        self.training = training
        if training:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=8),
                transforms.ColorJitter(brightness=0.08, contrast=0.08),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ])
    def __len__(self):
        return len(self.frame)
    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        with Image.open(row.path) as im:
            image = im.convert("RGB")
        x = self.transform(image)
        y = CLASSES.index(row.label)
        return x, y


def make_loader(frame, batch_size, image_size, training, num_workers=0):
    ds = BUSIDataset(frame, image_size=image_size, training=training)
    return DataLoader(ds, batch_size=batch_size, shuffle=training,
                      num_workers=num_workers, pin_memory=False, drop_last=False)


def train_one_epoch(model, loader, criterion, optimizer, device, scaler, use_amp):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / max(total, 1), total_correct / max(total, 1)

@torch.no_grad()
def evaluate_loss(model, loader, criterion, device, use_amp):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / max(total, 1), total_correct / max(total, 1)


def train_model(manifest: str, output_dir: str, epochs: int = 20, batch_size: int = 8,
                image_size: int = 224, seed: int = 42, patience: int = 5,
                warmup_epochs: int = 3, num_workers: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Device: {device}; AMP: {use_amp}")
    frame = pd.read_csv(manifest)
    train_df, val_df = frame[frame.split == "train"], frame[frame.split == "val"]
    if train_df.empty or val_df.empty:
        raise ValueError("Manifest must contain non-empty train and val splits. Run prepare_data.py first.")
    train_loader = make_loader(train_df, batch_size, image_size, True, num_workers)
    val_loader = make_loader(val_df, batch_size, image_size, False, num_workers)
    model = build_model(len(CLASSES), pretrained=True).to(device)
    freeze_backbone(model)
    labels = np.array([CLASSES.index(label) for label in train_df.label])
    weights = compute_class_weight(class_weight="balanced", classes=np.arange(len(CLASSES)), y=labels)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
    scheduler = None
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    best_loss, bad_epochs, history = float("inf"), 0, []
    best_path = out / "best_model.pth"

    for epoch in range(1, epochs + 1):
        if epoch == warmup_epochs + 1:
            print("Unfreezing full model for fine-tuning.")
            unfreeze_all(model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, use_amp)
        val_loss, val_acc = evaluate_loss(model, val_loader, criterion, device, use_amp)
        if scheduler is not None:
            scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]
        row = {"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_acc,
               "val_loss": val_loss, "val_accuracy": val_acc, "learning_rate": lr}
        history.append(row)
        print(f"Epoch {epoch:02d}/{epochs} | train loss {train_loss:.4f} acc {train_acc:.3f} | val loss {val_loss:.4f} acc {val_acc:.3f} | lr {lr:.2e}")
        if val_loss < best_loss:
            best_loss, bad_epochs = val_loss, 0
            torch.save({"model_state_dict": model.state_dict(), "class_names": CLASSES,
                        "image_size": image_size, "mean": MEAN, "std": STD,
                        "best_val_loss": best_loss, "epoch": epoch}, best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping after {epoch} epochs.")
                break
    pd.DataFrame(history).to_csv(out / "training_history.csv", index=False)
    with open(out / "training_config.json", "w", encoding="utf-8") as f:
        json.dump({"epochs_requested": epochs, "batch_size": batch_size, "image_size": image_size,
                   "seed": seed, "patience": patience, "warmup_epochs": warmup_epochs,
                   "device": str(device), "classes": CLASSES}, f, indent=2)
    print(f"Best checkpoint: {best_path.resolve()}")
    return best_path
