import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

IMAGE_SIZE = 224

def main():
    parser = argparse.ArgumentParser(description="Train a skin-lesion image classifier.")
    parser.add_argument("--data-dir", default="data", help="Folder containing train/ and val/")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--output-dir", default=".", help="Where model files will be saved")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise SystemExit(
            "Dataset folders not found. Expected data/train/<class_name>/ and "
            "data/val/<class_name>/; see README.md."
        )

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(train_dir, transform=train_transform)
    val_ds = datasets.ImageFolder(val_dir, transform=val_transform)
    if len(train_ds.classes) < 2:
        raise SystemExit("Please provide at least two class folders in data/train/.")
    if train_ds.classes != val_ds.classes:
        raise SystemExit(
            f"Train/validation class folders must match.\n"
            f"Train: {train_ds.classes}\nValidation: {val_ds.classes}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Transfer learning: start from ImageNet-pretrained MobileNetV2 weights.
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.last_channel, len(train_ds.classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_accuracy = -1.0

    print(f"Device: {device}")
    print(f"Classes: {train_ds.classes}")
    print(f"Training images: {len(train_ds)} | Validation images: {len(val_ds)}")

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / max(total, 1)
        train_accuracy = correct / max(total, 1)

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        val_accuracy = val_correct / max(val_total, 1)
        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"loss={train_loss:.4f} | train_acc={train_accuracy:.3f} | "
            f"val_acc={val_accuracy:.3f}"
        )

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.cpu().state_dict(), output_dir / "best_model.pth")
            model.to(device)
            with open(output_dir / "class_names.json", "w", encoding="utf-8") as f:
                json.dump(train_ds.classes, f, indent=2)

    print(f"Finished. Best validation accuracy: {best_accuracy:.3f}")
    print(f"Saved model files in: {output_dir.resolve()}")
    print("Important: report validation results honestly and do not describe this as a medical diagnostic tool.")

if __name__ == "__main__":
    main()
