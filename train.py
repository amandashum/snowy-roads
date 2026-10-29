"""
This program trains a four-class classifier for visible road snow conditions.

It reads the existing training folders and reserves approximately 20 percent
of their camera IDs for validation. Images from one camera remain together
within this training/validation split. The original held-out test folder is
not used for training or checkpoint selection, and source images are not moved.

It loads pretrained ResNet-18 weights, freezes the backbone and its batch
normalization statistics, and trains a new four-output classification layer.
Each epoch reports loss and accuracy for training and validation. The model
with the lowest validation loss is saved with its class order, preprocessing
identifier, seed, and version. Separate JSON files preserve split membership
and epoch history. An optional image limit supports explicitly marked smoke
tests; those checkpoints must not be used to claim held-out performance.

The program does not audit all duplicate images, resolve existing train/test
camera overlap, select an uncertainty threshold, or make plowing decisions.
"""

import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from inference import CLASSES, build_model, preprocessing

DATA = Path("data/Snow-Covered-Roads-Dataset-main/dataset")


def split_indices(samples, seed=42):
    """
    Reserve 20 percent of training cameras for validation, keeping cameras together.

    Side effects:
        None. Existing files and stored data remain unchanged.

    Returns:
        Training and validation index lists with no shared camera IDs.
    """
    cameras = sorted({Path(path).name.split("_")[0] for path, _ in samples})
    if len(cameras) < 2:
        raise ValueError("At least two cameras are required for grouped validation")
    random.Random(seed).shuffle(cameras)
    validation = set(cameras[:max(1, len(cameras) // 5)])
    train, valid = [], []
    for index, (path, _) in enumerate(samples):
        destination = valid if Path(path).name.split("_")[0] in validation else train
        destination.append(index)
    for name, indices in (("training", train), ("validation", valid)):
        if {samples[i][1] for i in indices} != set(range(len(CLASSES))):
            raise ValueError(f"Missing classes in {name}; inspect camera grouping")
    return train, valid


def run_epoch(model, loader, device, optimizer=None):
    """
    Measure loss and accuracy, updating only the head when an optimizer is supplied.

    Side effects:
        Update trainable model parameters when an optimizer is supplied.
        Print batch progress and running loss during training or validation.

    Returns:
        A dictionary containing average loss, accuracy, and image count.
    """
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    phase = "Training" if optimizer is not None else "Validation"
    for batch, (images, targets) in enumerate(loader, start=1):
        images, targets = images.to(device), targets.to(device)
        with torch.set_grad_enabled(optimizer is not None):
            outputs = model(images)
            loss = torch.nn.functional.cross_entropy(outputs, targets)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * len(targets)
        correct += (outputs.argmax(1) == targets).sum().item()
        count += len(targets)
        if batch == 1 or batch % 10 == 0 or batch == len(loader):
            print(f"  {phase} | Batch {batch}/{len(loader)} | "
                  f"Loss: {total_loss / count:.4f} | Accuracy: {correct / count:.1%}", flush=True)
    return {"loss": total_loss / count, "accuracy": correct / count, "count": count}


def main():
    """
    Train the classifier and save the checkpoint with the lowest validation loss.

    Side effects:
        Read training images and cached pretrained weights, print progress, and write
        model checkpoints, split manifests, and training history.

    Returns:
        None. Results are written or assertions are checked directly.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, help="Per-split smoke-test limit; not full evaluation")
    parser.add_argument("--output", type=Path, default=Path("models/classifier.pt"))
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or (args.limit is not None and args.limit < 4):
        parser.error("Use positive epochs/batch size and a limit of at least four")
    torch.manual_seed(args.seed)

    dataset = ImageFolder(args.data / "train", transform=preprocessing())

    if dataset.classes != CLASSES:
        raise ValueError(f"Expected {CLASSES}; found {dataset.classes}")
    
    train, valid = split_indices(dataset.samples, args.seed)
    rng = random.Random(args.seed)

    rng.shuffle(train)
    rng.shuffle(valid)

    if args.limit:
        train, valid = train[:args.limit], valid[:args.limit]
    loaders = [DataLoader(Subset(dataset, indices), batch_size=args.batch_size,
                          shuffle=shuffle, num_workers=0) for indices, shuffle in ((train, True), (valid, False))]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(pretrained=True).to(device)
    
    for parameter in model.parameters():
        parameter.requires_grad = False
    
    for parameter in model.fc.parameters():
        parameter.requires_grad = True
    
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
   
    manifest = {name: [str(Path(dataset.samples[i][0]).relative_to(args.data)) for i in indices]
                for name, indices in (("train", train), ("validation", valid))}
    
    args.output.with_suffix(".split.json").write_text(json.dumps(manifest, indent=2))
    
    history, best = [], float("inf")
    
    print(f"Device: {device} | Training images: {len(train):,} | "
          f"Validation images: {len(valid):,}", flush=True)
   
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1} out of {args.epochs}", flush=True)
        
        result = {"epoch": epoch + 1, "train": run_epoch(model, loaders[0], device, optimizer),
                  "validation": run_epoch(model, loaders[1], device)}
        
        history.append(result)
        
        print(f"Epoch {epoch + 1}/{args.epochs} complete | "
              f"Training loss: {result['train']['loss']:.4f} | "
              f"Validation loss: {result['validation']['loss']:.4f} | "
              f"Validation accuracy: {result['validation']['accuracy']:.1%}", flush=True)
        
        if result["validation"]["loss"] < best:
            best = result["validation"]["loss"]
            torch.save({"state_dict": model.cpu().state_dict(), "classes": CLASSES,
                        "preprocessing": "ResNet18_Weights.IMAGENET1K_V1",
                        "model_version": args.output.stem, "seed": args.seed,
                        "epoch": epoch + 1, "smoke_test": bool(args.limit),
                        "uncertainty_threshold": None}, args.output)
        
            model.to(device)
        
            print(f"  Best model saved: {args.output}", flush=True)
        
        args.output.with_suffix(".history.json").write_text(json.dumps(history, indent=2))

    from matplotlib.figure import Figure

    figure = Figure(figsize=(8, 5))
    axes = figure.subplots()
    
    epochs = [row["epoch"] for row in history]
    
    axes.plot(epochs, [row["train"]["loss"] for row in history],
              marker="o", label="Training loss")
    axes.plot(epochs, [row["validation"]["loss"] for row in history],
              marker="s", linestyle="--", label="Validation loss")
    axes.set(title="Training and validation loss",
             xlabel="Epoch", ylabel="Cross-entropy loss")
    
    axes.set_xticks(epochs)
    axes.grid(alpha=0.25)
    axes.legend()
    
    figure.tight_layout()
    graph_path = Path("graphs") / args.output.with_suffix(".loss.png").name
    
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(graph_path, dpi=200)
    print(f"Loss graph saved: {graph_path}", flush=True)


if __name__ == "__main__":
    main()
