"""
This program measures a trained road classifier on the existing held-out test set.

It loads a local checkpoint, verifies the four-class mapping and preprocessing
contract, and rejects checkpoints marked as smoke tests. It applies the same
image preprocessing used during training and runs the model without updating
its parameters. It calculates accuracy, per-class precision and recall,
per-class F1, macro F1, and a confusion matrix with actual classes as rows
and predicted classes as columns.

It also compares accuracy with a baseline that always predicts the most
common class in the source training folders. CPU pipeline time includes image
loading, preprocessing, and batched inference; it is not a measurement of
interactive upload latency. Results are printed and saved as a JSON report.
The test set is not used to tune thresholds or select model parameters.
Existing camera overlap and near-duplicate leakage still require a data audit.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from inference import CLASSES, load_checkpoint, preprocessing
from train import DATA


def metrics(matrix):
    """
    Calculate per-class metrics and macro F1 from a true-row confusion matrix.

    Side effects:
        None. Existing files and stored data remain unchanged.

    Returns:
        A dictionary containing accuracy, macro F1, per-class metrics, and the confusion matrix.
    """
    rows = []
    for i, label in enumerate(CLASSES):
        tp = matrix[i][i]
        support = sum(matrix[i])
        predicted = sum(row[i] for row in matrix)
        precision = tp / predicted if predicted else 0
        recall = tp / support if support else 0
        rows.append({"class": label, "support": support, "precision": precision,
                     "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0})
    count = sum(sum(row) for row in matrix)
    return {"accuracy": sum(matrix[i][i] for i in range(len(CLASSES))) / count if count else 0,
            "macro_f1": sum(row["f1"] for row in rows) / len(CLASSES),
            "per_class": rows, "confusion_matrix": matrix, "count": count}


def main():
    """
    Evaluate the test folder without selecting thresholds or tuning model parameters.

    Side effects:
        Read the model and test images, print metrics, and write an evaluation report.

    Returns:
        None. Results are written or assertions are checked directly.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--model", type=Path, default=Path("models/classifier.pt"))
    parser.add_argument("--output", type=Path, default=Path("reports/evaluation.json"))
    args = parser.parse_args()
    dataset = ImageFolder(args.data / "test", transform=preprocessing())
    if dataset.classes != CLASSES:
        raise ValueError("Test classes do not match the checkpoint contract")
    model, metadata = load_checkpoint(args.model)
    if metadata.get("smoke_test"):
        raise ValueError("Smoke-test checkpoints are not suitable for held-out evaluation")
    matrix = [[0] * len(CLASSES) for _ in CLASSES]
    total_loss, count = 0.0, 0
    batch_losses = []
    loader = DataLoader(dataset, batch_size=32)
    print(f"Evaluating {len(dataset):,} test images", flush=True)
    start = time.perf_counter()
    with torch.inference_mode():
        for batch, (images, targets) in enumerate(loader, start=1):
            outputs = model(images)
            batch_losses.append(
                torch.nn.functional.cross_entropy(outputs, targets).item()
            )
            total_loss += torch.nn.functional.cross_entropy(outputs, targets, reduction="sum").item()
            count += len(targets)
            for actual, predicted in zip(targets.tolist(), outputs.argmax(1).tolist()):
                matrix[actual][predicted] += 1
            if batch == 1 or batch % 10 == 0 or batch == len(loader):
                print(f"  Evaluation | Batch {batch}/{len(loader)} | "
                      f"Eval loss: {total_loss / count:.4f}", flush=True)
    report = metrics(matrix)
    report["loss"] = total_loss / count
    report["cpu_pipeline_seconds_per_image"] = (time.perf_counter() - start) / len(dataset)
    print(f"Evaluation complete | Eval loss: {report['loss']:.4f} | "
          f"Accuracy: {report['accuracy']:.1%}", flush=True)
    train_counts = [len(list((args.data / "train" / label).glob("*.jpeg"))) for label in CLASSES]
    majority = max(range(len(CLASSES)), key=train_counts.__getitem__)
    report["majority_baseline_accuracy"] = sum(matrix[majority]) / len(dataset)
    report["model_version"] = metadata["model_version"]
    report["classes"] = CLASSES
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    from matplotlib.figure import Figure

    figure = Figure(figsize=(8, 5))
    axes = figure.subplots()
    axes.plot(range(1, len(batch_losses) + 1), batch_losses,
              label="Test batch loss")
    axes.set(title="Evaluation loss",
             xlabel="Test batch", ylabel="Cross-entropy loss")
    axes.grid(alpha=0.25)
    axes.legend()
    figure.tight_layout()
    graph_path = Path("graphs") / args.output.with_suffix(".loss.png").name
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(graph_path, dpi=200)
    print(f"Evaluation loss graph saved: {graph_path}", flush=True)


if __name__ == "__main__":
    main()
