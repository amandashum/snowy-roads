"""
This program suggests the visible snow condition in an uploaded road image.

It validates JPEG or PNG content and the 10 MB file-size limit, corrects image
orientation, and applies the same resize, center crop, and normalization used
for training. It loads a local ResNet-18 checkpoint, checks the saved class
order and preprocessing identifier, and returns the suggested class, model
scores, model version, and uncertainty status when a threshold is available.

The four classes are clear, light-covered, medium-to-heavy-covered, and plowed
surface. Scores are not calibrated guarantees of correctness. The program
does not determine road safety or whether a snowplow should be dispatched;
municipal staff make those decisions. Running this file prints a JSON result.
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torchvision import models, transforms

LABELS = json.loads((Path(__file__).parent / "config/labels.json").read_text())
CLASSES = [entry["id"] for entry in LABELS["classes"]]


def preprocessing():
    """
    Return the deterministic transform shared by training and inference.

    Side effects:
        None. Existing files and stored data remain unchanged.

    Returns:
        A deterministic image preprocessing transform.
    """
    return models.ResNet18_Weights.IMAGENET1K_V1.transforms()


def build_model(pretrained=False):
    """
    Construct ResNet-18 with four outputs in the verified class order.

    Side effects:
        Load pretrained weights when requested; download them if absent from the local cache.

    Returns:
        A ResNet-18 model with four classification outputs.
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))
    return model


def load_image(path):
    """
    Decode a JPEG or PNG up to 10 MB and apply its orientation metadata.

    Side effects:
        Read and decode the supplied local image file.

    Returns:
        An orientation-corrected RGB image.
    """
    path = Path(path)
    if path.stat().st_size > 10 * 1024 * 1024:
        raise ValueError("Image exceeds the 10 MB limit")
    with Image.open(path) as image:
        if image.format not in {"JPEG", "PNG"}:
            raise ValueError("Only JPEG and PNG images are supported")
        image.load()
        return ImageOps.exif_transpose(image).convert("RGB")


def load_checkpoint(path, device="cpu"):
    """
    Load a local weights-only checkpoint and verify its inference contract.

    Side effects:
        Read a local checkpoint and load its parameters into the model.

    Returns:
        The evaluation-mode model and checkpoint metadata.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint["classes"] != CLASSES:
        raise ValueError("Checkpoint class mapping does not match config/labels.json")
    if checkpoint["preprocessing"] != "ResNet18_Weights.IMAGENET1K_V1":
        raise ValueError("Unsupported checkpoint preprocessing")
    model = build_model().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def predict(image_path, checkpoint_path, loaded_checkpoint=None):
    """
    Return model scores and a suggestion; scores are not calibrated certainty.

    Side effects:
        Read a local image and checkpoint; run inference on the CPU.

    Returns:
        A dictionary containing the suggested condition, model scores, and limitations.
    """
    model, metadata = loaded_checkpoint if loaded_checkpoint is not None else load_checkpoint(checkpoint_path)
    tensor = preprocessing()(load_image(image_path)).unsqueeze(0)
    with torch.inference_mode():
        scores = model(tensor).softmax(dim=1)[0].tolist()
    index = max(range(len(scores)), key=scores.__getitem__)
    threshold = metadata.get("uncertainty_threshold")
    return {"predicted_class": CLASSES[index], "class_scores": dict(zip(CLASSES, scores)),
            "model_version": metadata["model_version"], "label_mapping_version": LABELS["version"],
            "uncertain": scores[index] < threshold if threshold is not None else None,
            "limitation": "Visible condition only; staff decide whether plowing is needed."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=Path("models/classifier.pt"))
    args = parser.parse_args()
    print(json.dumps(predict(args.image, args.model), indent=2))
