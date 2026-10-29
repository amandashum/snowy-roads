"""
This program checks essential classifier behavior with small automated tests.

It verifies that training and validation camera groups do not overlap, that
split selection is reproducible, and that a missing class is rejected. It
checks evaluation calculations when a minority class is completely missed.
Temporary image fixtures confirm that upload validation accepts real PNG
content and rejects an unsupported format disguised with a JPEG filename.

These tests do not train a production model, establish classification quality,
complete a dataset leakage audit, or validate a dashboard. Run them with pytest
and inspect any failures before using the classifier in a demonstration.
"""

import pytest
from pathlib import Path
from uuid import uuid4
from PIL import Image

from inference import load_image
from train import split_indices
from evaluate import metrics


def test_camera_split_is_disjoint_and_reproducible():
    """
    Keep each camera in one split and retain all four classes.

    Side effects:
        Create temporary test images when required and raise on failed assertions.

    Returns:
        None. Results are written or assertions are checked directly.
    """
    samples = [(f"cam{camera}_{label}.jpeg", label) for camera in range(20) for label in range(4)]
    train, valid = split_indices(samples)
    assert (train, valid) == split_indices(samples)
    assert set(train).isdisjoint(valid)
    assert len(train) + len(valid) == len(samples)
    train_cameras = {samples[i][0].split("_")[0] for i in train}
    assert train_cameras.isdisjoint({samples[i][0].split("_")[0] for i in valid})


def test_missing_validation_class_is_rejected():
    """
    Refuse a split that cannot support four-class evaluation.

    Side effects:
        Create temporary test images when required and raise on failed assertions.

    Returns:
        None. Results are written or assertions are checked directly.
    """
    with pytest.raises(ValueError, match="Missing classes"):
        split_indices([("cam1_a.jpeg", 0), ("cam2_a.jpeg", 1)])


def test_metrics_account_for_missed_classes():
    """
    Count missed minority classes rather than dropping them from macro F1.

    Side effects:
        Create temporary test images when required and raise on failed assertions.

    Returns:
        None. Results are written or assertions are checked directly.
    """
    report = metrics([[2, 0, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    assert report["accuracy"] == 0.8
    assert report["macro_f1"] == pytest.approx(0.7)
    assert report["per_class"][1]["recall"] == 0


def test_image_validation_uses_actual_format():
    """
    Accept a valid PNG and reject a disguised unsupported image.

    Side effects:
        Create temporary test images when required and raise on failed assertions.

    Returns:
        None. Results are written or assertions are checked directly.
    """
    folder = Path("models")
    folder.mkdir(exist_ok=True)
    valid = folder / f"test-{uuid4().hex}.png"
    invalid = folder / f"test-{uuid4().hex}.jpg"
    try:
        Image.new("RGB", (20, 20)).save(valid)
        assert load_image(valid).mode == "RGB"
        Image.new("RGB", (20, 20)).save(invalid, format="BMP")
        with pytest.raises(ValueError, match="JPEG and PNG"):
            load_image(invalid)
    finally:
        valid.unlink(missing_ok=True)
        invalid.unlink(missing_ok=True)
