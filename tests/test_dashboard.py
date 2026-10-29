"""
This program verifies local report handling and the Streamlit dashboard flow.

It tests invalid image rejection, persistence when inference is unavailable,
preservation of model suggestions during staff review, and traversal rejection.
Headless Streamlit checks cover both workspace views and missing-photo feedback.
All fixture reports are written into a task-specific temporary directory.
These checks do not establish browser accessibility or model accuracy.
"""

import base64
import io
import json
from pathlib import Path
from uuid import uuid4

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

import reports


@pytest.fixture
def storage(monkeypatch):
    """
    Isolate fixture photos and reports from real local submissions.

    Side effects:
        Create a temporary folder and remove only its known test files afterward.

    Returns:
        Yield the test storage directory.
    """
    folder = Path("models") / f"dashboard-test-{uuid4().hex}"
    folder.mkdir(parents=True)
    monkeypatch.setattr(reports, "RECORDS", folder)
    monkeypatch.setattr(reports, "UPLOADS", folder)
    yield folder
    for path in folder.iterdir():
        path.unlink()
    folder.rmdir()


def test_report_preserves_suggestion_during_staff_review(storage, monkeypatch):
    """
    Preserve the initial prediction while recording a different staff condition.

    Side effects:
        Create and review a temporary report with explicitly mocked inference.

    Returns:
        None. Assertions verify the saved record.
    """
    import inference
    monkeypatch.setattr(inference, "predict", lambda *args: {
        "predicted_class": "light", "class_scores": {"clear": .1, "light": .6, "medium": .2, "plowed": .1},
        "model_version": "test-fixture", "label_mapping_version": "1", "uncertain": None})
    image = io.BytesIO()
    Image.new("RGB", (32, 32)).save(image, format="PNG")
    record = reports.create_report({"image_base64": base64.b64encode(image.getvalue()).decode()})
    updated = reports.review_report(record["record_id"], {
        "staff_class": "medium", "priority": "assess-soon", "action": "assess-plowing", "staff_note": "Inspect road"})
    saved = json.loads(reports.record_path(record["record_id"]).read_text())
    assert updated["review_status"] == "corrected"
    assert saved["predicted_class"] == "light"
    assert saved["staff_class"] == "medium"
    assert saved["captured_at"] is None


def test_unavailable_model_keeps_report_for_review(storage, monkeypatch):
    """
    Save a report without inventing a suggestion when inference fails.

    Side effects:
        Save a temporary report with intentionally unavailable inference.

    Returns:
        None. Assertions verify the manual-review state.
    """
    import inference

    def unavailable(*args):
        """
        Simulate a missing local checkpoint.

        Side effects:
            Raise FileNotFoundError.

        Returns:
            None. This fixture always raises.
        """
        raise FileNotFoundError("fixture")

    monkeypatch.setattr(inference, "predict", unavailable)
    image = io.BytesIO()
    Image.new("RGB", (32, 32)).save(image, format="PNG")
    record = reports.create_report({"image_base64": base64.b64encode(image.getvalue()).decode()})
    assert record["inference_status"] == "unavailable"
    assert record["predicted_class"] is None
    assert reports.record_path(record["record_id"]).exists()


def test_invalid_upload_and_traversal_are_rejected(storage):
    """
    Reject invalid image content and report identifiers before writing records.

    Side effects:
        Exercise validation against isolated test storage.

    Returns:
        None. Assertions verify rejection and empty storage.
    """
    with pytest.raises(ValueError):
        reports.create_report({"image_base64": "not base64!"})
    with pytest.raises(ValueError):
        reports.record_path("../outside")
    assert not list(storage.iterdir())


def test_dashboard_views_and_missing_photo_message(storage):
    """
    Render both dashboard views and require a photo before report submission.

    Side effects:
        Run the Streamlit app headlessly and simulate workspace controls.

    Returns:
        None. Assertions verify displayed feedback and absence of exceptions.
    """
    app = AppTest.from_file(Path("app.py").resolve(), default_timeout=30).run()
    assert not app.exception
    app.button[0].click().run()
    assert any("Choose a road photo" in message.value for message in app.error)
    app.radio(key="workspace").set_value("Staff review").run()
    assert not app.exception
    assert any("No reports" in message.value for message in app.info)
