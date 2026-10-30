"""
This program verifies local report handling and the Streamlit dashboard flow.

It tests invalid image rejection, persistence when inference is unavailable,
preservation of model suggestions during staff review, and traversal rejection.
Headless Streamlit checks cover submissions, workspace switching, staff decisions,
queue filters, and missing-photo feedback. Cache checks cover checkpoint refresh
and smoke-test rejection.
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
    assert any("Drag and drop one JPEG or PNG" in message.value for message in app.caption)
    app.button(key="analyze_photo").click().run()
    assert any("Please upload a road photo before analyzing" == message.value for message in app.error)
    app.radio(key="workspace").set_value("Staff review").run()
    assert not app.exception
    assert any("No reports" in message.value for message in app.info)


@pytest.mark.parametrize("available", [True, False])
def test_submission_receipt_survives_workspace_switch(storage, monkeypatch, available):
    """Open results once, reopen without rerunning inference, and retain the receipt."""
    import streamlit as st
    from components import report_form

    image = io.BytesIO()
    Image.new("RGB", (32, 32)).save(image, format="PNG")
    image.seek(0)
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: image)
    calls = []

    def predict(image_path, model_path):
        calls.append((image_path, model_path))
        if not available:
            raise FileNotFoundError("fixture")
        return {
            "predicted_class": "light",
            "class_scores": {"clear": .1, "light": .6, "medium": .2, "plowed": .1},
            "model_version": "test-fixture", "label_mapping_version": "1", "uncertain": None}

    monkeypatch.setattr(report_form, "classify_photo", predict)
    app = AppTest.from_file(Path("app.py").resolve(), default_timeout=30).run()
    app.button(key="analyze_photo").click().run()
    assert any("Confirm local storage" in message.value for message in app.error)
    assert not list(storage.glob("*.json"))
    app.checkbox[0].check()
    app.text_input[0].set_value("Test intersection")
    app.button(key="analyze_photo").click().run()
    assert not app.exception
    receipt = app.session_state["receipt"]
    assert receipt["inference_status"] == ("available" if available else "unavailable")
    assert receipt["location"] == "Test intersection"
    assert calls == [(storage / f"{receipt['record_id']}.png", reports.ROOT / "models/classifier.pt")]
    assert any("Report saved" in message.value for message in app.success)
    if available:
        assert len(app.table) == 1
        assert list(app.table[0].value["Match Score"]) == ["60.0%", "20.0%", "10.0%", "10.0%"]
    else:
        assert not app.table
        assert any("Analysis unavailable" in message.value for message in app.warning)
    assert app.button(key="close_analysis_results").label == "Done"

    app.button(key="close_analysis_results").click().run()
    assert not app.exception
    assert not app.table
    assert not any(button.key == "close_analysis_results" for button in app.button)
    app.button(key="analyze_photo").click().run()
    assert not app.exception
    assert len(calls) == 1
    assert len(list(storage.glob("*.json"))) == 1
    app.button(key="close_analysis_results").click().run()
    app.button(key="view_receipt_results").click().run()
    assert not app.exception
    assert len(app.table) == (1 if available else 0)
    assert len(calls) == 1
    app.button(key="close_analysis_results").click().run()

    app.radio(key="workspace").set_value("Staff review").run()
    assert not app.exception
    assert app.metric[0].value == "1"
    app.radio(key="workspace").set_value("Report a condition").run()
    assert not app.exception
    assert app.session_state["receipt"] == receipt
    assert any("Report saved" in message.value for message in app.success)
    assert not any(button.key == "close_analysis_results" for button in app.button)
    assert len(calls) == 1
    assert len(list(storage.glob("*.json"))) == 1
    app.button(key="new_photo").click().run()
    assert not app.exception
    assert app.session_state["receipt"] is None
    assert app.session_state["upload_revision"] == 1
    assert app.text_input(key="report_location").value == ""
    assert not app.checkbox(key="report_consent").value
    assert len(list(storage.glob("*.json"))) == 1


@pytest.mark.parametrize("available", [True, False])
def test_staff_review_components_save_and_filter_reports(storage, available):
    """Review available and unavailable suggestions without changing original outputs."""
    image = io.BytesIO()
    Image.new("RGB", (32, 32)).save(image, format="PNG")

    def predict(*args):
        if not available:
            raise FileNotFoundError("fixture")
        return {
            "predicted_class": "light",
            "class_scores": {"clear": .1, "light": .6, "medium": .2, "plowed": .1},
            "model_version": "test-fixture", "label_mapping_version": "1", "uncertain": None}

    record = reports.create_report(
        {"image_base64": base64.b64encode(image.getvalue()).decode()}, predictor=predict)
    app = AppTest.from_file(Path("app.py").resolve(), default_timeout=30).run()
    app.radio(key="workspace").set_value("Staff review").run()
    assert not app.exception
    assert app.metric[0].value == "1"
    assert app.metric[1].value == "1"
    assert not app.table
    app.button(key=f"view_analysis_{record['record_id']}").click().run()
    assert not app.exception
    if available:
        assert len(app.table) == 1
    else:
        assert any("Analysis unavailable" in message.value for message in app.warning)
        assert not app.table
    app.button(key="close_analysis_results").click().run()
    assert not app.exception
    assert not app.table
    app.toggle[0].set_value(True).run()
    assert not app.exception
    assert any("not a detected road region" in message.value for message in app.caption)
    app.selectbox[1].set_value("medium")
    app.selectbox[2].set_value("assess-soon")
    app.selectbox[3].set_value("assess-plowing")
    app.text_area[0].set_value("Inspect road")
    app.button(key=f"save_review_{record['record_id']}").click().run()
    assert not app.exception
    saved = json.loads(reports.record_path(record["record_id"]).read_text())
    assert saved["staff_class"] == "medium"
    assert saved["priority"] == "assess-soon"
    assert saved["action"] == "assess-plowing"
    assert saved["staff_note"] == "Inspect road"
    for key in ("predicted_class", "class_scores", "model_version", "inference_status"):
        assert saved[key] == record[key]
    assert app.metric[1].value == "0"
    assert any("queue is up to date" in message.value for message in app.success)
    app.button(key="refresh_reports").click().run()
    assert not app.exception
    assert any("No reports" in message.value for message in app.info)
    app.segmented_control[0].set_value("Reviewed").run()
    assert not app.exception
    assert app.selectbox[0].value == record["record_id"]
    app.segmented_control[0].set_value("All").run()
    assert not app.exception
    assert app.selectbox[0].value == record["record_id"]


@pytest.mark.parametrize("size", [(1600, 900), (900, 1600), (32, 32)])
def test_photo_previews_stay_compact_and_preserve_original(size):
    """Bound portrait and landscape previews without changing analysis input or upscaling."""
    from components.photo import preview_image

    image = io.BytesIO()
    Image.new("RGB", size).save(image, format="PNG")
    original = image.getvalue()
    image.seek(0)
    preview = preview_image(image)
    assert preview.height <= 240
    assert preview.width <= 560
    assert preview.width <= size[0] and preview.height <= size[1]
    assert preview.width / preview.height == pytest.approx(size[0] / size[1], abs=.01)
    assert image.getvalue() == original
    assert image.tell() == 0


def test_classifier_component_refreshes_cache_and_rejects_smoke_models(tmp_path, monkeypatch):
    """Reuse unchanged checkpoints, reload changed files, and reject smoke-test models."""
    import inference
    from components.classifier import cached_checkpoint, classify_photo

    model_path = tmp_path / "classifier.pt"
    model_path.write_bytes(b"fixture")
    image_path = tmp_path / "photo.png"
    loads = []
    predictions = []
    smoke_test = False

    def load_checkpoint(path):
        loaded = (object(), {"smoke_test": smoke_test})
        loads.append(loaded)
        return loaded

    def predict(image, checkpoint, loaded_checkpoint):
        predictions.append((image, checkpoint, loaded_checkpoint))
        return {"predicted_class": "light"}

    monkeypatch.setattr(inference, "load_checkpoint", load_checkpoint)
    monkeypatch.setattr(inference, "predict", predict)
    cached_checkpoint.clear()
    try:
        assert classify_photo(image_path, model_path) == {"predicted_class": "light"}
        classify_photo(image_path, model_path)
        assert len(loads) == 1
        assert predictions == [(image_path, model_path, loads[0])] * 2
        model_path.write_bytes(b"changed fixture")
        classify_photo(image_path, model_path)
        assert len(loads) == 2
        assert predictions[-1][2] is loads[1]
        smoke_test = True
        model_path.write_bytes(b"smoke-test fixture")
        with pytest.raises(ValueError, match="Smoke-test"):
            classify_photo(image_path, model_path)
        assert len(loads) == 3
        assert len(predictions) == 3
    finally:
        cached_checkpoint.clear()
