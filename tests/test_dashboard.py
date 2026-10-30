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


@pytest.mark.parametrize("scores, mixed", [
    ((.494, .470, .021, .015), True),
    ((.699, .101, .1, .1), True),
    ((.70, .15, .1, .05), False),
    ((.85, .05, .05, .05), False),
    ((.5, .5, 0, 0), True),
    ((.70, .550001, 0, 0), True),
    ((.70, .55, 0, 0), False),
])
def test_mixed_result_display_thresholds(scores, mixed):
    """Check approved display boundaries, including the gap rule independently of the top score."""
    from components.suggestion import mixed_result

    # Synthetic gap-boundary inputs isolate the rule even though normalized scores cannot reach it at 70%
    record = {"class_scores": dict(zip(("plowed", "medium", "clear", "light"), scores))}
    assert mixed_result(record) is mixed


@pytest.mark.parametrize("mixed, status", [(True, "pending"), (True, "confirmed"), (False, "pending")])
def test_analysis_dialog_explains_mixed_results_without_changing_record(mixed, status):
    """Show provisional uncertainty and truthful review wording while preserving saved model output."""
    record = {"record_id": "example-report", "inference_status": "available",
              "predicted_class": "plowed", "review_status": status, "uncertain": None,
              "class_scores": {"plowed": .494, "medium": .470, "clear": .021, "light": .015}
              if mixed else {"plowed": .85, "medium": .05, "clear": .05, "light": .05}}
    script = ("from components.suggestion import show_results\n"
              f"record = {record!r}\n"
              "show_results(record)\n"
              f"assert record == {record!r}\n")
    app = AppTest.from_string(script).run()
    assert not app.exception
    captions = [item.value for item in app.caption]
    if mixed:
        note = "Saved for staff review." if status == "pending" else "Staff review recorded."
        assert f"Results are mixed. {note}" in captions
    else:
        assert "*AI assessment*" in captions
        assert not any("Results are mixed" in item for item in captions)
    assert len(app.table) == 1
    assert not app.warning


@pytest.mark.parametrize("possible, expected", [(True, "Possible Plowed Surface"), (False, "Plowed Surface")])
def test_condition_heading_preserves_clearer_result_title(monkeypatch, possible, expected):
    """Keep the existing condition title unless the display explicitly marks it as possible."""
    from components import suggestion

    rendered = []
    monkeypatch.setattr(suggestion.st, "html", rendered.append)
    suggestion.render_condition_title("plowed", "DETECTED CONDITION", possible=possible)
    assert f'<h3 class="snowy-condition-title">{expected}</h3>' in rendered[0]


@pytest.mark.parametrize("zone, timestamp, expected", [
    ("America/Vancouver", "2026-10-29T16:10:00Z", "Oct 29, 2026 · 09:10 AM PDT"),
    ("America/New_York", "2026-10-29T16:10:00Z", "Oct 29, 2026 · 12:10 PM EDT"),
    ("America/New_York", "2026-12-29T16:10:00Z", "Dec 29, 2026 · 11:10 AM EST"),
    ("Asia/Kolkata", "2026-10-29T16:10:00Z", "Oct 29, 2026 · 09:40 PM IST"),
    ("Asia/Tokyo", "2026-10-29T16:10:00Z", "Oct 30, 2026 · 01:10 AM JST"),
    (None, "2026-10-29T16:10:00Z", "Oct 29, 2026 · 04:10 PM UTC"),
    ("Invalid/Zone", "2026-10-29T16:10:00Z", "Oct 29, 2026 · 04:10 PM UTC"),
    ("Asia/Tokyo", "2026-10-29T16:10:00", "Time unavailable"),
    ("Asia/Tokyo", "invalid", "Time unavailable"),
    ("Asia/Tokyo", None, "Time unavailable"),
])
def test_report_time_uses_viewer_timezone(monkeypatch, zone, timestamp, expected):
    """Verify browser timezones, daylight saving, date rollover, and safe fallbacks."""
    from types import SimpleNamespace
    from components import report_history

    monkeypatch.setattr(report_history.st, "context", SimpleNamespace(timezone=zone))
    assert report_history.report_time(timestamp) == expected


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
    assert app.checkbox(key="report_consent").label == "I agree to terms and conditions"
    assert any("Please agree to terms and conditions" in message.value for message in app.error)
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
    assert receipt["record_id"] in app.session_state["my_report_ids"]
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
    if available:
        app.button(key=f"correct_review_{record['record_id']}").click().run()
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


def test_priority_situations_use_reviewed_staff_priorities():
    """Exclude pending and unassigned reports and sort staff priorities before recency."""
    from components.staff_review import priority_situations

    rows = [
        {"record_id": "routine", "review_status": "confirmed", "priority": "routine", "reviewed_at": "2026-10-29"},
        {"record_id": "pending", "review_status": "pending", "priority": "urgent-assessment", "reviewed_at": None},
        {"record_id": "urgent-old", "review_status": "corrected", "priority": "urgent-assessment", "reviewed_at": "2026-10-27"},
        {"record_id": "unassigned", "review_status": "confirmed", "priority": "unassigned", "reviewed_at": "2026-10-29"},
        {"record_id": "soon", "review_status": "inconclusive", "priority": "assess-soon", "reviewed_at": "2026-10-29"},
        {"record_id": "urgent-new", "review_status": "confirmed", "priority": "urgent-assessment", "reviewed_at": "2026-10-28"},
    ]
    assert [row["record_id"] for row in priority_situations(rows)] == [
        "urgent-new", "urgent-old", "soon", "routine"]
    assert priority_situations([]) == []


def test_priority_panel_updates_and_reopens_saved_assessment(storage):
    """Show a saved priority even in an empty pending queue and reopen it from the panel."""
    image = io.BytesIO()
    Image.new("RGB", (32, 32)).save(image, format="PNG")

    def unavailable(*args):
        """Keep this interface test independent of the model artifact."""
        raise FileNotFoundError("fixture")

    record = reports.create_report(
        {"image_base64": base64.b64encode(image.getvalue()).decode(), "location": "Test Street"},
        predictor=unavailable)
    app = AppTest.from_file(Path("app.py").resolve(), default_timeout=30).run()
    app.radio(key="workspace").set_value("Staff review").run()
    assert not app.exception
    assert any("No reviewed reports" in item.value for item in app.info)
    app.selectbox[1].set_value("medium")
    app.selectbox[2].set_value("urgent-assessment")
    app.selectbox[3].set_value("recommend-plowing")
    app.button(key=f"save_review_{record['record_id']}").click().run()
    assert not app.exception
    assert any("No reports in this view" in item.value for item in app.info)
    assert app.button(key=f"priority_{record['record_id']}")
    assert any(message.value == "Plowing recommended" for message in app.error)
    app.button(key=f"notify_crew_{record['record_id']}").click().run()
    assert not app.exception
    assert any("No message has been sent" in message.value for message in app.info)
    assert any("Snow Plow Crew notified for the next available snow plow." == message.value
               for message in app.success)
    app.button(key="close_crew_notification").click().run()
    assert not app.exception
    assert not any("Snow Plow Crew notified" in message.value for message in app.success)
    saved = json.loads(reports.record_path(record["record_id"]).read_text())
    assert saved["action"] == "recommend-plowing"
    assert "dispatch_status" not in saved
    app.button(key=f"priority_{record['record_id']}").click().run()
    assert not app.exception
    assert app.segmented_control[0].value == "Reviewed"
    assert app.selectbox(key="selected_report").value == record["record_id"]
    assert app.selectbox[2].value == "urgent-assessment"

    app.selectbox[3].set_value("monitor")
    app.text_area[0].set_value("Continue monitoring after the latest inspection")
    app.button(key=f"save_review_{record['record_id']}").click().run()
    assert not app.exception
    assert not app.error
    assert not any(button.key == f"notify_crew_{record['record_id']}" for button in app.button)

    (storage / f"{record['record_id']}.png").unlink()
    app.run()
    assert not app.exception
    assert any("Photo preview unavailable" in item.value for item in app.caption)


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


def test_report_activity_keeps_actual_times_and_reassessments(storage, monkeypatch):
    """Keep upload, AI completion, and repeated staff decisions at their own saved times."""
    times = iter(["2026-10-29T16:10:00+00:00", "2026-10-29T16:11:00+00:00",
                  "2026-10-29T17:05:00+00:00", "2026-10-29T17:12:00+00:00"])
    monkeypatch.setattr(reports, "timestamp", lambda: next(times))
    image = io.BytesIO()
    Image.new("RGB", (32, 32)).save(image, format="PNG")

    def predict(*args):
        """Provide a condition suggestion without invoking the trained model."""
        return {"predicted_class": "light", "class_scores": None, "model_version": "fixture",
                "label_mapping_version": "1", "uncertain": None}

    record = reports.create_report({"image_base64": base64.b64encode(image.getvalue()).decode()},
                                   predictor=predict)
    identifier = record["record_id"]
    for action in ["recommend-plowing", "monitor"]:
        record = reports.review_report(identifier, {"staff_class": "medium", "priority": "assess-soon",
                                                    "action": action, "staff_note": action})
    saved = json.loads(reports.record_path(identifier).read_text())
    assert saved["uploaded_at"] == "2026-10-29T16:10:00+00:00"
    assert [event["at"] for event in saved["activity"]] == [
        "2026-10-29T16:10:00+00:00", "2026-10-29T16:11:00+00:00",
        "2026-10-29T17:05:00+00:00", "2026-10-29T17:12:00+00:00"]
    assert [event["action"] for event in saved["activity"] if event["kind"] == "staff-review"] == [
        "recommend-plowing", "monitor"]
    assert saved["reviewed_at"] == saved["activity"][-1]["at"]
    assert saved["predicted_class"] == "light"
    legacy = dict(saved)
    del legacy["activity"]
    assert [event["kind"] for event in reports.report_activity(legacy)] == ["received", "staff-review"]


def test_citizen_history_matches_staff_activity_and_excludes_other_reports(storage):
    """Refresh current staff decisions in citizen history and share the same activity display."""
    identifier = str(uuid4())
    other = str(uuid4())
    record = {"record_id": identifier, "uploaded_at": "2026-10-29T16:10:00+00:00",
              "location": "Oak Street", "review_status": "pending", "staff_note": "",
              "reviewed_at": None, "action": "undecided", "predicted_class": None,
              "inference_status": "unavailable", "captured_at": None, "priority": "unassigned",
              "staff_class": None}
    reports.save_record(record)
    reports.save_record(dict(record, record_id=other, location="Other person's report"))
    app = AppTest.from_file(Path("app.py").resolve(), default_timeout=30).run()
    app.radio(key="workspace").set_value("My reports").run()
    assert any("No reports yet" in item.value for item in app.info)
    app.session_state["my_report_ids"] = [identifier]
    app.run()
    assert not app.exception
    assert any(identifier in item.value for item in app.subheader)
    assert not any(other in item.value for item in app.subheader)
    assert any("04:10 PM UTC — Report received" in item.value for item in app.text)
    reports.review_report(identifier, {"staff_class": "medium", "priority": "assess-soon",
                                      "action": "recommend-plowing", "staff_note": "Assess next route"})
    app.button(key="refresh_my_reports").click().run()
    assert not app.exception
    assert any("Reviewed : Plowing recommended" in item.value for item in app.markdown)
    assert any("Assess next route" in item.value for item in app.text)
    citizen_activity = [item.value for item in app.text if " — " in item.value]
    assert any("Staff reviewed the photo" in item for item in citizen_activity)
    assert any("retained photo is unavailable" in item.value for item in app.warning)
    app.radio(key="workspace").set_value("Staff review").run()
    app.segmented_control[0].set_value("Reviewed").run()
    assert not app.exception
    staff_activity = [item.value for item in app.text if " — " in item.value]
    assert citizen_activity == staff_activity


def workflow_report(location="Oak Street"):
    """Build a legacy-compatible pending record for isolated workflow tests."""
    return {"record_id": str(uuid4()), "uploaded_at": "2026-10-29T16:10:00+00:00",
            "location": location, "review_status": "pending", "staff_note": "", "reviewed_at": None,
            "action": "undecided", "predicted_class": None, "inference_status": "unavailable",
            "captured_at": None, "priority": "unassigned", "staff_class": None}


@pytest.mark.parametrize("action", ["monitor", "reassess", "resolved", "plowing-completed"])
def test_outcomes_require_reason_without_overwriting_report(storage, action):
    """Reject missing reasons and retain earlier recommendations when staff save an outcome."""
    from components.staff_review import matches_filter, priority_situations

    record = workflow_report()
    reports.save_record(record)
    payload = {"staff_class": "medium", "priority": "assess-soon", "action": "recommend-plowing",
               "staff_note": "Plowing recommended after inspection"}
    reports.review_report(record["record_id"], payload)
    path = reports.record_path(record["record_id"])
    before = path.read_text()
    with pytest.raises(ValueError, match="short reason"):
        reports.review_report(record["record_id"], dict(payload, action=action, staff_note="  "))
    assert path.read_text() == before
    updated = reports.review_report(record["record_id"], dict(payload, action=action, staff_note="Staff verified outcome"))
    assert updated["activity"][-2]["action"] == "recommend-plowing"
    assert updated["activity"][-1]["action"] == action
    assert updated["activity"][-1]["note"] == "Staff verified outcome"
    assert matches_filter(updated, "Closed") == (action in reports.CLOSED_ACTIONS)
    assert matches_filter(updated, "Reassessment") == (action == "reassess")
    assert bool(priority_situations([updated])) == (action not in reports.CLOSED_ACTIONS)
    reopened = reports.review_report(record["record_id"], dict(payload, action="reassess", staff_note="New snow reported"))
    assert matches_filter(reopened, "Reassessment")
    assert not matches_filter(reopened, "Closed")


def test_location_grouping_preserves_separate_reports():
    """Group only matching location text, leaving unknown locations and abbreviations separate."""
    from components.staff_review import reports_at_location

    selected = workflow_report()
    related = workflow_report("  OAK   street ")
    rows = [selected, related, workflow_report("Oak St"), workflow_report(None), workflow_report("Pine Street")]
    assert reports_at_location(rows, selected) == [related]
    assert reports_at_location(rows, workflow_report(None)) == []


def test_staff_outcomes_and_related_navigation_match_citizen_history(storage):
    """Save a reassessment, open a related record, and display confirmed completion to its citizen."""
    first = workflow_report()
    second = workflow_report(" oak  STREET ")
    reports.save_record(first)
    reports.save_record(second)
    app = AppTest.from_file(Path("app.py").resolve(), default_timeout=30).run()
    app.session_state["receipt"] = first
    app.session_state["my_report_ids"] = [first["record_id"], second["record_id"]]
    app.run()
    app.button(key="view_my_report").click().run()
    assert not app.exception
    assert app.radio(key="workspace").value == "My reports"
    assert len(app.subheader) == 1
    assert first["record_id"] in app.subheader[0].value
    app.button(key="show_all_my_reports").click().run()
    assert len(app.subheader) == 2
    app.radio(key="workspace").set_value("Staff review").run()
    app.selectbox(key="selected_report").set_value(first["record_id"]).run()
    app.selectbox[1].set_value("medium")
    app.selectbox[3].set_value("reassess")
    app.button(key=f"save_review_{first['record_id']}").click().run()
    assert any("short reason" in item.value for item in app.error)
    app.text_area[0].set_value("Need a clearer current observation")
    app.button(key=f"save_review_{first['record_id']}").click().run()
    app.segmented_control[0].set_value("Reassessment").run()
    assert not app.exception
    assert app.selectbox(key="selected_report").value == first["record_id"]
    assert any("Needs reassessment" in item.value for item in app.warning)
    assert any(item.label == "Needs Reassessment" and item.value == "1" for item in app.metric)
    app.button(key=f"related_{second['record_id']}").click().run()
    assert app.selectbox(key="selected_report").value == second["record_id"]
    app.selectbox[1].set_value("plowed")
    app.selectbox[3].set_value("plowing-completed")
    app.text_area[0].set_value("Crew confirmed work completed")
    app.button(key=f"save_review_{second['record_id']}").click().run()
    app.segmented_control[0].set_value("Closed").run()
    assert not app.exception
    assert app.selectbox(key="selected_report").value == second["record_id"]
    assert json.loads(reports.record_path(first["record_id"]).read_text())["action"] == "reassess"
    app.radio(key="workspace").set_value("My reports").run()
    assert not app.exception
    assert any("Closed : Plowing completed" in item.value for item in app.markdown)
    assert any("Plowing completion confirmed by staff" in item.value for item in app.markdown)
    assert any("Crew confirmed work completed" in item.value for item in app.text)


def test_staff_quick_confirmation_preserves_decisions_and_advances(storage):
    """Confirm suggestions in All without losing saved decisions or model evidence."""
    image = io.BytesIO()
    Image.new("RGB", (32, 32)).save(image, format="PNG")

    def predict(*args):
        """Provide a deterministic suggestion without loading the model."""
        return {"predicted_class": "light", "class_scores": {"light": .8},
                "model_version": "fixture", "label_mapping_version": "1", "uncertain": None}

    payload = {"image_base64": base64.b64encode(image.getvalue()).decode()}
    first = reports.create_report(payload, predictor=predict)
    second = reports.create_report(payload, predictor=predict)
    reports.review_report(first["record_id"], {"staff_class": "medium", "priority": "assess-soon",
                                             "action": "monitor", "staff_note": "Check again later"})
    app = AppTest.from_file(Path("app.py").resolve(), default_timeout=30).run()
    app.radio(key="workspace").set_value("Staff review").run()
    assert not app.exception
    assert not app.text_area
    app.segmented_control[0].set_value("All").run()
    app.selectbox(key="selected_report").set_value(first["record_id"]).run()
    app.button(key=f"confirm_review_{first['record_id']}").click().run()
    assert not app.exception
    assert app.selectbox(key="selected_report").value == second["record_id"]
    saved = json.loads(reports.record_path(first["record_id"]).read_text())
    assert saved["review_status"] == "confirmed"
    assert saved["staff_class"] == "light"
    assert saved["priority"] == "assess-soon"
    assert saved["action"] == "monitor"
    assert saved["staff_note"] == "Check again later"
    for key in ("predicted_class", "class_scores", "model_version"):
        assert saved[key] == first[key]
    app.button(key=f"confirm_review_{second['record_id']}").click().run()
    assert not app.exception
    assert json.loads(reports.record_path(second["record_id"]).read_text())["review_status"] == "confirmed"
