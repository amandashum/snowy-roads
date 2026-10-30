"""Check optional capture-time entry and timezone conversion."""

import io
from datetime import date, datetime, time, timezone
from pathlib import Path

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from components.capture_time import capture_choices


@pytest.mark.parametrize("day, clock, zone, offsets", [
    (date(2026, 1, 15), time(14, 30), "America/St_Johns", [-3.5]),
    (date(2026, 10, 30), time(14, 30), "America/St_Johns", [-2.5]),
    (date(2026, 10, 30), time(14, 30), "UTC", [0]),
    (date(2026, 3, 8), time(2, 30), "America/St_Johns", []),
    (date(2026, 11, 1), time(1, 30), "America/St_Johns", [-2.5, -3.5]),
    (None, time(14, 30), "UTC", []),
])
def test_capture_time_respects_timezone_and_clock_changes(day, clock, zone, offsets):
    choices = capture_choices(day, clock, zone)
    assert [candidate.utcoffset().total_seconds() / 3600 for candidate in choices] == offsets


def test_optional_time_uses_today_in_selected_zone_and_allows_blank(tmp_path, monkeypatch):
    import streamlit as st
    import reports
    from components import report_form
    from components import capture_time

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 10, 31, 1, tzinfo=timezone.utc).astimezone(tz)

    monkeypatch.setattr(capture_time, "datetime", FixedDatetime)

    monkeypatch.setattr(reports, "RECORDS", tmp_path)
    monkeypatch.setattr(reports, "UPLOADS", tmp_path)
    image = io.BytesIO()
    Image.new("RGB", (32, 32)).save(image, format="PNG")
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: image)

    def unavailable(*args):
        raise FileNotFoundError("fixture")

    monkeypatch.setattr(report_form, "classify_photo", unavailable)
    app = AppTest.from_file(Path("app.py").resolve(), default_timeout=30).run()
    assert not app.exception
    assert len(app.checkbox) == 1  # Only submission consent remains.
    assert not app.date_input
    assert app.time_input(key="report_capture_clock").value is None
    assert app.selectbox(key="report_capture_zone").value is None
    assert "Newfoundland · America/St_Johns" in app.selectbox(key="report_capture_zone").options
    app.checkbox(key="report_consent").check()
    app.button(key="analyze_photo").click().run()
    assert not app.exception
    assert app.session_state["receipt"]["captured_at"] is None
    app.button(key="close_analysis_results").click().run()
    app.time_input(key="report_capture_clock").set_value(time(9, 15))
    app.selectbox(key="report_capture_zone").set_value("America/St_Johns")
    app.button(key="analyze_photo").click().run()
    assert not app.exception
    assert app.session_state["receipt"]["captured_at"] == "2026-10-30T09:15:00-02:30"
    app.button(key="close_analysis_results").click().run()
    app.time_input(key="report_capture_clock").set_value(None).run()
    app.button(key="analyze_photo").click().run()
    assert not app.exception
    assert app.session_state["receipt"]["captured_at"] is None


def test_blank_timezone_uses_browser_timezone_without_requiring_a_selection(monkeypatch):
    from types import SimpleNamespace
    import streamlit as st
    from components import capture_time

    monkeypatch.setattr(st, "context", SimpleNamespace(timezone="America/St_Johns"))
    script = ("from components.capture_time import render_capture_time\n"
              "import streamlit as st\n"
              "captured, error = render_capture_time()\n"
              "st.session_state.capture_result = captured\n"
              "st.session_state.capture_error = error\n")
    app = AppTest.from_string(script, default_timeout=30).run()
    assert app.selectbox(key="report_capture_zone").value is None
    app.time_input(key="report_capture_clock").set_value(time(14, 30)).run()
    assert not app.exception
    assert app.selectbox(key="report_capture_zone").value is None
    assert app.session_state["capture_error"] is None
    result = datetime.fromisoformat(app.session_state["capture_result"])
    assert result.time() == time(14, 30)
    assert result.utcoffset() == capture_time.datetime.now(capture_time.ZoneInfo("America/St_Johns")).utcoffset()
