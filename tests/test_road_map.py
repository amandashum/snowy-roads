"""Verify Road Map street matching, marker grouping, and workspace rendering."""

import json
from pathlib import Path
from uuid import uuid4

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

import reports
from components import road_map


@pytest.fixture
def storage(monkeypatch):
    """Isolate fixture photos and reports from real local submissions."""
    folder = Path("models") / f"road-map-test-{uuid4().hex}"
    folder.mkdir(parents=True)
    monkeypatch.setattr(reports, "RECORDS", folder)
    monkeypatch.setattr(reports, "UPLOADS", folder)
    yield folder
    for path in folder.iterdir():
        path.unlink()
    folder.rmdir()


def record(location, uploaded_at, **changes):
    """Build a minimal saved-report record."""
    base = {"record_id": str(uuid4()), "uploaded_at": uploaded_at, "captured_at": None, "location": location,
            "predicted_class": "light", "review_status": "pending", "staff_class": None,
            "action": "undecided", "unsuitable": False}
    base.update(changes)
    return base


@pytest.mark.parametrize("location, expected", [
    ("Water Street", "Water Street"),
    ("near 123 water st, St. John's", "Water Street"),
    ("Corner of New Gower St and Springdale", "New Gower Street"),
    ("Gower Street by the Basilica", "Gower Street"),
    ("Longs Hill", "Long's Hill"),
    ("Old Pennywell Rd", "Old Pennywell Road"),
    ("The Boulevard", "The Boulevard"),
    ("Waterford", None),
    ("Watr Stret", "Water Street"),
    ("Duckwrth Street", "Duckworth Street"),
    ("Elizebeth Avenue", "Elizabeth Avenue"),
    ("Topsial Road", "Topsail Road"),
    ("Kenmount Road near Avalon Mall", "Kenmount Road"),
    ("Signal Hill trail", None),
    ("", None),
    (None, None),
])
def test_match_street(location, expected):
    street = road_map.match_street(location)
    assert (street["name"] if street else None) == expected


def test_street_names_are_sorted_official_names():
    names = road_map.street_names()
    assert names == sorted(names) and len(names) == len(set(names))
    assert "Water Street" in names and "Boulevard" not in names


def test_gazetteer_points_are_in_st_johns():
    gazetteer, _ = road_map.load_streets()
    names = [street["name"] for street in gazetteer["streets"]]
    assert len(names) == len(set(names))
    for street in gazetteer["streets"]:
        assert 47.45 < street["lat"] < 47.65 and -52.90 < street["lon"] < -52.65, street["name"]


def test_markers_group_by_street_and_keep_latest_three(storage):
    records = [record("Water Street", f"2026-10-30T1{hour}:00:00+00:00") for hour in range(5)]
    records[4]["record_id"] = newest = str(uuid4())
    Image.new("RGB", (64, 48), "white").save(storage / f"{newest}.png")
    records += [record("Duckworth St", "2026-10-30T09:00:00+00:00"), record("Somewhere else", "2026-10-30T09:00:00+00:00"),
                record("Water Street", "2026-10-30T19:00:00+00:00", unsuitable=True, review_status="inconclusive")]
    markers, unmatched = road_map.build_markers(records)
    assert unmatched == 1
    water = next(marker for marker in markers if marker["street"] == "Water Street")
    assert water["count"] == 5
    assert len(water["reports"]) == 3
    assert water["reports"][0]["thumbnail"].startswith("data:image/jpeg;base64,")
    assert water["reports"][1]["thumbnail"] is None
    assert [marker["street"] for marker in markers] == ["Duckworth Street", "Water Street"]


@pytest.mark.parametrize("changes, expected", [
    ({}, ("Suggested: Light-covered surface", "Awaiting staff review", "pending")),
    ({"predicted_class": None}, ("Analysis unavailable", "Awaiting staff review", "pending")),
    ({"review_status": "corrected", "staff_class": "medium", "action": "recommend-plowing"},
     ("Staff: Medium-to-heavy-covered surface", "Reviewed · Recommend plowing", "reviewed")),
    ({"review_status": "confirmed", "staff_class": "light", "action": "reassess"},
     ("Staff: Light-covered surface", "Needs reassessment", "reassess")),
    ({"review_status": "corrected", "staff_class": "plowed", "action": "plowing-completed"},
     ("Staff: Plowed surface", "Closed · Plowing completed", "closed")),
])
def test_report_status_text(changes, expected):
    assert road_map.report_status(record("Water Street", "2026-10-30T10:00:00+00:00", **changes)) == expected


def test_map_html_inlines_maplibre_and_escapes_data():
    html = road_map.map_html([{"street": "</script><b>", "lat": 47.56, "lon": -52.7, "count": 1, "reports": []}])
    assert "maplibregl" in html and "__MAP" not in html and "__MAPLIBRE" not in html
    assert "</script><b>" not in html


def test_road_map_workspace_renders(storage):
    (storage / f"{uuid4()}.json").write_text(json.dumps(record("Water Street", "2026-10-30T10:00:00+00:00")))
    app = AppTest.from_file(Path("app.py").resolve(), default_timeout=60).run()
    app.radio(key="workspace").set_value("Road map").run()
    assert not app.exception
    assert any("Road Map" == title.value for title in app.title)
    assert not any("No reports name" in message.value for message in app.info)
