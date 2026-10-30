"""Show saved reports on an embedded MapLibre map of St. John's streets.

Supplied location text is matched against a local street list, so report
details are never sent to a geocoding service. Only base-map tiles load from
OpenFreeMap. A marker represents a street, not the exact photo position.
"""

import base64
import io
import json
import re
from difflib import SequenceMatcher
from functools import lru_cache

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

import reports
from components.labels import ACTIONS, DISPLAY
from components.layout import render_page_heading
from components.report_history import report_time

POPUP_REPORTS = 3
FUZZY_THRESHOLD = 0.85
FUZZY_MIN_LENGTH = 8
ABBREVIATIONS = {"st": "street", "rd": "road", "ave": "avenue", "av": "avenue", "dr": "drive",
                 "hwy": "highway", "blvd": "boulevard", "ln": "lane", "pl": "place"}


def normalize(text):
    """Lowercase location text, drop punctuation, and expand common street abbreviations."""
    words = re.sub(r"[^a-z0-9]+", " ", text.lower().replace("'", "").replace("\u2019", "")).split()
    return " ".join(ABBREVIATIONS.get(word, word) for word in words)


@lru_cache(maxsize=1)
def load_streets():
    """Return the gazetteer and its normalized names, longest first so specific names win."""
    gazetteer = json.loads((reports.ROOT / "config" / "streets.json").read_text(encoding="utf-8"))
    names = []
    for street in gazetteer["streets"]:
        for name in [street["name"], *street.get("aliases", [])]:
            names.append((normalize(name), street))
    names.sort(key=lambda pair: len(pair[0]), reverse=True)
    return gazetteer, tuple(names)


def street_names():
    """Return the official street names, sorted for the location dropdown."""
    return sorted(street["name"] for street in load_streets()[0]["streets"])


def match_street(location):
    """Return the known St. John's street mentioned in the location, tolerating small typos, or None."""
    if not location:
        return None
    normalized = normalize(location)
    text = f" {normalized} "
    names = load_streets()[1]
    for name, street in names:
        if f" {name} " in text:
            return street
    words = normalized.split()
    best, best_score = None, FUZZY_THRESHOLD
    for name, street in names:
        size = len(name.split())
        if len(name) < FUZZY_MIN_LENGTH:
            continue
        for start in range(max(len(words) - size + 1, 0)):
            score = SequenceMatcher(None, name, " ".join(words[start:start + size])).ratio()
            if score > best_score:
                best, best_score = street, score
    return best


@lru_cache(maxsize=256)
def thumbnail(path, modified):
    """Encode a small JPEG preview; the modification time keys the cache."""
    del modified
    with Image.open(path) as photo:
        preview = photo.convert("RGB")
        preview.thumbnail((240, 180))
    buffer = io.BytesIO()
    preview.save(buffer, format="JPEG", quality=72)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def report_status(record):
    """Describe the visible condition and review status in text, never colour alone."""
    action = record.get("action")
    if record.get("review_status") == "pending":
        predicted = record.get("predicted_class")
        condition = f"Suggested: {DISPLAY[predicted]}" if predicted in DISPLAY else "Analysis unavailable"
        return condition, "Awaiting staff review", "pending"
    condition = f"Staff: {DISPLAY.get(record.get('staff_class'), 'Condition not recorded')}"
    if action in reports.CLOSED_ACTIONS:
        return condition, f"Closed · {ACTIONS[action]}", "closed"
    if action == "reassess":
        return condition, "Needs reassessment", "reassess"
    return condition, f"Reviewed · {ACTIONS.get(action, 'Undecided')}", "reviewed"


def load_records():
    """Read every saved report, skipping unreadable files."""
    records = []
    for path in sorted(reports.RECORDS.glob("*.json")):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return records


def build_markers(records):
    """Group reports by matched street and keep the newest uploads for each popup.

    Returns the marker list and the number of reports whose location did not
    match a known street. Photos staff marked unsuitable are left off the map.
    """
    grouped, unmatched = {}, 0
    for record in records:
        if record.get("unsuitable"):
            continue
        street = match_street(record.get("location"))
        if street is None:
            unmatched += 1
            continue
        grouped.setdefault(street["name"], (street, []))[1].append(record)
    markers = []
    for name, (street, matched) in sorted(grouped.items()):
        matched.sort(key=lambda record: record.get("uploaded_at") or "", reverse=True)
        items = []
        for record in matched[:POPUP_REPORTS]:
            condition, status, kind = report_status(record)
            photo = reports.UPLOADS / f"{record['record_id']}.png"
            try:
                preview = thumbnail(str(photo), photo.stat().st_mtime_ns)
            except OSError:
                preview = None
            captured = record.get("captured_at")
            items.append({"thumbnail": preview, "condition": condition, "status": status, "status_kind": kind,
                          "time_label": "Photo taken" if captured else "Uploaded",
                          "time": report_time(captured or record.get("uploaded_at"))})
        markers.append({"street": name, "lat": street["lat"], "lon": street["lon"],
                        "count": len(matched), "reports": items})
    return markers, unmatched


@lru_cache(maxsize=1)
def map_template():
    """Inline the vendored MapLibre assets so the map does not depend on a script CDN."""
    dashboard = reports.ROOT / "dashboard"
    template = (dashboard / "road_map.html").read_text(encoding="utf-8")
    return (template.replace("__MAPLIBRE_CSS__", (dashboard / "vendor" / "maplibre-gl.css").read_text(encoding="utf-8"))
            .replace("__MAPLIBRE_JS__", (dashboard / "vendor" / "maplibre-gl.js").read_text(encoding="utf-8")))


def map_html(markers):
    """Embed marker data as JSON that cannot close its script element early."""
    data = json.dumps({"center": load_streets()[0]["center"], "markers": markers}).replace("</", "<\\/")
    return map_template().replace("__MAP_DATA__", data)


def render_road_map():
    """Render the Road Map workspace."""
    render_page_heading("Road Map", "See where road photos were reported across St. John's", "Snowy Roads")
    if st.button("Refresh map", key="refresh_road_map", icon=":material/refresh:"):
        st.rerun()
    markers, unmatched = build_markers(load_records())
    with st.container(key="road_map_card"):
        components.html(map_html(markers), height=520)
        st.caption("Each camera marks a street named in a report's location, not the exact photo position. "
                   "Select a camera to see up to three of the newest photos and their review status. "
                   "Use the bottom-right controls to zoom or enlarge the map.")
        if unmatched:
            st.caption(f"{unmatched} report{'s' if unmatched != 1 else ''} with a missing or unrecognized "
                       "location are not shown on the map.")
        st.caption("Photos from all saved reports appear here. Map data \u00a9 OpenStreetMap contributors; "
                   "tiles by OpenFreeMap.")
    if not markers:
        st.info("No reports name a recognized St. John's street yet. Add a street such as "
                "\u201cWater Street\u201d to a report's location to place it on the map.")
