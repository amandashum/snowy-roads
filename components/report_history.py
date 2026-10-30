"""Display this browser session's saved reports and latest staff decisions."""

import json
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import streamlit as st

import reports
from components.layout import render_page_heading
from components.photo import render_photo
from components.labels import DISPLAY


DECISIONS = {"undecided": "Next action pending", "monitor": "Monitoring conditions",
             "assess-plowing": "Further assessment needed", "recommend-plowing": "Plowing recommended",
             "no-action": "No action recommended", "reassess": "Needs reassessment",
             "resolved": "Resolved", "plowing-completed": "Plowing completed"}


def open_my_report(identifier):
    """Navigate to the submitted report before the workspace widget renders."""
    st.session_state.workspace = "My reports"
    st.session_state.focused_report = identifier


def report_time(value):
    """Format a saved timestamp in the viewer's browser timezone, falling back to UTC."""
    if not value:
        return "Time unavailable"
    try:
        saved = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if saved.tzinfo is None:
            return "Time unavailable"
        try:
            viewer_zone = ZoneInfo(st.context.timezone or "UTC")
        except (ZoneInfoNotFoundError, ValueError):
            viewer_zone = ZoneInfo("UTC")
        return saved.astimezone(viewer_zone).strftime("%b %d, %Y · %I:%M %p %Z")
    except (ValueError, TypeError):
        return "Time unavailable"


def render_report_history():
    """Read fresh staff decisions for session-owned reports without exposing other uploads."""
    render_page_heading("My Reports", "View your photos and the latest staff decisions", "Snowy Roads")
    st.caption("Reports submitted in this browser session")
    if st.button("Refresh status", key="refresh_my_reports", icon=":material/refresh:"):
        st.rerun()
    identifiers = list(st.session_state.get("my_report_ids", []))
    receipt = st.session_state.get("receipt")
    if receipt and receipt["record_id"] not in identifiers:
        identifiers.append(receipt["record_id"])
    if not identifiers:
        st.info("No reports yet. Submit a road photo to see it here.")
        return
    focused = st.session_state.get("focused_report")
    if focused in identifiers:
        if st.button("Show all my reports", key="show_all_my_reports"):
            st.session_state.pop("focused_report", None)
            st.rerun()
        identifiers = [focused]
    for identifier in reversed(identifiers):
        try:
            record = json.loads(reports.record_path(identifier).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            st.warning(f"Report #{identifier} is unavailable.")
            continue
        with st.container(border=True, key=f"my_report_{identifier}"):
            st.subheader(f"Report #{identifier}")
            st.text(record.get("location") or "Location not provided")
            reviewed = record.get("review_status") != "pending"
            decision = DECISIONS.get(record.get("action"), "Next action pending")
            status = "Closed" if record.get("action") in reports.CLOSED_ACTIONS else "Reviewed"
            st.markdown(f"**{status} : {decision}**" if reviewed else "**Received : Awaiting staff review**")
            st.text(f'Staff note: “{record["staff_note"]}”' if record.get("staff_note")
                    else "Staff note: No note yet.")
            render_activity(record)
            if record.get("action") == "plowing-completed":
                st.markdown("**Plowing completion confirmed by staff.**")
            elif record.get("action") != "resolved":
                st.markdown("**No service time confirmed.**")
            with st.expander("View photo"):
                image = reports.UPLOADS / f"{identifier}.png"
                if image.exists():
                    render_photo(image, f"history_{identifier}", max_height=260)
                else:
                    st.warning("The retained photo is unavailable.")


def render_activity(record):
    """Show the same timestamped facts to citizens and staff without inventing dispatch events."""
    for event in reports.report_activity(record):
        kind = event["kind"]
        if kind == "received":
            message = "Report received"
        elif kind == "ai-analysis":
            message = f'AI analysis suggested {DISPLAY.get(event.get("condition"), "an unknown condition")}'
        elif kind == "analysis-unavailable":
            message = "AI analysis unavailable; awaiting staff review"
        elif kind == "staff-review":
            decision = DECISIONS.get(event.get("action"), "Next action pending")
            condition = DISPLAY.get(event.get("condition"), "Condition not recorded")
            message = f"Staff reviewed the photo — {condition} · {decision}"
        else:
            continue
        st.text(f'{report_time(event.get("at"))} — {message}')
        if kind == "staff-review" and event.get("note"):
            st.text(f'Reason: {event["note"]}')
