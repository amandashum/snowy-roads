"""Render the staff queue, retained observations, and separate review decisions."""

import json

import streamlit as st

import reports
from components.labels import ACTIONS, DISPLAY, PRIORITIES
from components.suggestion import assessment_flag, show_suggestion


def load_reports():
    """Read reports newest first and warn about unreadable records."""
    records = []
    for path in reports.RECORDS.glob("*.json"):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            st.warning(f"A saved report could not be read: {path.name}")
    return sorted(records, key=lambda row: row["uploaded_at"], reverse=True)


def render_observation(record):
    """Render the retained photo, optional analyzed crop, and original suggestion."""
    selected = record["record_id"]
    image = reports.UPLOADS / f"{selected}.png"
    if image.exists():
        st.image(str(image), alt="Original road observation retained for staff review", width="stretch")
        if st.toggle("Show analyzed center crop", key=f"crop_{selected}"):
            from PIL import Image
            from torchvision.transforms.functional import center_crop, resize
            with Image.open(image) as original:
                crop = center_crop(resize(original.convert("RGB"), 256), 224)
            st.image(crop, alt="Center region used by the classifier after resizing", width=224)
            st.caption("The classifier uses this center crop; it is not a detected road region.")
    else:
        st.warning("The retained photo is unavailable.")
    st.write(f"Location: {record['location'] or 'Unknown'}")
    st.caption(f"Captured: {record['captured_at'] or 'Unknown'}")
    st.caption(f"Uploaded: {record['uploaded_at']} · Review status: {record['review_status']}")
    show_suggestion(record)


def render_review_form(record):
    """Render and save staff decisions without replacing model outputs."""
    selected = record["record_id"]
    st.subheader("Staff decision")
    with st.form(f"review_{selected}"):
        choices = list(DISPLAY)
        current = record["staff_class"] or record["predicted_class"]
        condition = st.selectbox("Reviewed condition", choices, index=choices.index(current) if current in choices else None,
                                 format_func=DISPLAY.get, placeholder="Select a reviewed condition")
        priority = st.selectbox("Priority", list(PRIORITIES), format_func=PRIORITIES.get,
                                index=list(PRIORITIES).index(record["priority"]))
        action = st.selectbox("Action recommendation", list(ACTIONS), format_func=ACTIONS.get,
                              index=list(ACTIONS).index(record["action"]))
        note = st.text_area("Review notes", value=record["staff_note"], max_chars=2000)
        st.caption("Saving records a staff assessment. It does not contact crews or dispatch a snowplow.")
        save = st.form_submit_button("Save staff review", type="primary", width="stretch")
    if save:
        try:
            reports.review_report(selected, {"staff_class": condition, "priority": priority,
                                            "action": action, "staff_note": note})
            st.success("Staff review saved. Refresh reports to update the queue.")
        except (ValueError, OSError) as error:
            st.error(f"The review could not be saved: {error}")


def render_staff_review():
    """Render the filtered staff queue and the selected report's review controls."""
    st.title("Staff review")
    st.write("Confirm road conditions and record the next action. Dispatch remains a staff decision.")
    records = load_reports()
    with st.container(horizontal=True):
        st.metric("Reports", len(records))
        st.metric("Awaiting review", sum(row["review_status"] == "pending" for row in records))
        st.metric("Pending snow assessments", sum(row["review_status"] == "pending" and
                  row["predicted_class"] in {"light", "medium"} for row in records))
    st.divider()
    filter_name = st.segmented_control("Show reports", ["Pending", "All", "Reviewed"], default="Pending")
    filtered = [row for row in records if filter_name == "All" or
                (row["review_status"] == "pending") == (filter_name == "Pending")]
    if st.button("Refresh reports", icon=":material/refresh:"):
        st.rerun()
    if not filtered:
        st.info("No reports in this view. Submit a road photo or choose another filter.")
        return
    selected = st.selectbox("Choose a report", [row["record_id"] for row in filtered],
                            format_func=lambda identifier: next(
                                f"{row['record_id'][:8]} · {row['location'] or 'Unknown location'} · {assessment_flag(row)}"
                                for row in filtered if row["record_id"] == identifier))
    record = next(row for row in filtered if row["record_id"] == selected)
    left, right = st.columns([1.2, 1], gap="large")
    with left:
        render_observation(record)
    with right:
        render_review_form(record)
