"""Render the staff queue, retained observations, and separate review decisions."""

import json

import streamlit as st

import reports
from components.labels import ACTIONS, DISPLAY, PRIORITIES
from components.layout import render_page_heading, render_section_heading
from components.photo import render_photo
from components.suggestion import assessment_flag, show_results


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
    render_section_heading("01", "Road Observation")
    st.subheader(record["location"] or "Unknown location")
    st.caption(f"REPORT {selected[:8].upper()} · {record['review_status'].replace('-', ' ').title()}")
    image = reports.UPLOADS / f"{selected}.png"
    if image.exists():
        render_photo(image, f"observation_{selected}", max_height=260)
        with st.expander("Image Details"):
            if st.toggle("Show analyzed center crop", key=f"crop_{selected}"):
                from PIL import Image
                from torchvision.transforms.functional import center_crop, resize
                with Image.open(image) as original:
                    crop = center_crop(resize(original.convert("RGB"), 256), 224)
                st.image(crop, alt="Center region used by the classifier after resizing", width=224)
                st.caption("The classifier uses this center crop; it is not a detected road region.")
    else:
        st.warning("The retained photo is unavailable.")
    st.caption(f"Captured: {record['captured_at'] or 'Unknown'}")
    with st.container(key=f"analysis_{selected}"):
        st.caption("PHOTO ANALYSIS")
        if record["inference_status"] == "available":
            st.markdown(f"**{DISPLAY[record['predicted_class']]}**")
        else:
            st.warning("Analysis unavailable. Review the photo manually.")
        if st.button("View Results", key=f"view_analysis_{selected}", icon=":material/bar_chart:", width="stretch"):
            show_results(record)


def render_review_form(record):
    """Render and save staff decisions without replacing model outputs."""
    selected = record["record_id"]
    render_section_heading("02", "Your Assessment")
    st.caption("Confirm the condition and record what happens next")
    with st.form(f"review_{selected}", border=False):
        choices = list(DISPLAY)
        current = record["staff_class"] or record["predicted_class"]
        condition = st.selectbox("Reviewed Condition", choices, index=choices.index(current) if current in choices else None,
                                 format_func=DISPLAY.get, placeholder="Select a reviewed condition")
        priority_column, action_column = st.columns(2)
        with priority_column:
            priority = st.selectbox("Priority", list(PRIORITIES), format_func=PRIORITIES.get,
                                    index=list(PRIORITIES).index(record["priority"]))
        with action_column:
            action = st.selectbox("Next Action", list(ACTIONS), format_func=ACTIONS.get,
                                  index=list(ACTIONS).index(record["action"]))
        note = st.text_area("Review Notes", value=record["staff_note"], max_chars=2000, height=120,
                            placeholder="Add context for the next person…")
        save = st.form_submit_button("Save Review", key=f"save_review_{selected}", type="primary", width="stretch")
    if save:
        try:
            reports.review_report(selected, {"staff_class": condition, "priority": priority,
                                            "action": action, "staff_note": note})
            st.session_state.review_saved = True
            st.rerun()
        except (ValueError, OSError) as error:
            st.error(f"The review could not be saved: {error}")


def render_staff_review():
    """Render the filtered staff queue and the selected report's review controls."""
    render_page_heading("Review Road Conditions", "Review a photo, confirm the condition, and choose the next step", "Snowy Roads")
    if st.session_state.pop("review_saved", False):
        st.success("Review saved. The queue is up to date.")
    records = load_reports()
    with st.container(horizontal=True, key="review_stats"):
        st.metric("Reports", len(records))
        st.metric("Awaiting Review", sum(row["review_status"] == "pending" for row in records))
        st.metric("Snow to Assess", sum(row["review_status"] == "pending" and
                  row["predicted_class"] in {"light", "medium"} for row in records))
    with st.container(border=True, key="queue_controls"):
        filters, refresh = st.columns([4, 1], vertical_alignment="bottom")
        with filters:
            filter_name = st.segmented_control("Show Reports", ["Pending", "All", "Reviewed"], default="Pending")
        with refresh:
            if st.button("Refresh", key="refresh_reports", icon=":material/refresh:", width="stretch"):
                st.rerun()
    filtered = [row for row in records if filter_name == "All" or
                (row["review_status"] == "pending") == (filter_name == "Pending")]
    if not filtered:
        st.info("No reports in this view. Submit a road photo or choose another filter.")
        return
    selected = st.selectbox("Choose a Report", [row["record_id"] for row in filtered],
                            format_func=lambda identifier: next(
                                f"{row['record_id'][:8]} · {row['location'] or 'Unknown location'} · {assessment_flag(row)}"
                                for row in filtered if row["record_id"] == identifier))
    record = next(row for row in filtered if row["record_id"] == selected)
    left, right = st.columns([1, 1], gap="large")
    with left, st.container(border=True, key="observation_card"):
        render_observation(record)
    with right, st.container(border=True, key="assessment_card"):
        render_review_form(record)
