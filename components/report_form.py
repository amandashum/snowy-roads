"""Render citizen photo uploads, observation details, and the saved receipt."""

import base64
import hashlib

import streamlit as st

import reports
from components.classifier import classify_photo
from components.layout import render_page_heading, render_section_heading
from components.photo import render_photo
from components.suggestion import render_condition_title, show_results
from components.report_history import open_my_report
from components.road_map import street_names


def start_new_report():
    """Clear the submission controls without deleting saved reports."""
    st.session_state.receipt = None
    st.session_state.upload_revision = st.session_state.get("upload_revision", 0) + 1
    for key in ("report_location", "report_capture_time", "report_consent", "last_submission"):
        st.session_state.pop(key, None)


def render_report_form():
    """Render compact upload and details cards, followed by the latest saved result."""
    render_page_heading("Check a Road Condition", "Upload a photo to check the condition and save a report",
                        "Snowy Roads")
    open_results = False
    left, right = st.columns([1, 1], gap="large")
    with left, st.container(border=True, key="upload_card"):
        render_section_heading("01", "Upload a Photo")
        st.caption("Drag and drop one JPEG or PNG, or choose Upload · Maximum 10 MB")
        revision = st.session_state.get("upload_revision", 0)
        photo = st.file_uploader("Road photo", type=["jpg", "jpeg", "png"], label_visibility="collapsed",
                                 key=f"road_photo_{revision}", accept_multiple_files=False,
                                 help="JPEG or PNG, up to 10 MB and 20 million pixels")
        valid_photo = False
        if photo is not None:
            valid_photo = render_photo(photo, "upload", max_height=180)
        else:
            st.caption("Choose a clear view of the road surface")
    with right, st.container(border=True, key="details_card"):
        render_section_heading("02", "Details & Analysis")
        st.caption("Add a location to make the report easier to find")
        with st.form("citizen_report", border=False):
            location = st.selectbox("Location (optional)", street_names(), index=None, key="report_location",
                                    accept_new_options=True,
                                    placeholder="Start typing a street, intersection, or landmark",
                                    help="Pick a St. John's street from the list to show the photo on the "
                                         "Road Map. Small spelling mistakes still find the street.") or ""
            location = location.strip()[:200]
            captured_at = st.text_input("Photo Taken At (optional)", key="report_capture_time",
                                        placeholder="2026-10-30T14:30:00-03:30",
                                        help="Include the timezone offset, or leave blank if unknown", max_chars=64)
            consent = st.checkbox("I agree to terms and conditions",
                                  key="report_consent")
            submit = st.form_submit_button("Analyze Photo", key="analyze_photo", type="primary", width="stretch")
        if submit:
            if photo is None:
                st.error("Please upload a road photo before analyzing")
            elif not valid_photo:
                st.error("Please upload a valid JPEG or PNG photo before analyzing")
            elif not consent:
                st.error("Please agree to terms and conditions before submitting")
            else:
                content = photo.getvalue()
                signature = (hashlib.sha256(content).hexdigest(), location, captured_at)
                try:
                    if signature != st.session_state.get("last_submission") or not st.session_state.receipt:
                        with st.spinner("Analyzing photo…"):
                            st.session_state.receipt = reports.create_report({
                                "image_base64": base64.b64encode(content).decode(),
                                "location": location, "captured_at": captured_at}, predictor=classify_photo)
                        st.session_state.last_submission = signature
                    identifier = st.session_state.receipt["record_id"]
                    history = st.session_state.setdefault("my_report_ids", [])
                    if identifier not in history:
                        history.append(identifier)
                    open_results = True
                except (ValueError, OSError) as error:
                    st.error(f"The report could not be submitted: {error}")
    if st.session_state.receipt:
        receipt = st.session_state.receipt
        with st.container(border=True, key="saved_report"):
            summary, actions = st.columns([1.4, 1], gap="large", vertical_alignment="center")
            with summary:
                render_condition_title(receipt["predicted_class"] if receipt["inference_status"] == "available"
                                       else None, "LATEST SAVED REPORT")
                st.success(f"Report saved · Report #{receipt['record_id']}")
                st.caption(receipt["location"] or "Location not provided")
            with actions:
                st.button("View my report", key="view_my_report", type="primary", width="stretch",
                          on_click=open_my_report, args=(receipt["record_id"],))
                view_results = st.button("View Results", key="view_receipt_results", icon=":material/bar_chart:",
                                         width="stretch")
                st.button("Check Another Photo", key="new_photo", icon=":material/add_photo_alternate:",
                          on_click=start_new_report, width="stretch")
        if open_results or view_results:
            show_results(receipt)
