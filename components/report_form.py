"""Render citizen photo uploads, observation details, and the saved receipt."""

import base64

import streamlit as st

import reports
from components.classifier import classify_photo
from components.result_card import show_result_card


def render_report_form():
    """Render the submission workspace and save photos when reports are submitted."""
    st.title("Report a road condition")
    st.write("Upload a recent road photo for a suggested condition and staff assessment.")
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.subheader("Upload a road photo")
        st.caption("Drag and drop one JPEG or PNG into the area below, or use the upload button to browse.")
        photo = st.file_uploader("Road photo", type=["jpg", "jpeg", "png"],
                                 accept_multiple_files=False,
                                 help="JPEG or PNG, up to 10 MB and 20 million pixels. Avoid identifying people.")
        if photo is not None:
            try:
                st.image(photo, alt="Selected road photo for submission", width="stretch")
            except Exception:
                st.error("This photo could not be previewed. Choose a valid JPEG or PNG.")
    with right:
        st.subheader("Observation details")
        with st.form("citizen_report"):
            location = st.text_input("Location (optional)", max_chars=200,
                                     placeholder="Street, intersection, or nearby landmark")
            captured_at = st.text_input("Photo capture time (optional)",
                                        placeholder="2026-10-29T14:30:00-07:00",
                                        help="Include the timezone offset. Leave blank if unknown.", max_chars=64)
            submit = st.form_submit_button("Analyze and save report", type="primary", width="stretch")
        show_result = False
        if submit:
            if photo is None:
                st.error("Choose a road photo first.")
            else:
                try:
                    with st.spinner("Saving photo and checking the road condition…"):
                        st.session_state.receipt = reports.create_report({
                            "image_base64": base64.b64encode(photo.getvalue()).decode(),
                            "location": location, "captured_at": captured_at}, predictor=classify_photo)
                    show_result = True
                except (ValueError, OSError) as error:
                    st.error(f"The report could not be submitted: {error}")
        if st.session_state.receipt:
            receipt = st.session_state.receipt
            st.success(f"Report saved · {receipt['record_id'][:8]}")
            reopen = st.button("View latest result", key="view_latest_result")
            st.caption("This receipt belongs to your most recently saved report.")
            if show_result or reopen:
                show_result_card(receipt)
