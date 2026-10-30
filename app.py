"""Single-page Streamlit entry point for citizen reports and staff review."""

import streamlit as st

import reports
from components.layout import LOGO, render_brand, render_footer, render_sidebar
from components.report_form import render_report_form
from components.staff_review import render_staff_review


def main():
    """Initialize local storage and session state, then render the selected workspace."""
    st.set_page_config(page_title="Snowy Roads · Road observations", page_icon=str(LOGO), layout="wide")
    reports.RECORDS.mkdir(exist_ok=True)
    reports.UPLOADS.mkdir(exist_ok=True)
    if "receipt" not in st.session_state:
        st.session_state.receipt = None

    render_brand()
    page = render_sidebar()
    if page == "Report a condition":
        render_report_form()
    else:
        render_staff_review()
    render_footer()


if __name__ == "__main__":
    main()
