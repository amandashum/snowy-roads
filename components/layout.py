"""Render the shared dashboard shell and single-page workspace navigation."""

import base64

import streamlit as st

import reports

LOGO = reports.ROOT / "tests" / "olaf-removebg-preview.png"


def render_sidebar():
    """Render navigation and return the selected workspace."""
    with st.sidebar:
        st.image(str(LOGO), alt="Snowy Roads logo", width=80)
        st.header("Snowy Roads")
        st.caption("Winter road observations")
        page = st.radio("Workspace", ["Report a condition", "Staff review"], key="workspace")
    return page


def render_brand():
    """Display the local dashboard branding."""
    logo_data = base64.b64encode(LOGO.read_bytes()).decode("ascii")
    template = (reports.ROOT / "dashboard" / "brand.html").read_text(encoding="utf-8")
    st.html(template.replace("{{LOGO_URI}}", f"data:image/png;base64,{logo_data}"))


def render_footer():
    """Display the shared model limitations."""
    st.divider()
    st.caption("Traffic-camera training data may not represent citizen phone photos. All suggestions require human review.")
