"""Render the shared dashboard shell and single-page workspace navigation."""

import base64

import streamlit as st

import reports


def render_sidebar():
    """Render navigation and return the selected workspace."""
    with st.sidebar:
        st.header("Snowy Roads")
        st.caption("Winter road observations")
        page = st.radio("Workspace", ["Report a condition", "Staff review"], key="workspace")
        st.divider()
        st.caption("LOCAL DEMONSTRATION")
        st.write("Photos and review records are saved on this computer.")
        st.caption("No staff sign-in is configured. These views are not access controls.")
        st.caption("For emergencies, use your municipality’s established reporting process.")
    return page


def render_brand():
    """Display the local background, shared surface styling, and branding."""
    dashboard = reports.ROOT / "dashboard"
    background = base64.b64encode(
        (dashboard / "assets" / "snowy_road_background.png").read_bytes()
    ).decode("ascii")
    stylesheet = (dashboard / "theme.css").read_text(encoding="utf-8")
    st.html(f"<style>{stylesheet.replace('__BACKGROUND_IMAGE__', background)}</style>")
    st.html((reports.ROOT / "dashboard" / "brand.html").read_text(encoding="utf-8"))


def render_footer():
    """Display the shared model limitations."""
    st.divider()
    st.caption("Traffic-camera training data may not represent citizen phone photos. All suggestions require human review.")
