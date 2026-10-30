"""Render the shared dashboard shell and single-page workspace navigation."""

import base64

import streamlit as st

import reports


def render_sidebar():
    """Render navigation and return the selected workspace."""
    with st.sidebar:
        render_brand()
        st.caption("WORKSPACE")
        page = st.radio("Workspace", ["Report a condition", "My reports", "Road map", "Staff review"], key="workspace",
                        format_func=lambda page: {"Report a condition": "Report a Condition", "My reports": "My Reports",
                                                  "Road map": "Road Map", "Staff review": "Staff Review"}[page],
                        label_visibility="collapsed")
    return page


def apply_theme():
    """Apply shared typography, surfaces, and the supplied scenic background."""
    dashboard = reports.ROOT / "dashboard"
    background = base64.b64encode(
        (dashboard / "assets" / "snowy_road_background.png").read_bytes()
    ).decode("ascii")
    stylesheet = (dashboard / "theme.css").read_text(encoding="utf-8")
    st.html(f"<style>{stylesheet.replace('__BACKGROUND_IMAGE__', background)}</style>")


def render_brand():
    """Display the sidebar wordmark."""
    st.html((reports.ROOT / "dashboard" / "brand.html").read_text(encoding="utf-8"))


def render_page_heading(title, description, eyebrow):
    """Give both workspaces the same clear heading hierarchy."""
    from html import escape

    with st.container(key="page_heading"):
        st.html(f'<p class="snowy-eyebrow">{escape(eyebrow)}</p>')
        st.title(title)
        st.write(description)


def render_section_heading(number, title):
    """Label a workflow step without competing with the page title."""
    from html import escape

    st.html(f'<div class="snowy-section-heading"><span>{escape(number)}</span>'
            f'<h2>{escape(title)}</h2></div>')
