"""Display a saved report result in a dismissible modal card."""

import streamlit as st

from components.suggestion import show_suggestion


@st.dialog("Road condition result", width="medium", dismissible=True)
def show_result_card(receipt):
    """Show the saved result without repeating inference or changing the report."""
    with st.container(key=f"report-result-{receipt['record_id']}"):
        st.success(f"Report saved · {receipt['record_id'][:8]}")
        show_suggestion(receipt)
        st.caption("Press Esc or use the close icon to dismiss. Your report stays saved.")
