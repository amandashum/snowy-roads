"""Display photo-analysis results in a shared dialog."""

from html import escape

import streamlit as st

from components.labels import DISPLAY


def assessment_flag(record):
    """Explain why a report belongs in the assessment queue without assigning urgency."""
    if record["inference_status"] != "available":
        return "Manual condition review"
    if record["uncertain"] is True:
        return "Uncertain suggestion"
    if record["predicted_class"] in {"light", "medium"}:
        return "Assess snow-removal need"
    return "Routine condition review"


def render_condition_title(condition, label):
    """Use one heading treatment for the result dialog and saved-report summary."""
    title = DISPLAY[condition].title() if condition is not None else "Ready for Staff Review"
    st.html(f'<div class="snowy-condition-heading"><p class="snowy-condition-label">{escape(label)}</p>'
            f'<h3 class="snowy-condition-title">{escape(title)}</h3></div>')


@st.dialog("Road Condition Analysis", width="medium", on_dismiss="rerun")
def show_results(record):
    """Show the original prediction and all class scores without changing the record."""
    if record["inference_status"] != "available":
        st.warning("Analysis unavailable — your photo is saved for staff review")
    else:
        with st.container(key="detected_condition"):
            render_condition_title(record["predicted_class"], "DETECTED CONDITION")
        if record["uncertain"] is True:
            st.warning("Uncertain result. Check the photo before choosing a condition.")
        scores = sorted(record["class_scores"].items(), key=lambda item: item[1], reverse=True)
        st.table([{"Condition": DISPLAY[label], "Match Score": f"{score:.1%}"}
                  for label, score in scores], alt="Photo match scores for all four road conditions")
    st.caption(f"Report {record['record_id'][:8]}")
    if st.button("Done", key="close_analysis_results", type="primary", width="stretch"):
        st.rerun()
