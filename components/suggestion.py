"""Display original model suggestions consistently across dashboard views."""

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


def show_suggestion(record):
    """Render the original suggestion, scores, version, and uncertainty limitations."""
    if record["inference_status"] != "available":
        st.warning("The model is unavailable. This report is saved for manual review.")
        return
    st.markdown(f"### {DISPLAY[record['predicted_class']]}")
    st.caption(f"Model: {record['model_version']} · Label mapping: {record['label_mapping_version']}")
    st.info(assessment_flag(record))
    if record["uncertain"] is None:
        st.caption("Uncertainty threshold has not been validated. Every suggestion needs staff review.")
    with st.expander("Inspect model scores"):
        st.caption("Scores describe model preference, not verified probabilities of correctness.")
        st.table([{"Condition": DISPLAY[label], "Model score": f"{score:.1%}"}
                  for label, score in record["class_scores"].items()], alt="Scores for all four condition classes")
    st.caption("A condition suggestion does not establish road safety or whether plowing is needed.")
