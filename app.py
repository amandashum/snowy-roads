"""
This program displays the local Snowy Roads upload and staff review dashboard.

Citizen report mode previews a JPEG/PNG photo, collects optional location and
capture time, and saves a report with a real classifier suggestion when the
trained model is available. Capture time stays unknown unless supplied by the
user. Staff review mode lists saved reports, highlights snow-covered or
uncertain observations for assessment, and lets staff record a reviewed
condition, priority, action recommendation, and note. Original model outputs
remain separate from staff decisions and persist across application restarts.

Streamlit provides interactive controls while local HTML/CSS adds a restrained
brand treatment. No external fonts or remote images are loaded. This app is a
local demonstration with no staff authentication or public deployment. It
does not dispatch snowplows or declare roads safe. Model scores are not
calibrated probabilities, and citizen phone photos require further evaluation.
"""

import base64
import json

import streamlit as st

from reports import LABELS, RECORDS, ROOT, UPLOADS, create_report, review_report

DISPLAY = {item["id"]: item["display_label"] for item in LABELS}
DISPLAY.update(inconclusive="Inconclusive", unsuitable="Unsuitable photo")
PRIORITIES = {"unassigned": "Unassigned", "routine": "Routine", "assess-soon": "Assess soon",
              "urgent-assessment": "Urgent assessment"}
ACTIONS = {"undecided": "Undecided", "monitor": "Monitor conditions",
           "assess-plowing": "Assess need for plowing", "recommend-plowing": "Recommend plowing",
           "no-action": "No action recommended"}


@st.cache_resource(max_entries=2)
def cached_checkpoint(path, modified_ns, size):
    """
    Reuse a CPU model until its checkpoint modification time or size changes.

    Side effects:
        Read model parameters and retain the model in Streamlit's resource cache.

    Returns:
        The evaluation-mode model and checkpoint metadata.
    """
    from inference import load_checkpoint
    return load_checkpoint(path)


def classify_photo(image_path, model_path):
    """
    Classify a photo using a cached model that refreshes after training saves.

    Side effects:
        Read checkpoint metadata and an image; run CPU inference.

    Returns:
        The original classifier suggestion and scores.
    """
    from inference import predict
    stat = model_path.stat()
    loaded = cached_checkpoint(str(model_path), stat.st_mtime_ns, stat.st_size)
    if loaded[1].get("smoke_test"):
        raise ValueError("Smoke-test checkpoints cannot power real report suggestions")
    return predict(image_path, model_path, loaded_checkpoint=loaded)


def assessment_flag(record):
    """
    Explain why a report belongs in the assessment queue without assigning urgency.

    Side effects:
        None.

    Returns:
        A plain-language assessment flag.
    """
    if record["inference_status"] != "available":
        return "Manual condition review"
    if record["uncertain"] is True:
        return "Uncertain suggestion"
    if record["predicted_class"] in {"light", "medium"}:
        return "Assess snow-removal need"
    return "Routine condition review"


def show_suggestion(record):
    """
    Show the original model output with scores, version, and uncertainty limitations.

    Side effects:
        Render Streamlit condition, score, and explanation elements.

    Returns:
        None. The suggestion is displayed directly.
    """
    st.subheader("Model suggestion")
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


def load_reports():
    """
    Read locally saved reports and identify unreadable records without replacing them.

    Side effects:
        Read JSON files and display a warning for damaged records.

    Returns:
        Reports sorted with the latest upload first.
    """
    records = []
    for path in RECORDS.glob("*.json"):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            st.warning(f"A saved report could not be read: {path.name}")
    return sorted(records, key=lambda row: row["uploaded_at"], reverse=True)


st.set_page_config(page_title="Snowy Roads · Road observations", page_icon=":material/weather_snowy:", layout="wide")
RECORDS.mkdir(exist_ok=True)
UPLOADS.mkdir(exist_ok=True)
if "receipt" not in st.session_state:
    st.session_state.receipt = None

st.html((ROOT / "dashboard" / "brand.html").read_text(encoding="utf-8"))
with st.sidebar:
    st.header("Snowy Roads")
    st.caption("Winter road observations")
    page = st.radio("Workspace", ["Report a condition", "Staff review"], key="workspace")
    st.divider()
    st.caption("LOCAL DEMONSTRATION")
    st.write("Photos and review records are saved on this computer.")
    st.caption("No staff sign-in is configured. These views are not access controls.")
    st.caption("For emergencies, use your municipality’s established reporting process.")

if page == "Report a condition":
    st.title("Report a road condition")
    st.write("Upload a recent road photo for a suggested condition and staff assessment.")
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        photo = st.file_uploader("Road photo", type=["jpg", "jpeg", "png"],
                                 help="JPEG or PNG, up to 10 MB and 20 million pixels. Avoid identifying people.")
        if photo is not None:
            try:
                st.image(photo, alt="Selected road photo for submission", width="stretch")
            except Exception:
                st.error("This photo could not be previewed. Choose a valid JPEG or PNG.")
        else:
            with st.container(border=True):
                st.subheader("A useful photo shows the road surface")
                st.write("Include the travel lane, use a clear view, and take the photo from a safe position.")
                st.caption("No photo has been selected.")
    with right:
        st.subheader("Observation details")
        with st.form("citizen_report"):
            location = st.text_input("Location (optional)", max_chars=200,
                                     placeholder="Street, intersection, or nearby landmark")
            captured_at = st.text_input("Photo capture time (optional)",
                                        placeholder="2026-10-29T14:30:00-07:00",
                                        help="Include the timezone offset. Leave blank if unknown.", max_chars=64)
            st.caption("Upload time is recorded separately. Missing capture time and location remain unknown.")
            consent = st.checkbox("Save this photo and supplied details locally for staff review.")
            submit = st.form_submit_button("Analyze and save report", type="primary", width="stretch")
        if submit:
            if photo is None:
                st.error("Choose a road photo first.")
            elif not consent:
                st.error("Confirm local storage before submitting.")
            else:
                try:
                    with st.spinner("Saving photo and checking the road condition…"):
                        st.session_state.receipt = create_report({
                            "image_base64": base64.b64encode(photo.getvalue()).decode(),
                            "location": location, "captured_at": captured_at}, predictor=classify_photo)
                except (ValueError, OSError) as error:
                    st.error(f"The report could not be submitted: {error}")
        if st.session_state.receipt:
            receipt = st.session_state.receipt
            st.success(f"Report saved · {receipt['record_id'][:8]}")
            show_suggestion(receipt)
            st.caption("This receipt belongs to your most recently saved report.")
else:
    st.title("Staff review")
    st.write("Confirm road conditions and record the next action. Dispatch remains a staff decision.")
    records = load_reports()
    with st.container(horizontal=True):
        st.metric("Reports", len(records))
        st.metric("Awaiting review", sum(row["review_status"] == "pending" for row in records))
        st.metric("Pending snow assessments", sum(row["review_status"] == "pending" and
                  row["predicted_class"] in {"light", "medium"} for row in records))
    st.divider()
    filter_name = st.segmented_control("Show reports", ["Pending", "All", "Reviewed"], default="Pending")
    filtered = [row for row in records if filter_name == "All" or
                (row["review_status"] == "pending") == (filter_name == "Pending")]
    if st.button("Refresh reports", icon=":material/refresh:"):
        st.rerun()
    if not filtered:
        st.info("No reports in this view. Submit a road photo or choose another filter.")
    else:
        selected = st.selectbox("Choose a report", [row["record_id"] for row in filtered],
                                format_func=lambda identifier: next(
                                    f"{row['record_id'][:8]} · {row['location'] or 'Unknown location'} · {assessment_flag(row)}"
                                    for row in filtered if row["record_id"] == identifier))
        record = next(row for row in filtered if row["record_id"] == selected)
        left, right = st.columns([1.2, 1], gap="large")
        with left:
            image = UPLOADS / f"{selected}.png"
            if image.exists():
                st.image(str(image), alt="Original road observation retained for staff review", width="stretch")
                if st.toggle("Show analyzed center crop", key=f"crop_{selected}"):
                    from PIL import Image
                    from torchvision.transforms.functional import center_crop, resize
                    with Image.open(image) as original:
                        crop = center_crop(resize(original.convert("RGB"), 256), 224)
                    st.image(crop, alt="Center region used by the classifier after resizing", width=224)
                    st.caption("The classifier uses this center crop; it is not a detected road region.")
            else:
                st.warning("The retained photo is unavailable.")
            st.write(f"Location: {record['location'] or 'Unknown'}")
            st.caption(f"Captured: {record['captured_at'] or 'Unknown'}")
            st.caption(f"Uploaded: {record['uploaded_at']} · Review status: {record['review_status']}")
            show_suggestion(record)
        with right:
            st.subheader("Staff decision")
            with st.form(f"review_{selected}"):
                choices = list(DISPLAY)
                current = record["staff_class"] or record["predicted_class"]
                condition = st.selectbox("Reviewed condition", choices, index=choices.index(current) if current in choices else None,
                                         format_func=DISPLAY.get, placeholder="Select a reviewed condition")
                priority = st.selectbox("Priority", list(PRIORITIES), format_func=PRIORITIES.get,
                                        index=list(PRIORITIES).index(record["priority"]))
                action = st.selectbox("Action recommendation", list(ACTIONS), format_func=ACTIONS.get,
                                      index=list(ACTIONS).index(record["action"]))
                note = st.text_area("Review notes", value=record["staff_note"], max_chars=2000)
                st.caption("Saving records a staff assessment. It does not contact crews or dispatch a snowplow.")
                save = st.form_submit_button("Save staff review", type="primary", width="stretch")
            if save:
                try:
                    review_report(selected, {"staff_class": condition, "priority": priority,
                                             "action": action, "staff_note": note})
                    st.success("Staff review saved. Refresh reports to update the queue.")
                except (ValueError, OSError) as error:
                    st.error(f"The review could not be saved: {error}")

st.divider()
st.caption("Traffic-camera training data may not represent citizen phone photos. All suggestions require human review.")
