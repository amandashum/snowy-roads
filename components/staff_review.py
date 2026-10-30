"""Render the staff queue, retained observations, and separate review decisions."""

import json

import streamlit as st

import reports
from components.labels import ACTIONS, DISPLAY, PRIORITIES
from components.layout import render_page_heading, render_section_heading
from components.photo import preview_image, render_photo
from components.suggestion import assessment_flag, show_results
from components.report_history import render_activity, report_time, DECISIONS


def load_reports():
    """Read reports newest first and warn about unreadable records."""
    records = []
    for path in reports.RECORDS.glob("*.json"):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            st.warning(f"A saved report could not be read: {path.name}")
    return sorted(records, key=lambda row: row["uploaded_at"], reverse=True)


def priority_situations(records):
    """Return reviewed reports ordered by staff priority, newest review first within each level."""
    order = {"urgent-assessment": 0, "assess-soon": 1, "routine": 2}
    reviewed = [row for row in records if row["review_status"] in
                {"confirmed", "corrected", "inconclusive"} and row["priority"] in order
                and row.get("action") not in reports.CLOSED_ACTIONS]
    reviewed.sort(key=lambda row: row.get("reviewed_at") or "", reverse=True)
    return sorted(reviewed, key=lambda row: order[row["priority"]])


def open_priority_situation(identifier):
    """Switch to reviewed reports and select the clicked situation before widgets render."""
    st.session_state.review_filter = "Reviewed"
    st.session_state.selected_report = identifier


@st.dialog("Plow Crew Notification · Demo", width="small", on_dismiss="rerun")
def show_crew_notification(record):
    """Preview notification feedback without sending a message or changing dispatch records."""
    st.info("Demo only. No message has been sent to a plow crew.")
    st.success("Snow Plow Crew notified for the next available snow plow.",
               icon=":material/local_shipping:")
    st.text(record["location"] or "Unknown location")
    st.caption(f"Report {record['record_id'][:8]}")
    if st.button("Close", icon=":material/close:", key="close_crew_notification", width="stretch"):
        st.rerun()


def confirm_delete_report(record):
    """Ask for confirmation before permanently deleting a closed report."""
    identifier = record["record_id"]
    st.write(f"Report {identifier} · {record.get('location') or 'Unknown location'}")
    st.warning("Permanently delete this report, its photo, notes, and activity history? This cannot be undone.")
    cancel, delete = st.columns(2)
    with cancel:
        if st.button("Cancel", key="cancel_delete_report", width="stretch"):
            st.session_state.pop("delete_requested", None)
            st.rerun()
    with delete:
        if st.button("Permanently delete", key="confirm_delete_report", type="primary", width="stretch"):
            try:
                reports.delete_closed_report(identifier)
            except (OSError, ValueError) as error:
                st.error(f"The report could not be fully deleted: {error}")
            else:
                st.session_state.my_report_ids = [item for item in st.session_state.get("my_report_ids", [])
                                                  if item != identifier]
                if (st.session_state.get("receipt") or {}).get("record_id") == identifier:
                    st.session_state.pop("receipt", None)
                if st.session_state.get("focused_report") == identifier:
                    st.session_state.pop("focused_report", None)
                st.session_state.pop("delete_requested", None)
                st.session_state.review_deleted = True
                st.rerun()


def render_priority_situations(records):
    """Display a scrollable panel of reviewed photos with staff-assigned priorities and actions."""
    situations = priority_situations(records)
    with st.container(border=True, key="priority_situations", height="stretch"):
        st.subheader("Priority Situations")
        st.caption(f"{len(situations)} reviewed reports · Staff-assigned priority")
        if not situations:
            st.info("No reviewed reports have been assigned priority.")
            return
        with st.container(height="stretch", border=False, key="priority_list"):
            for row in situations:
                identifier = row["record_id"]
                with st.container(border=True, key=f"priority_card_{identifier}"):
                    urgent = row["priority"] == "urgent-assessment"
                    st.badge(PRIORITIES[row["priority"]], icon=":material/flag:",
                             color="red" if urgent else "blue")
                    image = reports.UPLOADS / f"{identifier}.png"
                    try:
                        st.image(preview_image(image, max_height=110),
                                 alt=f"Reviewed road photo: {row['location'] or 'Unknown location'}",
                                 width="content")
                    except (OSError, ValueError):
                        st.caption("Photo preview unavailable")
                    st.text(row["location"] or "Unknown location")
                    st.caption(f"Captured: {row['captured_at'] or 'Unknown'}")
                    st.caption(f"Review: {row['review_status'].title()}")
                    if row["action"] == "recommend-plowing":
                        with st.container(key=f"plowing_recommended_{identifier}"):
                            st.error("Plowing recommended", icon=":material/local_shipping:")
                            st.caption("Staff recommendation · Not dispatched")
                        if st.button("Notify Plow Crew", icon=":material/notifications_active:",
                                     key=f"notify_crew_{identifier}", width="stretch",
                                     help="Demo notification preview. No message is sent."):
                            show_crew_notification(row)
                    else:
                        st.write(ACTIONS[row["action"]])
                    st.button("Open assessment", key=f"priority_{identifier}", width="stretch",
                              on_click=open_priority_situation, args=(identifier,))


def render_observation(record):
    """Render the retained photo, optional analyzed crop, and original suggestion."""
    selected = record["record_id"]
    st.subheader("Road Observation")
    st.subheader(record["location"] or "Unknown location")
    st.caption(f"REPORT {selected[:8].upper()} · {record['review_status'].replace('-', ' ').title()}")
    image = reports.UPLOADS / f"{selected}.png"
    if image.exists():
        render_photo(image, f"observation_{selected}", max_height=260)
        with st.expander("Image Details"):
            if st.toggle("Show analyzed center crop", key=f"crop_{selected}"):
                from PIL import Image
                from torchvision.transforms.functional import center_crop, resize
                with Image.open(image) as original:
                    crop = center_crop(resize(original.convert("RGB"), 256), 224)
                st.image(crop, alt="Center region used by the classifier after resizing", width=224)
                st.caption("The classifier uses this center crop; it is not a detected road region.")
    else:
        st.warning("The retained photo is unavailable.")
    st.caption(f"Captured: {record['captured_at'] or 'Unknown'}")
    with st.expander("Report activity"):
        st.text(f"Report #{selected}")
        render_activity(record)
    with st.container(key=f"analysis_{selected}"):
        st.caption("PHOTO ANALYSIS")
        if record["inference_status"] == "available":
            st.markdown(f"**{DISPLAY[record['predicted_class']]}**")
        else:
            st.warning("Analysis unavailable. Review the photo manually.")
        if st.button("View Results", key=f"view_analysis_{selected}", icon=":material/bar_chart:", width="stretch"):
            show_results(record)


def confirm_suggestion(record, identifiers):
    """Save a confirmation and select the next report before its widget renders."""
    selected = record["record_id"]
    try:
        reports.review_report(selected, {"staff_class": record["predicted_class"],
                                        "priority": record["priority"], "action": record["action"],
                                        "staff_note": record["staff_note"]})
    except (ValueError, OSError) as error:
        st.session_state.review_error = str(error)
        return
    position = identifiers.index(selected)
    remaining = identifiers[position + 1:] + identifiers[:position]
    if remaining:
        st.session_state.selected_report = remaining[0]
    st.session_state.review_saved = True


def render_review_choice(record, identifiers):
    """Offer quick confirmation or reveal the existing detailed assessment form."""
    selected = record["record_id"]
    if st.session_state.get("review_details_report") != selected:
        st.session_state.review_details_report = None
    available = record["inference_status"] == "available" and record["predicted_class"] in DISPLAY
    if available:
        st.subheader("Is the AI suggestion correct?")
        with st.container(horizontal=True):
            st.button("Yes — Confirm & Next", key=f"confirm_review_{selected}", type="primary",
                      on_click=confirm_suggestion, args=(record, identifiers))
            if st.button("No — Review Details", key=f"correct_review_{selected}"):
                st.session_state.review_details_report = selected
        st.caption("Confirmation keeps the current priority and next action. It does not dispatch a crew.")
    if not available or st.session_state.get("review_details_report") == selected:
        render_review_form(record)


def render_review_form(record):
    """Render and save staff decisions without replacing model outputs."""
    selected = record["record_id"]
    render_section_heading("02", "Your Assessment")
    st.caption("Confirm the condition and record what happens next")
    with st.form(f"review_{selected}", border=False):
        choices = list(DISPLAY)
        current = record["staff_class"] or record["predicted_class"]
        condition = st.selectbox("Reviewed Condition", choices, index=choices.index(current) if current in choices else None,
                                 format_func=DISPLAY.get, placeholder="Select a reviewed condition")
        priority_column, action_column = st.columns(2)
        with priority_column:
            priority = st.selectbox("Priority", list(PRIORITIES), format_func=PRIORITIES.get,
                                    index=list(PRIORITIES).index(record["priority"]))
        with action_column:
            action = st.selectbox("Next Action", list(ACTIONS), format_func=ACTIONS.get,
                                  index=list(ACTIONS).index(record["action"]))
        note = st.text_area("Review Notes", value=record["staff_note"], max_chars=2000, height=120,
                            placeholder="Add context for the next person…",
                            help="A short reason is required for monitoring, reassessment, resolution, and completed plowing. Visible to the citizen.")
        save = st.form_submit_button("Save Review", key=f"save_review_{selected}", type="primary", width="stretch")
    if save:
        try:
            reports.review_report(selected, {"staff_class": condition, "priority": priority,
                                            "action": action, "staff_note": note})
            st.session_state.review_details_report = None
            st.session_state.review_saved = True
            st.rerun()
        except (ValueError, OSError) as error:
            st.error(f"The review could not be saved: {error}")


def render_staff_review():
    """Render the filtered staff queue and the selected report's review controls."""
    render_page_heading("Review Road Conditions", "Review a photo, confirm the condition, and choose the next step", "Snowy Roads")
    if st.session_state.pop("review_saved", False):
        st.success("Review saved. The queue is up to date.")
    if st.session_state.pop("review_deleted", False):
        st.success("Closed report deleted.")
    if error := st.session_state.pop("review_error", None):
        st.error(f"The review could not be saved: {error}")
    records = load_reports()
    with st.container(horizontal=True, key="review_stats"):
        st.metric("Reports", len(records))
        st.metric("Awaiting Review", sum(row["review_status"] == "pending" for row in records))
        st.metric("Needs Reassessment", sum(row.get("action") == "reassess" for row in records))
        st.metric("Snow to Assess", sum(row["review_status"] == "pending" and
                  row["predicted_class"] in {"light", "medium"} for row in records))
    with st.container(border=True, key="queue_controls"):
        filters, refresh = st.columns([4, 1], vertical_alignment="bottom")
        with filters:
            filter_name = st.segmented_control("Show Reports", ["Pending", "Reassessment", "All", "Reviewed", "Closed"],
                                               default="Pending", key="review_filter")
        with refresh:
            if st.button("Refresh", key="refresh_reports", icon=":material/refresh:", width="stretch"):
                st.rerun()
    filtered = [row for row in records if matches_filter(row, filter_name)]
    if not filtered:
        st.info("No reports in this view. Submit a road photo or choose another filter.")
        render_priority_situations(records)
        return
    identifiers = [row["record_id"] for row in filtered]
    if st.session_state.get("selected_report") not in identifiers:
        st.session_state.selected_report = identifiers[0]
    selected = st.selectbox("Choose a Report", identifiers, key="selected_report",
                        format_func=lambda identifier: next(
                            f"{row['record_id'][:8]} · {row['location'] or 'Unknown location'} · {assessment_flag(row)}"
                            for row in filtered if row["record_id"] == identifier))
    record = next(row for row in filtered if row["record_id"] == selected)
    if record.get("action") == "reassess":
        st.warning("Needs reassessment: review the latest observations and record the next step.")
    with st.container(key="review_panels"):
        workspace, panel = st.columns([3, 1], gap="large")
        with workspace, st.container(border=True, key="review_card"):
            render_observation(record)
            st.divider()
            render_review_choice(record, identifiers)
            if filter_name == "Closed" and record.get("action") in reports.CLOSED_ACTIONS:
                if st.button("Delete report", key=f"delete_report_{selected}", icon=":material/delete:"):
                    st.session_state.delete_requested = selected
                if st.session_state.get("delete_requested") == selected:
                    confirm_delete_report(record)
        with panel:
            render_priority_situations(records)
    render_related_reports(records, record)


def matches_filter(record, filter_name):
    """Select queue records by review state or explicit staff outcome."""
    if filter_name == "All":
        return True
    if filter_name == "Pending":
        return record["review_status"] == "pending"
    if filter_name == "Reviewed":
        return (record["review_status"] != "pending"
                and record.get("action") not in reports.CLOSED_ACTIONS)
    if filter_name == "Reassessment":
        return record.get("action") == "reassess"
    if filter_name == "Closed":
        return record.get("action") in reports.CLOSED_ACTIONS
    return False


def reports_at_location(records, selected):
    """Match supplied location text by case and whitespace without inferring duplicate incidents."""
    location = " ".join((selected.get("location") or "").split()).casefold()
    if not location:
        return []
    return [row for row in records if row["record_id"] != selected["record_id"]
            and " ".join((row.get("location") or "").split()).casefold() == location]


def open_related_report(identifier):
    """Open an individual related report without applying decisions to its neighbors."""
    st.session_state.review_filter = "All"
    st.session_state.selected_report = identifier


def render_related_reports(records, selected):
    """Show other observations at the same supplied location with separate photos and histories."""
    with st.expander("Reports at this location"):
        st.caption("Matching location text · Reports may describe different incidents or dates")
        related = reports_at_location(records, selected)
        if not related:
            st.info("No other reports match this location.")
        for row in related:
            identifier = row["record_id"]
            with st.container(border=True):
                st.text(f"Report #{identifier}")
                st.text(row.get("location") or "Location not provided")
                st.caption(f'Received: {report_time(row.get("uploaded_at"))}')
                st.caption(f'Captured: {row.get("captured_at") or "Unknown"}')
                status = "Awaiting assessment" if row["review_status"] == "pending" else DECISIONS.get(row.get("action"), "Reviewed")
                st.text(status)
                image = reports.UPLOADS / f"{identifier}.png"
                if image.exists():
                    render_photo(image, f"related_{identifier}", max_height=140)
                else:
                    st.caption("Photo preview unavailable")
                st.button("Open report", key=f"related_{identifier}", on_click=open_related_report,
                          args=(identifier,))
