"""Collect an optional photo time for today in the selected timezone."""

from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import streamlit as st


TIME_ZONES = {
    "America/St_Johns": "Newfoundland",
    "America/Halifax": "Atlantic",
    "America/Toronto": "Eastern",
    "America/Winnipeg": "Central",
    "America/Regina": "Saskatchewan",
    "America/Edmonton": "Mountain",
    "America/Vancouver": "Pacific",
    "America/Whitehorse": "Yukon",
    "UTC": "UTC",
}


def capture_choices(day, clock, zone_name):
    """Return valid instants for a local time, including both repeated daylight-saving times."""
    if day is None or clock is None:
        return []
    local = datetime.combine(day, clock).replace(tzinfo=None)
    zone = ZoneInfo(zone_name)
    choices = []
    for fold in (0, 1):
        candidate = local.replace(tzinfo=zone, fold=fold)
        round_trip = candidate.astimezone(timezone.utc).astimezone(zone)
        if round_trip.replace(tzinfo=None) == local and not any(
            other.utcoffset() == candidate.utcoffset() for other in choices
        ):
            choices.append(candidate)
    return choices


def render_capture_time():
    """Return an offset-aware timestamp or an actionable validation error."""
    browser_zone = st.context.timezone or "UTC"
    try:
        ZoneInfo(browser_zone)
    except (ZoneInfoNotFoundError, ValueError):
        browser_zone = "UTC"
    zones = list(dict.fromkeys([browser_zone, *TIME_ZONES]))
    time_column, zone_column = st.columns(2)
    with time_column:
        clock = st.time_input("Photo Taken At (optional)", value=None, step=60, format="24h",
                              key="report_capture_clock",
                              help="24-hour time, such as 14:30 · Assumes today in the selected timezone · Leave blank if unknown")
    with zone_column:
        selected_zone = st.selectbox("Time Zone (optional)", zones, index=None, key="report_capture_zone",
                                     placeholder=f"Local Time · {TIME_ZONES.get(browser_zone, browser_zone)}",
                                     format_func=lambda zone: f"{TIME_ZONES.get(zone, zone)} · {zone}"
                                     if zone != "UTC" else "UTC · Coordinated Universal Time",
                                     help=f"Leave blank to use your browser's timezone ({browser_zone})")
    selected_zone = selected_zone or browser_zone
    if clock is None:
        return None, None
    day = datetime.now(ZoneInfo(selected_zone)).date()
    choices = capture_choices(day, clock, selected_zone)
    if not choices:
        return None, "That time falls in a daylight-saving clock change — choose another time"
    if len(choices) == 2:
        occurrence = st.selectbox("Which Time?", [0, 1], index=None, key="report_capture_occurrence",
                                  placeholder="Choose the first or second occurrence",
                                  format_func=lambda index: f"{'First' if index == 0 else 'Second'} occurrence · "
                                  f"{choices[index].tzname()} (UTC{choices[index].strftime('%z')})")
        if occurrence is None:
            return None, "The clocks repeat this time — choose its first or second occurrence"
        selected = choices[occurrence]
    else:
        selected = choices[0]
    return selected.isoformat(), None
