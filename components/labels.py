"""Shared display labels for model conditions and staff decisions."""

from reports import LABELS

DISPLAY = {item["id"]: item["display_label"] for item in LABELS}
DISPLAY.update(inconclusive="Inconclusive", unsuitable="Unsuitable photo")
PRIORITIES = {"unassigned": "Unassigned", "routine": "Routine", "assess-soon": "Assess soon",
              "urgent-assessment": "Urgent assessment"}
ACTIONS = {"undecided": "Undecided", "monitor": "Monitor conditions",
           "assess-plowing": "Assess need for plowing", "recommend-plowing": "Recommend plowing",
           "no-action": "No action recommended"}
