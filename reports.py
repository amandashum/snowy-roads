"""
This program stores citizen road reports and staff review decisions locally.

It validates JPEG/PNG uploads, removes photo metadata, attempts classification
with the trained checkpoint, and saves each report as a JSON record. Reports
remain available for manual review when inference is unavailable. Staff
condition, priority, action, and notes remain separate from model suggestions.
Atomic JSON replacement protects existing records from incomplete writes.
Photos and records are excluded from version control. This local demonstration
does not provide authentication, public hosting, or snowplow dispatch.
"""

import base64
import binascii
import io
import json
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
RECORDS = ROOT / "reviews"
UPLOADS = ROOT / "uploads"
LABELS = json.loads((ROOT / "config/labels.json").read_text())["classes"]
CLASS_IDS = {item["id"] for item in LABELS}
CLOSED_ACTIONS = {"resolved", "plowing-completed"}


def timestamp():
    """
    Create a UTC timestamp for an upload or staff review.

    Side effects:
        None.

    Returns:
        An ISO-formatted timestamp with a UTC offset.
    """
    return datetime.now(timezone.utc).isoformat()


def record_path(identifier):
    """
    Resolve a canonical report identifier inside the local review directory.

    Side effects:
        Raise ValueError for malformed identifiers.

    Returns:
        The report JSON path.
    """
    if str(uuid.UUID(identifier)) != identifier:
        raise ValueError("Invalid report identifier")
    return RECORDS / f"{identifier}.json"


def save_record(record):
    """
    Save a report atomically so interrupted writes do not replace a valid record.

    Side effects:
        Write a local temporary JSON file and replace its destination.

    Returns:
        None. The report is stored locally.
    """
    path = record_path(record["record_id"])
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2), encoding="utf-8")
    temporary.replace(path)


def create_report(payload, predictor=None):
    """
    Validate a citizen image and preserve its model suggestion for staff review.

    Side effects:
        Write a metadata-free photo and JSON report locally; attempt CPU inference.

    Returns:
        The saved report, including an unavailable state if inference fails.
    """
    received_at = timestamp()
    location = payload.get("location", "")
    captured = payload.get("captured_at") or None
    if not isinstance(location, str) or len(location) > 200:
        raise ValueError("Location must contain at most 200 characters")
    if captured is not None:
        if not isinstance(captured, str) or len(captured) > 64:
            raise ValueError("Invalid capture time")
        parsed = datetime.fromisoformat(captured.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("Capture time must include a timezone")
    encoded = payload.get("image_base64")
    if not isinstance(encoded, str):
        raise ValueError("Choose a JPEG or PNG photo")
    try:
        content = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError("The image could not be decoded") from error
    if not content or len(content) > 10 * 1024 * 1024:
        raise ValueError("Choose an image no larger than 10 MB")
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(content)) as photo:
            if photo.format not in {"JPEG", "PNG"}:
                raise ValueError("Only JPEG and PNG photos are supported")
            if photo.width * photo.height > 20_000_000:
                raise ValueError("Choose an image with at most 20 million pixels")
            photo.load()
            normalized = ImageOps.exif_transpose(photo).convert("RGB")
    identifier = str(uuid.uuid4())
    image_path = UPLOADS / f"{identifier}.png"
    normalized.save(image_path, format="PNG")
    record = {"record_id": identifier, "uploaded_at": received_at,
              "captured_at": captured, "location": location.strip() or None,
              "predicted_class": None, "class_scores": None, "model_version": None,
              "label_mapping_version": "1", "uncertain": None,
              "inference_status": "unavailable", "review_status": "pending",
              "staff_class": None, "priority": "unassigned", "action": "undecided",
              "staff_note": "", "unsuitable": False, "reviewed_at": None,
              "activity": [{"at": received_at, "kind": "received"}]}
    try:
        from inference import predict
        prediction = (predictor or predict)(image_path, ROOT / "models/classifier.pt")
        record.update({key: prediction[key] for key in
                       ("predicted_class", "class_scores", "model_version", "label_mapping_version", "uncertain")})
        record["inference_status"] = "available"
        record["activity"].append({"at": timestamp(), "kind": "ai-analysis",
                                   "condition": record["predicted_class"]})
    except Exception:
        # A missing or still-writing checkpoint must not fabricate a suggestion
        record["inference_status"] = "unavailable"
        record["activity"].append({"at": timestamp(), "kind": "analysis-unavailable"})
    try:
        save_record(record)
    except Exception:
        image_path.unlink(missing_ok=True)
        raise
    return record


def review_report(identifier, payload):
    """
    Record staff decisions without changing the original model suggestion.

    Side effects:
        Validate and replace one local report JSON record.

    Returns:
        The updated report.
    """
    record = json.loads(record_path(identifier).read_text(encoding="utf-8"))
    condition = payload.get("staff_class")
    priority = payload.get("priority")
    action = payload.get("action")
    note = payload.get("staff_note", "")
    if condition not in CLASS_IDS | {"inconclusive", "unsuitable"}:
        raise ValueError("Select a reviewed condition")
    if priority not in {"unassigned", "routine", "assess-soon", "urgent-assessment"}:
        raise ValueError("Select a valid priority")
    if action not in {"undecided", "monitor", "assess-plowing", "recommend-plowing", "no-action",
                      "reassess", "resolved", "plowing-completed"}:
        raise ValueError("Select a valid action")
    if not isinstance(note, str) or len(note) > 2000:
        raise ValueError("Notes must contain at most 2,000 characters")
    if action in {"monitor", "reassess", "resolved", "plowing-completed"} and not note.strip():
        raise ValueError("Add a short reason for monitoring, reassessment, resolution, or completed plowing")
    status = "inconclusive" if condition in {"inconclusive", "unsuitable"} else (
        "confirmed" if condition == record["predicted_class"] else "corrected")
    activity = report_activity(record)
    reviewed_at = timestamp()
    activity.append({"at": reviewed_at, "kind": "staff-review", "condition": condition,
                     "priority": priority, "action": action, "note": note})
    record.update(staff_class=condition, priority=priority, action=action, staff_note=note,
                  unsuitable=condition == "unsuitable", review_status=status, reviewed_at=reviewed_at,
                  activity=activity)
    save_record(record)
    return record


def delete_closed_report(identifier):
    """Permanently delete a closed report's local photo and record.

    Raises ValueError when the report is not closed and OSError when a file
    cannot be accessed. A missing photo is already considered deleted.
    """
    path = record_path(identifier)
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("action") not in CLOSED_ACTIONS:
        raise ValueError("Only closed reports can be deleted")
    photo = UPLOADS / f"{identifier}.png"
    try:
        photo.unlink(missing_ok=True)
    except OSError as error:
        raise OSError("The photo could not be deleted; the report was retained") from error
    try:
        path.unlink()
    except OSError as error:
        raise OSError("The photo was removed, but the report could not be deleted") from error


def report_activity(record):
    """Return saved events, or only known upload and latest-review facts for older records."""
    if "activity" in record:
        return list(record["activity"])
    activity = [{"at": record.get("uploaded_at"), "kind": "received"}]
    if record.get("reviewed_at"):
        activity.append({"at": record["reviewed_at"], "kind": "staff-review",
                         "condition": record.get("staff_class"), "action": record.get("action"),
                         "priority": record.get("priority"), "note": record.get("staff_note", "")})
    return activity
