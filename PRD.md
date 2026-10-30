# Snowy Roads — Product Requirements

Status: Draft for review
Date: October 30, 2026
Project: CanAI climate and emergency response hackathon MVP

## 1. Product Summary

Snowy Roads is a local prototype for reporting and reviewing visible winter
road conditions. Citizens submit photographs with optional location and
capture time. A classifier suggests a visible-condition label, and municipal
staff confirm or correct the observation, assign priority, and record follow-up.

The product supports human assessment. It does not establish road safety,
determine operational urgency, or dispatch snowplows.

## 2. Problem

Road photographs provide useful evidence, but reviewers still need to
interpret each image and track the resulting decision.

The product investigates whether consistent condition suggestions and a
structured review workflow can reduce repetitive sorting while preserving
the original evidence and staff judgment. Faster review and improved
consistency remain hypotheses requiring evaluation.

## 3. Users and Needs

| User | Need |
| --- | --- |
| Citizen | Submit an observation and understand its review status |
| Municipal reviewer | Inspect evidence, assess conditions, and record a decision |
| Demonstrator or evaluator | Reproduce the workflow and understand model limitations |

Citizen and staff workspaces are demonstration views. They do not provide
authentication or enforce separate access permissions.

## 4. Goals and Success Measures

- Complete the photo submission and staff review workflow locally.
- Preserve model suggestions separately from staff decisions.
- Keep reports reviewable when model analysis fails.
- Make assessment, follow-up, and closure understandable.
- Evaluate classification performance and representative failure cases.

Candidate measures include submission completion, review task completion,
review effort, correction frequency, held-out classification metrics, and
CPU pipeline timing.

Numeric performance and usability targets have not been agreed. No
operational benefit or production readiness is claimed.

## 5. Current State

The repository contains a Streamlit dashboard, local report persistence,
a four-class ResNet-18 classifier, training and evaluation scripts, and
automated tests.

Implemented workflows include citizen submission, session-based report
history, staff assessment, priority assignment, reassessment, closure,
and confirmed deletion of closed reports.

Remaining validation includes held-out result review, leakage audits,
representative phone-photo assessment, dataset usage-term verification,
and manual interface and accessibility checks.

Requirements below describe expected product behavior. Their inclusion
does not establish that acceptance testing has passed.

## 6. Primary Workflows

### Citizen submission

1. Select one JPEG or PNG.
2. Preview the photograph.
3. Optionally provide location and capture time.
4. Acknowledge the submission terms.
5. Select Analyze Photo.
6. Receive a saved-report confirmation and analysis result, or an
   explanation that manual review is needed.
7. Follow the report through My Reports during the current browser session.

### Staff assessment

1. Open a report from the review queue.
2. Inspect the photograph, supplied details, and model suggestion.
3. Confirm or correct the condition, or mark the image unsuitable or
   inconclusive.
4. Assign priority and record the next action and notes as appropriate.
5. Save the assessment.
6. Monitor, reassess, or close the report as further judgment requires.

## 7. Functional Requirements

All requirements in this section belong to the MVP scope.

| ID | Requirement | Acceptance criteria |
| --- | --- | --- |
| FR-01 | Validate photo submissions | Accept one actual JPEG or PNG up to 10 MB and 20 million pixels. Reject invalid or oversized inputs with an understandable error. Preview alone does not create a report. |
| FR-02 | Capture observation context | Location and capture time are optional. Missing values remain unknown. Capture time requires a timezone offset and is never inferred from upload time. |
| FR-03 | Save and acknowledge submissions | Successful submission creates a report identifier, local photo, and record. Repeating an unchanged submission in the current form reopens its result without creating another report. Resetting the form preserves saved reports. |
| FR-04 | Suggest visible conditions | Use the original classes: clear, light, medium, and plowed. Show the suggested condition and available class scores. Preserve model and label-mapping identifiers in the record. |
| FR-05 | Explain unavailable analysis | When inference is unavailable, retain the photo and report for manual review and show an explicit explanation. Never substitute a fabricated prediction. |
| FR-06 | Support human assessment | Staff can confirm or correct a suggestion and mark unsuitable or inconclusive evidence. Saved staff decisions do not overwrite original model output or trigger automatic retraining. |
| FR-07 | Preserve quick-review decisions | Confirm & Next saves the suggested condition, preserves existing priority, action, and notes, and advances through the available queue. |
| FR-08 | Record priority and follow-up | Staff assign priority and the next action. Monitoring, reassessment, resolution, and completed plowing require a short reason. The model does not assign urgency. |
| FR-09 | Maintain lifecycle views | Pending contains unassessed reports. Reviewed contains assessed open reports. Reassessment contains reports marked Needs reassessment and overlaps Reviewed. Closed contains Resolved or Plowing completed. All includes every saved report. |
| FR-10 | Support closure and reopening | Closing removes reports from Reviewed and active Priority Situations. Needs reassessment reopens a closed report. Resolved does not imply that plowing occurred. |
| FR-11 | Show priority situations | Include assessed, open reports with staff-assigned priority. Order Urgent assessment before Assess soon before Routine, then newest reviews within each priority. |
| FR-12 | Show citizen report history | My Reports shows submissions associated with the current browser session, including staff decisions, notes, and activity. Refresh retrieves saved updates. A new session does not restore membership. |
| FR-13 | Preserve activity | Show recorded receipt, analysis, and staff-review events without inventing missing historical events. Display event times in the viewer's timezone with UTC fallback. |
| FR-14 | Show related observations | Match supplied location text after case and whitespace normalization. Keep reports independent; matching locations do not establish duplicate incidents or apply shared decisions. |
| FR-15 | Restrict destructive deletion | Offer deletion only in Closed and require confirmation. Successful deletion removes the photo, record, notes, and activity, including the report's appearance in My Reports. Cancellation preserves the report. |
| FR-16 | Identify simulated notification | Notify Plow Crew provides demonstration feedback only. It contacts no crew and records no dispatch status. |
| FR-17 | Show reports on a St. John's map | Road Map embeds a MapLibre map with a camera marker on each known St. John's street named in a report's location, showing that street's report count. Selecting a marker opens a popup with up to three of the newest photos and their condition, time, and review status in text. Bottom-right controls zoom and enlarge the map. Location text is matched locally and never sent to a geocoder; the Location field suggests known streets as the user types (fuzzy filtering), still accepts free text, and close misspellings of a known street are matched. Unrecognized locations and photos marked unsuitable are not mapped. |

## 8. AI Behavior and Evaluation

### Decision boundary

The classifier describes visible appearance using the dataset's four
classes. Staff remain responsible for assessment, priority, and actions.

The product must not describe a score as a calibrated probability of
correctness or interpret a condition label as proof of road safety.

### Uncertainty

The current result dialog marks results as possible and mixed when the
top score is below 70% or the top-two gap is below 15 percentage points.
These are provisional presentation heuristics.

A validated inference uncertainty threshold has not been selected.
Stored uncertainty remains null without a configured threshold.
All reports require staff review regardless of their scores.

Low scores alone cannot reliably identify unrelated or unsuitable images.

### Evaluation requirements

- Use the Montréal Snow-Covered Roads Dataset exclusively for MVP model
  training, validation, and testing.
- Preserve the four original class identifiers and their meanings.
- Use camera-grouped training/validation separation.
- Select checkpoints using validation loss, without tuning on test results.
- Share preprocessing across training, evaluation, and inference.
- Report accuracy, macro F1, per-class precision and recall, confusion
  matrix, majority baseline, and CPU pipeline timing.
- Audit camera overlap and duplicate or neighboring-frame leakage before
  making reliable performance claims.
- Inspect confused classes and representative failure examples.
- Assess phone photographs separately because they differ from the
  traffic-camera training data.

Current ImageNet center cropping is preprocessing, not road-region detection.

## 9. Data and Privacy Requirements

- Retain normalized PNG photos locally under uploads/ and JSON records
  under reviews/.
- Remove embedded image metadata from saved photos.
- Keep supplied location and capture time separate from image metadata.
- Exclude uploaded photos, review records, and raw datasets from Git.
- Preserve original model output, model identifiers, staff decisions,
  timestamps, and activity needed to understand each assessment.
- Make clear that staff notes are visible in My Reports.
- Keep the demonstration on localhost because access controls are absent.
- Do not introduce external uploads or communications without approval.

Retention periods, production access controls, and production note
visibility are unresolved. The prototype does not establish those policies.

## 10. Usability and Quality Requirements

- Use plain-language labels and clear primary actions.
- Keep photographs readable without distorting their proportions.
- Explain validation, analysis, persistence, and review failures clearly.
- Provide understandable empty, loading, success, and error states.
- Support keyboard navigation, visible focus, readable contrast, and
  meaningful labels.
- Communicate status through text as well as color.
- Verify narrow-screen layouts and browser interaction manually.
- Preserve saved reports when users switch workspaces.

Accessibility and responsive usability require direct verification;
automated workflow tests alone do not establish conformance.

## 11. Out of Scope

- Public or production deployment.
- Authentication and production multi-user operations.
- Live camera ingestion.
- Automatic road closure or safety decisions.
- Safe-route recommendations.
- Snow-depth or friction estimation.
- Reliable black-ice detection.
- Storm forecasting.
- Automatic alerts, crew dispatch, routing, or fleet allocation.
- Automatic retraining from staff corrections.

Weather context, before-and-after comparisons, and additional camera sources
are possible later extensions, subject to separate review. A street-level
report map (FR-17) is implemented; exact photo positions are not.

## 12. Acceptance and Validation

Demonstration acceptance should include:

1. Submit a valid photo with and without optional context.
2. Reject invalid, oversized, and unsupported uploads.
3. Repeat an unchanged submission without creating a duplicate report.
4. Preserve a report when inference fails.
5. Confirm and correct suggestions while preserving original model output.
6. Save priority, notes, and follow-up actions.
7. Close a report, verify queue changes, and reopen it for reassessment.
8. Confirm that citizen history reflects saved staff decisions.
9. Cancel deletion, then verify confirmed deletion of a closed report.
10. Verify that simulated crew notification contacts nobody.
11. Complete keyboard, layout, and error-state checks.
12. Review evaluation outputs and document remaining model limitations.

Existing automated tests cover parts of these workflows. This PRD does
not record a new test run or acceptance result.

## 13. Risks and Open Decisions

| Risk or decision | Required follow-up |
| --- | --- |
| Camera imagery may not represent citizen phone photos | Review representative phone-photo failures before broader use |
| Dataset leakage may inflate measured performance | Audit camera overlap and duplicates |
| Scores may create false confidence | Keep limitations visible and require human review |
| Workspace navigation may be mistaken for access control | Keep the demonstration local and explain its boundary |
| Staff notes are visible to citizens in the prototype | Decide visibility rules before any production design |
| Road Map shows every saved photo to any viewer | Decide public photo visibility and moderation before any production design |
| Deletion is permanent | Preserve explicit confirmation and clearly describe the consequence |
| Dataset and pretrained-weight terms need verification | Verify usage and redistribution permissions before sharing assets |
| Performance and usability targets are unset | Agree on targets and measurement methods |
| Retention and production ownership are undecided | Resolve before considering deployment |

## 14. Sources and Related Documents

- [Project context](PROJECT_CONTEXT.md): product direction and boundaries.
- [README](README.md): current operation and documented limitations.
- [Label mapping](config/labels.json): current class display names.
- app.py and components/: implemented user workflows.
- reports.py and inference.py: persistence and inference behavior.
- tests/test_dashboard.py and tests/test_classifier.py: existing test coverage.

This document defines product requirements. It does not authorize
implementation changes, deployment, publication, or external data transfer.
