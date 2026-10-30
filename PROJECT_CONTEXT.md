# Project Context

Snow-Covered Road Classifier

## Project Summary

Citizens submit road photographs with optional location and capture time.
Municipal staff review the suggested visible condition, assign priority,
and record the next action. Citizens can follow their reports within the
current browser session through My Reports.

The tool supports winter road monitoring through structured visual observations.
It does not determine road safety or replace operational assessments.

**Core workflow:** Submit photo → suggest visible condition → staff assessment → monitor, reassess, or close.

### Current direction and implementation status

Version: 1.3
Date: October 30, 2026
Project type: CanAI climate and emergency response hackathon MVP
Status: Local classifier and staff-review prototype implemented;
held-out evaluation and demonstration review remain the next focus.

The current direction is a local four-class road-condition classifier
with citizen photo submission and municipal staff review. The original
classes remain `clear`, `light`, `medium`, and `plowed`. Model suggestions
support staff assessment; priority, action recommendations, and plowing
decisions remain human responsibilities.

The implementation uses a pretrained ResNet-18 with a frozen backbone
and a trainable four-output head. Training reserves approximately
20 percent of training camera IDs for validation. The best checkpoint
is selected by lowest validation loss, without using the test set.

Training saves checkpoint improvements and epoch history during the run.
After the final epoch, it generates and saves one complete training and
validation loss graph. Evaluation processes every test batch before
saving its final metrics report and complete batch-loss graph.

The Streamlit dashboard stores submitted photos and review records locally.
Model suggestions and staff decisions remain separate. Reports remain
available for manual review when inference is unavailable. There is no
automatic retraining, dispatch, authentication, or public deployment.

This document distinguishes implemented behavior from evaluation goals and
historical planning. Setup and run commands are documented in [README.md](README.md).

#### Immediate next steps

1. Review held-out evaluation metrics and confused class pairs.
2. Inspect incorrect predictions and representative phone photos.
3. Audit camera overlap and duplicate leakage before performance claims.
4. Validate dashboard layout, keyboard access, and the demonstration flow.
5. Select an uncertainty threshold using validation evidence if needed.

An uncertainty threshold has not been selected. ImageNet center cropping
is shared across training and inference; it is not road-region detection.
Model scores do not establish road safety or operational urgency.

## Problem Being Solved

Winter road photographs provide useful observations, but each image still
requires interpretation and a clear follow-up decision. Reviewers need to
distinguish unassessed reports, assessed reports requiring attention, and
completed reports while preserving the original evidence and decision history.

The project investigates whether computer vision can reduce repetitive image
sorting by suggesting consistent visible-condition labels, while keeping
staff responsible for assessment, priority, and action. Faster review and
improved consistency are intended benefits, not demonstrated outcomes.

A photograph cannot establish road safety, detect every hazard, or guarantee
that conditions remain unchanged after capture.

## Main Users

Primary reviewers are municipal winter maintenance staff and road-condition
reviewers. Citizens submit observations and follow their session's reports.

The prototype aims to:

- Reduce repetitive image sorting.
- Present consistent condition labels.
- Keep each suggestion connected to its source image.
- Let staff confirm or correct suggestions.

These benefits are hypotheses requiring evaluation.

## Current Scope

#### Solution methodology

1. **Prepare the data.** Use the original four dataset classes: `clear`,
   `light`, `medium`, and `plowed`. Inspect labels and distributions, separate
   training and validation by camera, and audit test overlap and duplicates
   before making performance claims.
2. **Train the classifier.** Adapt pretrained ResNet-18 with a frozen backbone,
   including batch normalization statistics, and a trainable four-output head.
   Reserve approximately 20 percent of training camera IDs for validation
   using seed 42. Select the checkpoint by lowest validation loss. Share
   ImageNet resize, center crop, and normalization across training,
   evaluation, and inference. The crop is not road-region detection.
3. **Capture and analyze reports.** Validate JPEG/PNG uploads, remove embedded
   metadata, and save normalized PNG photos and JSON records locally.
   Keep supplied location and capture time separate; missing values remain
   unknown. If inference fails, retain the report for manual review.
4. **Keep staff in control.** Present the photo and suggestion together.
   Staff confirm or correct conditions, identify unsuitable or inconclusive
   images, assign priority, and save actions and notes. Preserve model output
   separately from staff decisions. Corrections do not automatically retrain
   the model.
5. **Track follow-up.** Separate assessed open reports from closed reports,
   preserve saved review events, and allow reassessment when new evidence
   requires another decision. Plowing recommendations do not dispatch crews.
6. **Validate the result.** Evaluate held-out performance, confused classes,
   representative failures, and CPU pipeline timing. Test persistence,
   review transitions, inference failure, and deletion controls. Assess browser
   layout and keyboard access separately. Operational benefits require
   evidence beyond automated tests.

#### Implemented report lifecycle

| View         | Included reports                                         |
| ------------ | -------------------------------------------------------- |
| Pending      | Reports with no saved staff assessment                    |
| Reviewed     | Assessed reports whose action is not a closing outcome    |
| Reassessment | Reports with the Needs reassessment action                |
| Closed       | Reports marked Resolved or Plowing completed              |
| All          | Every saved report, including closed reports              |

Reassessment is an overlapping follow-up view: its assessed reports also
appear in Reviewed. Closed reports are excluded from Reviewed and the active
Priority Situations panel. Choosing Needs reassessment reopens a closed report.
Resolved does not assert that plowing occurred; Plowing completed records
staff confirmation. Monitoring, reassessment, resolution, and completed
plowing require a short reason.

Only Closed offers Delete report. Confirmation permanently removes that
report's photo, JSON record, notes, and activity history, including its
appearance in My Reports. Deletion cannot be undone. All has no delete control.

Priority Situations contains assessed, open reports with staff-assigned
priority: Urgent assessment, Assess soon, then Routine, with newest reviews
first within a priority. The model does not assign operational urgency.

#### Current interaction and data boundaries

- Report a Condition previews a photo before submission. Analyze Photo saves
  it and opens the result dialog. Repeating an unchanged submission reopens
  its result rather than creating another report.
- Confirm & Next saves the suggested condition while preserving existing
  priority, action, and notes. Review Details opens manual assessment controls.
- My Reports tracks submissions from the current browser session, not an
  authenticated account. Staff notes are visible in this citizen view.
- Shared activity records show receipt, analysis, and saved staff reviews.
  Display times use the viewer's browser timezone, with UTC fallback; stored
  timestamps remain unchanged. Missing historical events are not invented.
- Related reports match location text after case and whitespace normalization.
  They are not inferred duplicates, and reviewing one does not update others.
- Photos persist under `uploads/`; individual JSON records persist under
  `reviews/`. Both are excluded from Git. The workspace selector is navigation,
  not authentication. Keep the demonstration bound to localhost.
- Notify Plow Crew is a demonstration confirmation only. It contacts no crew,
  reserves no equipment, and saves no dispatch status.

#### Model uncertainty and evaluation status

The result dialog uses provisional display rules: a top score below 70 percent
or a top-two gap below 15 percentage points marks the result as possible and
mixed. These are presentation heuristics, not validated confidence thresholds.
They do not change stored `uncertain` values or bypass staff review.
Inference uncertainty remains null until a validation-selected threshold is
configured. Scores are not calibrated probabilities of correctness.

On October 30, 2026, all 47 dashboard tests passed after rerunning with access
to Python temporary folders. This includes closed/reviewed separation,
reopening, and deletion visibility. Automated tests do not establish model
accuracy, operational benefit, or browser accessibility. Held-out evaluation
review, leakage audits, usage-term verification, representative phone-photo
assessment, and manual interface checks remain necessary.

### Minimum viable product requirements

#### Feature 1: Upload a road image

- Accept one JPEG or PNG image at a time.
- Limit uploads to 10 MB.
- Validate actual image decoding and file size.
- Show the image before analysis.
- Let staff replace or remove the image.
- Provide optional location and capture-time fields.
- Mark missing location and capture time as unknown.
- Never infer capture time from upload time.
- Include sample images only when redistribution terms permit.

#### Feature 2: Classify the visible road condition

- Run a trained image classifier on the uploaded image.
- Apply the same preprocessing used during training.
- Return the predicted dataset class and model version.
- Show class scores only when the model produces them.
- Describe scores as model scores, not verified probabilities of correctness.
- Mark low-score predictions as uncertain using a validation-selected threshold.
- Let staff flag unsuitable images or images without a visible road.

A confidence threshold alone cannot reliably detect unrelated images.
The MVP must not promise automatic rejection of every unsuitable image.

#### Feature 3: Display the result for staff review

Display:

- Original image and analyzed road region, when cropping applies.
- Suggested condition label and plain-language definition.
- Model score, when available.
- Review status and model version.
- Supplied location and capture time.
- A short limitation statement.

Staff can:

- Confirm the suggestion.
- Correct the label.
- Mark the image unsuitable or inconclusive.
- Add a short note.

Preserve the original model suggestion separately from the staff decision.
Do not automatically retrain the model using staff corrections.

### Review record

Store these fields for each submitted review:

| Field | Description |
| --- | --- |
| record_id | Unique local record identifier |
| image_reference | Local image identifier, when retained |
| uploaded_at | Upload timestamp |
| captured_at | Supplied capture timestamp or null |
| location | Supplied location or null |
| predicted_class | Original model class identifier |
| class_scores | Model scores, when available |
| model_version | Model artifact identifier |
| label_mapping_version | Class-definition version |
| review_status | Pending, confirmed, corrected, or inconclusive |
| staff_class | Staff-selected class or null |
| unsuitable | Whether staff marked the image unsuitable |
| staff_note | Optional review note |
| reviewed_at | Review timestamp or null |
| inference_status | Whether model analysis was available |
| uncertain | Model uncertainty flag; null without a configured threshold |
| priority | Staff-assigned priority; initially unassigned |
| action | Staff-selected next action or closing outcome |
| activity | Receipt, analysis, and saved staff-review events |

Staff confirmation records a reviewed observation, not a verified declaration of road safety.

## Out of Scope

Exclude these capabilities from the MVP:

- Live camera integration.
- Automatic road closure decisions.
- Safe-route recommendations.
- Snow-depth or friction estimation.
- Reliable black-ice detection.
- Snowplow routing or fleet allocation.
- Storm forecasting.
- Automatic public alerts.
- Production authentication and multi-user operations.

Image classification describes appearance within an image.
It cannot establish current conditions outside the frame or after capture.

### Future extensions

Consider these only after validating the core classifier:

- A map of reviewed observations with capture timestamps.
- Before-and-after comparisons for the same camera view.
- Additional Canadian camera viewpoints.
- Weather context alongside visual observations.
- A reviewed dataset for later model improvement.

## Project Classification

Personal project workspace. Existing project context: CanAI climate and emergency response hackathon MVP.

## Tech Stack

### Proposed default stack

- Python for training and inference.
- PyTorch and torchvision for the classifier.
- Streamlit for the upload and review interface.
- Local JSON storage for review metadata.
- Local execution for the hackathon demonstration.

These choices are implementation defaults, not deployed infrastructure.
No paid API or cloud deployment is required by this plan.
Pretrained weight availability and licensing require verification.

### Model strategy

Use transfer learning: adapt an existing image model to the road-condition classes.
Start with a small pretrained classifier such as ResNet-18.
Replace its final classification layer with four outputs matching the verified Montréal class identifiers.

Train the classification layer first.
Fine-tune additional layers only if time and validation results justify it.
Use a representative subset if full training exceeds the available time.
Record subset selection, class counts, seed, preprocessing, and model version.

A road-region crop may reduce distractions from sky, buildings, and roadside snow.
Use one consistent crop method during training and inference.
Avoid camera-specific crops unless the interface explicitly supports those cameras.

### Inference flow

1. Validate and decode the image.
2. Apply the configured road crop, if required.
3. Resize and normalize using training settings.
4. Run the classifier.
5. Return the class identifier, scores, and model version.
6. Display the suggestion for staff review.
7. Save the review metadata when staff submit their decision.

Do not invent predictions when the model is unavailable.
Clearly identify any separate interface-only demonstration using fixed sample results.

## Repository Structure

### Documented project structure

| Path | Responsibility |
| --- | --- |
| app.py | Single-page dashboard entry point, initialization, and workspace routing |
| components/ | Dashboard layout, report form, staff review, shared suggestions, and model cache |
| train.py | Training and validation entry point |
| evaluate.py | Held-out evaluation and confusion matrix |
| inference.py | Shared preprocessing and model loading |
| config/labels.json | Verified class definitions and display mapping |
| models/ | Model artifact and training metadata |
| data/README.md | Dataset access, terms, and preparation instructions |
| reviews/ | Local review metadata; excluded from version control |
| README.md | Setup, execution, limitations, and demo instructions |

Keep downloaded datasets, uploaded images, and generated reviews outside version control.

## Important Decisions

### Classification labels

**Do not invent dataset label meanings.**

Use the Montréal dataset's original four classes after inspecting its files and documentation.
Record exact class identifiers, definitions, and example images before training.

Create a versioned label mapping with these fields:

| Field | Purpose |
| --- | --- |
| class_id | Original dataset identifier |
| source_label | Original dataset label |
| display_label | Plain-language interface label |
| definition | Visual criteria for the class |
| example_reference | Representative permitted image |

Use the original four Montréal classes throughout training, evaluation, inference, and staff review.
Display labels must preserve the original class meanings.
Do not replace these classes with a new three-class scheme.

Keep `uncertain`, `unsuitable`, and `inconclusive` separate from the trained condition classes.

### Dataset plan

#### Selected dataset: Montréal Snow-Covered Roads Dataset

The project will use this dataset exclusively for MVP training, validation, and testing.

Source: https://github.com/modraa-pixels/Snow-Covered-Roads-Dataset

The repository describes over 40,000 annotated traffic-camera images from Montréal's 2022 winter season.
It describes four road-cover states and includes daytime and nighttime observations.
Its annotations were generated through an automated annotation framework.

Before use:

1. Confirm downloads contain usable images and labels.
2. Inspect the exact four classes and their distributions.
3. Review usage, attribution, and redistribution terms.
4. Manually inspect samples from every class.
5. Identify camera, sequence, and timestamp metadata where available.
6. Document missing metadata and label ambiguity.

Public repository access does not establish unrestricted image reuse.
Do not commit raw dataset images into the project repository by default.

Dataset selection is settled.
Implementation must verify access, usage terms, and exact class definitions.
If access fails, document the blocker and resolve Montréal dataset access before training.

### Instructions for AI coding tools

Implement the three-feature MVP before adding extensions.
Use the Montréal dataset exclusively for MVP training, validation, and testing.
Verify its four classes before writing interface labels.
Keep model inference separate from interface code.
Use one preprocessing implementation for evaluation and inference.
Preserve staff decisions separately from model suggestions.
Never fabricate scores, metrics, labels, timestamps, or trained model artifacts.
Keep assumptions, unavailable dependencies, and changed scope visible in README.md.

### Original ten-hour implementation plan (historical)

This is the historical hackathon schedule, not the current completion status.
See the implementation status and immediate next steps above for current work.

| Time | Work | Completion evidence |
| --- | --- | --- |
| Hours 0–1 | Verify dataset access, labels, terms, and samples | Montréal access confirmed and label mapping |
| Hours 1–2 | Prepare a subset and separated splits | Split manifest and class counts |
| Hours 2–4 | Train the baseline classifier | Saved model and validation results |
| Hours 4–6 | Build upload and inference interface | Working image-to-label flow |
| Hours 6–7 | Add staff review and local records | Confirmed and corrected example records |
| Hours 7–8 | Evaluate held-out images and failure cases | Metrics and confusion matrix |
| Hours 8–9 | Improve usability and document limitations | Runnable README and interface checks |
| Hours 9–10 | Rehearse the demonstration | Repeatable demonstration script |

If Montréal dataset access fails during hour one, prioritize resolving access and document the schedule impact.
If training fails, deliver an honestly labeled interface prototype rather than fabricated model performance.

## Constraints

Target WCAG 2.2 AA in the interface without claiming compliance before assessment.
Support keyboard operation, visible focus, descriptive controls, and readable error messages.
Never communicate condition or review status through color alone.

Avoid collecting names or other unnecessary personal information.
Strip embedded image metadata before retaining uploaded images.
Keep supplied location separate from embedded metadata.
Use permitted, anonymized images for public demonstrations.
The implemented prototype retains submitted photos and reports locally across
app restarts. My Reports membership is session-scoped; stored photos are not.
Check Another Photo resets the submission form without deleting saved reports.
Confirmed deletion is available for individual closed reports only.

## Approval Boundaries

Follow Amanda's applicable Change Control Policy and project instructions. This context document does not grant implementation, deployment, publication, or data-transfer approval.

## Validation

### Data separation

Keep related camera sequences or recording periods within the same split.
Prevent duplicate and near-duplicate images from crossing training and test splits.
Hold out cameras where possible to assess performance on unfamiliar viewpoints.
If metadata prevents reliable grouping, disclose the resulting evaluation limitation.

### Model evaluation

Report:

- Number of images and class counts for each split.
- Overall accuracy.
- Precision and recall for each class.
- Macro F1, which averages class-level precision-recall balance equally.
- A confusion matrix showing mistaken class pairs.
- Performance by lighting condition when metadata supports it.
- Inference time on the actual demonstration hardware.

Compare results against a baseline that always predicts the most common class.
Select uncertainty thresholds using validation data, not the held-out test set.
Do not claim an accuracy target before evaluating the model.

### Interface checks

Verify valid images, corrupt files, oversized files, missing metadata, and unavailable model artifacts.
Verify staff confirmation, correction, unsuitable flags, and saved record integrity.
Confirm keyboard access, readable labels, and text-based status indicators.

### Acceptance criteria

The MVP is complete when:

- Staff can upload a valid JPEG or PNG image.
- The app rejects invalid uploads with a clear message.
- A real model returns a label from the verified class mapping.
- The interface shows the original image and model suggestion together.
- Staff can confirm, correct, or mark the result inconclusive.
- Saved reviews preserve both model output and staff input.
- The app shows a clear unavailable state when inference fails.
- Evaluation includes held-out results and known limitations.
- README instructions let another developer run the demo.

### Demonstration script

1. Explain the manual road-image review problem.
2. Upload a held-out image with a visible road.
3. Show the predicted label and class definition.
4. Confirm or correct the suggestion.
5. Show the preserved prediction and staff decision.
6. Demonstrate an uncertain or unsuitable image.
7. Present actual evaluation results and remaining limitations.

Suggested pitch:

“Our prototype helps municipal staff organize winter road photographs through suggested condition labels and human review.”

## Glossary

| Term | Meaning in this project |
| --- | --- |
| Model score | Classifier output; not a calibrated probability of correctness. |
| Reassessment | An assessed report marked Needs reassessment; also included in Reviewed. |
| Resolved | A closing outcome that does not assert plowing occurred. |
| Plowing completed | A closing outcome recording staff confirmation of completed plowing. |
| My Reports | Reports associated with the current browser session. |
| MVP | Minimum viable product. |
