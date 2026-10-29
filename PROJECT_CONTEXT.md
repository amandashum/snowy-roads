# Snow-Covered Road Classifier

## Current direction and implementation status

Version: 1.2
Date: October 29, 2026
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

The remaining sections retain the MVP requirements and original planning
context. Implemented behavior and run procedures are documented in README.md.

### Immediate next steps

1. Review held-out evaluation metrics and confused class pairs.
2. Inspect incorrect predictions and representative phone photos.
3. Audit camera overlap and duplicate leakage before performance claims.
4. Validate dashboard layout, keyboard access, and the demonstration flow.
5. Select an uncertainty threshold using validation evidence if needed.

An uncertainty threshold has not been selected. ImageNet center cropping
is shared across training and inference; it is not road-region detection.
Model scores do not establish road safety or operational urgency.

## 1. Product summary

Municipal staff upload a road image, receive a suggested visible-condition classification, and review the result.

The tool supports winter road monitoring through structured visual observations.
It does not determine road safety or replace operational assessments.

**Core workflow:** Upload road image → classify visible condition → display result for staff review.

## 2. Problem statement

Municipal staff must inspect road photographs to understand visible snow conditions.
Manual inspection can slow the organization of winter road observations.

How can computer vision suggest consistent road-condition labels while keeping staff responsible for the final observation?

## 3. Target users and value

Primary users are municipal winter maintenance staff and road-condition reviewers.

The prototype aims to:

- Reduce repetitive image sorting.
- Present consistent condition labels.
- Keep each suggestion connected to its source image.
- Let staff confirm or correct suggestions.

These benefits are hypotheses requiring evaluation.

## 4. Minimum viable product

### Feature 1: Upload a road image

- Accept one JPEG or PNG image at a time.
- Limit uploads to 10 MB.
- Validate actual image decoding and file size.
- Show the image before analysis.
- Let staff replace or remove the image.
- Provide optional location and capture-time fields.
- Mark missing location and capture time as unknown.
- Never infer capture time from upload time.
- Include sample images only when redistribution terms permit.

### Feature 2: Classify the visible road condition

- Run a trained image classifier on the uploaded image.
- Apply the same preprocessing used during training.
- Return the predicted dataset class and model version.
- Show class scores only when the model produces them.
- Describe scores as model scores, not verified probabilities of correctness.
- Mark low-score predictions as uncertain using a validation-selected threshold.
- Let staff flag unsuitable images or images without a visible road.

A confidence threshold alone cannot reliably detect unrelated images.
The MVP must not promise automatic rejection of every unsuitable image.

### Feature 3: Display the result for staff review

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

## 5. Classification labels

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

## 6. Dataset plan

### Selected dataset: Montréal Snow-Covered Roads Dataset

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

## 7. Technical approach

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

## 8. Suggested project structure

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

## 9. Review record

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

Staff confirmation records a reviewed observation, not a verified declaration of road safety.

## 10. Validation plan

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

## 11. Acceptance criteria

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

## 12. Ten-hour implementation plan

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

## 13. Demonstration script

1. Explain the manual road-image review problem.
2. Upload a held-out image with a visible road.
3. Show the predicted label and class definition.
4. Confirm or correct the suggestion.
5. Show the preserved prediction and staff decision.
6. Demonstrate an uncertain or unsuitable image.
7. Present actual evaluation results and remaining limitations.

Suggested pitch:

“Our prototype helps municipal staff organize winter road photographs through suggested condition labels and human review.”

## 14. Scope boundaries

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

## 15. Accessibility and privacy

Target WCAG 2.2 AA in the interface without claiming compliance before assessment.
Support keyboard operation, visible focus, descriptive controls, and readable error messages.
Never communicate condition or review status through color alone.

Avoid collecting names or other unnecessary personal information.
Strip embedded image metadata before retaining uploaded images.
Keep supplied location separate from embedded metadata.
Use permitted, anonymized images for public demonstrations.
Default to session-only image handling unless retention is explicitly enabled.
Clearly explain local review storage and provide a reset option.

## 16. Future extensions

Consider these only after validating the core classifier:

- A queue of images awaiting review.
- A map of reviewed observations with capture timestamps.
- Before-and-after comparisons for the same camera view.
- Additional Canadian camera viewpoints.
- Weather context alongside visual observations.
- A reviewed dataset for later model improvement.

## 17. Instructions for AI coding tools

Implement the three-feature MVP before adding extensions.
Use the Montréal dataset exclusively for MVP training, validation, and testing.
Verify its four classes before writing interface labels.
Keep model inference separate from interface code.
Use one preprocessing implementation for evaluation and inference.
Preserve staff decisions separately from model suggestions.
Never fabricate scores, metrics, labels, timestamps, or trained model artifacts.
Keep assumptions, unavailable dependencies, and changed scope visible in README.md.
