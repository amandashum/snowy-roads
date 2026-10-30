# Snow-covered road classifier

A local prototype for submitting winter road photos and supporting municipal
staff review. A ResNet-18 classifier suggests one of four visible conditions:
`clear`, `light`, `medium`, or `plowed`. Staff confirm the observation, assign
priority, and decide the next action.

The app does not determine road safety or dispatch snowplows.
See [project context](PROJECT_CONTEXT.md) for the problem statement, solution
methodology, implemented workflows, evaluation goals, and original MVP plan.

## Quick start

Install **Python 3.13** with the Windows Python launcher (`py`) or Python on
PATH. Open PowerShell in the project folder and run:

```powershell
.\setup.ps1
.venv\Scripts\python.exe -m streamlit run app.py
```

Setup creates an isolated `.venv`, installs the pinned dependencies, checks
for conflicts, and verifies that `models/classifier.pt` loads on CPU.
Package installation requires internet access. The checkpoint is included;
you do not need the dataset or retraining to run the dashboard.

Open the local URL printed by Streamlit, normally `http://127.0.0.1:8501`.
Press Ctrl+C in the terminal to stop it. Keep this demonstration on localhost:
the workspace selector provides navigation, not authentication.

## Use the dashboard

1. **Report a Condition:** upload a JPEG or PNG (up to 10 MB and 20 million
   pixels), optionally enter location and capture time, and select Analyze
   Photo. Previewing alone does not submit a report. Location is a searchable
   dropdown: start typing and it suggests matching St. John's streets, even
   with small spelling mistakes. You can still enter an intersection or
   landmark that is not on the list.
2. **My Reports:** see submissions from your current browser session, staff
   decisions, notes, and activity. Refresh status to read the latest review.
   Session membership is not restored after starting a new browser session.
3. **Road Map:** see an embedded MapLibre map of St. John's with a camera icon
   on each street named in a report's location (for example, "Water Street"
   or "water st"). Select a camera to see up to three of the newest photos
   from that street with their condition and review status. Use the
   bottom-right controls to zoom in, zoom out, or enlarge the map to full screen.
4. **Staff Review:** inspect the photo and suggestion, then confirm or correct
   the condition. Use Review Details to set priority, action, and notes.
   Reports remain available for manual review if model analysis fails.

### Review queues

| View         | What it contains                                      |
| ------------ | ----------------------------------------------------- |
| Pending      | Reports awaiting their first staff assessment          |
| Reviewed     | Assessed reports that remain open                      |
| Reassessment | Assessed reports marked Needs reassessment             |
| Closed       | Reports marked Resolved or Plowing completed           |
| All          | Every saved report, including closed reports           |

Closing removes a report from Reviewed and Priority Situations. Reassessment
reopens it. Resolved does not claim plowing occurred; Plowing completed records
staff confirmation. Monitoring, reassessment, and closing require a short reason.

**Delete report is available only in Closed.** Confirmation permanently removes
the photo, record, notes, and activity history. This cannot be undone.

Priority Situations lists open, assessed reports by staff-assigned priority.
Notify Plow Crew shows demo feedback only; no crew is contacted or dispatched.

## Model and data boundaries

- Photos are normalized to PNG with embedded metadata removed and retained in
  `uploads/`. Report JSON records persist in `reviews/`. Both are excluded from
  Git. Resetting the submission form does not delete saved reports.
- Supplied location and capture time remain separate from image metadata.
  Missing values stay unknown. Capture times require a timezone offset.
- Model suggestions remain separate from staff decisions. Changing the model
  does not reclassify saved reports; corrections do not retrain it automatically.
- Model scores are not calibrated confidence percentages. The result dialog's
  mixed-result rules (top score below 70% or top-two gap below 15 percentage
  points) are provisional display heuristics. No validated uncertainty threshold
  is configured, and every submission requires staff review.
- Citizen phone photos differ from the traffic-camera training data. The model
  cannot establish safety, urgency, snow depth, or reliable black-ice detection.
- Staff notes are visible in My Reports. This prototype has no authentication
  or production multi-user controls.
- Road Map shows photos from **all** saved reports, not only the current
  session's. Locations are matched against the local street list in
  `config/streets.json`; report text is never sent to a geocoding service.
  A camera marks an approximate point on the named street, not the photo's
  exact position. Reports without a recognized street, and photos staff marked
  unsuitable, are not mapped.
- The map uses MapLibre GL JS 5.24.0 (BSD-3-Clause), vendored in
  `dashboard/vendor/`. Base-map tiles load from OpenFreeMap (OpenStreetMap
  data), so the map background needs internet access; without it, markers and
  popups still work on a plain background.

## Test the application

```powershell
.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
```

Dashboard tests cover submissions, persistence, staff decisions, queue transitions,
model-unavailable handling, and confirmed deletion. All 47 dashboard tests passed
on October 30, 2026. Road Map tests cover street matching, marker grouping,
popup status text, and workspace rendering. This does not establish model
accuracy or accessibility; browser layout and keyboard operation still need
manual review.

## Reproduce training and evaluation

The dataset is required only for training and evaluation. Obtain the
[Snow-Covered Roads Dataset](https://github.com/modraa-pixels/Snow-Covered-Roads-Dataset)
and extract it so the following path contains both `train/` and `test/`:

```text
data/Snow-Covered-Roads-Dataset-main/dataset/
    train/{clear,light,medium,plowed}/
    test/{clear,light,medium,plowed}/
```

Each class folder must contain images. Preserve the original split and class
names. Check for an extra enclosing extraction folder if files are not found.
Use `--data "path/to/dataset"` with training and evaluation for another location.
Verify usage and redistribution terms before sharing images; raw data and ZIP
archives are excluded from Git.

Run from the project folder:

```powershell
# Optional pipeline smoke test; do not use this checkpoint for real predictions
.venv\Scripts\python.exe train.py --epochs 1 --limit 64 --output models/smoke.pt

# Train the classifier, then evaluate its best checkpoint
.venv\Scripts\python.exe train.py --epochs 3
.venv\Scripts\python.exe evaluate.py

# Analyze one existing JPEG or PNG without the dashboard
.venv\Scripts\python.exe inference.py "path\to\photo.jpg"
```

Training freezes the pretrained backbone and trains the four-output head.
Approximately 20% of training camera IDs are reserved for validation with seed
42. The checkpoint is selected by lowest validation loss. Training, evaluation,
and inference share ImageNet resize, center crop, and normalization; the crop
is not road-region detection. Pretrained weights download if not already cached.

| Output                           | Purpose                                      |
| -------------------------------- | -------------------------------------------- |
| `models/classifier.pt`           | Best validation-loss checkpoint              |
| `models/classifier.split.json`   | Training and validation membership           |
| `models/classifier.history.json` | Metrics for each completed epoch             |
| `graphs/classifier.loss.png`     | Complete training and validation loss curves |
| `reports/evaluation.json`        | Held-out metrics and CPU pipeline timing     |
| `graphs/evaluation.loss.png`     | Loss by test batch, not learning progress    |

Evaluation reports accuracy, macro F1, per-class precision/recall, a majority
baseline, and a confusion matrix (actual rows, predicted columns). It rejects
smoke-test checkpoints and does not update the model. Inference prints JSON
to the terminal without saving a prediction file.

Default paths overwrite earlier generated outputs. Preserve experiments with
matching custom paths:

```powershell
.venv\Scripts\python.exe train.py --epochs 3 --output models/experiment-02.pt
.venv\Scripts\python.exe evaluate.py --model models/experiment-02.pt --output reports/experiment-02.json
```

The supplied test split may share cameras or neighboring frames with training.
Camera-grouped validation does not resolve test leakage. Duplicate audits,
label inspection, usage-term verification, and held-out result review remain
necessary before making reliable performance claims.

## Project structure

```text
app.py                  Streamlit entry point and workspace routing
components/             Submission, history, road map, staff review, and model display
dashboard/              Theme, map template, and vendored MapLibre assets
reports.py              Upload validation and local report persistence
inference.py            Model loading, preprocessing, and prediction
train.py / evaluate.py  Training and held-out evaluation
config/labels.json      Class definitions and display mapping
config/streets.json     Approximate St. John's street points for the Road Map
models/                 Checkpoint and training metadata
```

## Troubleshooting

- **PowerShell cannot find setup:** run `.\setup.ps1` from the project folder.
  If script execution is blocked, use
  `powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1` for that run.
- **Setup rejects an existing environment:** rename the incompatible `.venv`
  and rerun setup. Setup preserves existing environments. Do not distribute
  `.venv`; create one per computer. Dependencies are pinned in
  [requirements.txt](requirements.txt); choose a compatible PyTorch CPU/CUDA build.
- **Dependency conflicts:** the existing development environment has previously
  reported unrelated FastAPI/Starlette and rembg/NumPy conflicts. Use setup's
  isolated environment and investigate `pip check` failures separately.
- **Analysis unavailable:** verify `models/classifier.pt` exists and is a valid
  non-smoke checkpoint. The app retains the report for manual assessment and
  refreshes its model cache when the checkpoint changes.
- **Temporary-folder permission errors in tests:** check access to Python's
  temporary directory. Distinguish setup/cleanup errors from assertion failures.
