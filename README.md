# Snow-covered road classifier

Local four-class ResNet-18 baseline for clear, light-covered,
medium-to-heavy-covered, and plowed surfaces. Staff decide whether plowing
is needed. A local Streamlit dashboard supports photo submission and staff
review. No automatic operational decision or dispatch is implemented.

## Environment

Python 3.13 with versions recorded in requirements.txt. The verified local
environment has torch 2.9.0+cu128 and torchvision 0.24.0+cu128; requirements
record their base versions so they do not force CUDA on other machines.
Choose the appropriate PyTorch CPU/CUDA build for your hardware before
installing remaining requirements. Streamlit 1.66.1 is installed in the local
`.venv`, which inherits the existing PyTorch environment. Its dependencies
are installed locally; NumPy and Pillow match requirements.txt.

## Run the dashboard

The trained checkpoint is included at `models/classifier.pt`. After installing
the dependencies, you can run the dashboard without downloading the dataset
or retraining the model. Dataset preparation and training below are only
needed to reproduce or retrain the classifier.

From the project directory:

```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL printed by Streamlit (normally http://127.0.0.1:8501).
Use **Report a condition** to preview and submit a photo with optional location
and capture time. Use **Staff review** to inspect saved reports, confirm or
correct conditions, and record priority, notes, and action recommendations.
The mode selector is navigation, not authentication. Keep this demonstration
bound to localhost.

Photos are normalized to PNG with embedded metadata removed and saved under
`uploads/`; individual JSON records are stored under `reviews/`. Both are
excluded from Git. Capture times must include a timezone offset and remain
unknown when not supplied. Model and staff decisions are saved separately.
There is no automatic retraining or communication with snowplow crews.

The dashboard reads `models/classifier.pt`. If it is missing, being written,
or invalid, the report is retained for manual review without an invented label.
The cached model refreshes after a checkpoint changes. Existing saved reports
are not automatically reclassified when training finishes.

```powershell
.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
```

The inherited system environment has unrelated `fastapi`/`starlette` and
`rembg`/`numpy` requirement conflicts reported by `pip check`. Dashboard imports
and tests are checked separately; this is not a fully isolated clean environment.

## Data and training

### Prepare the local dataset

The dataset contains over 40,000 photos and is not included in this Git
repository. `.gitignore` excludes `/data/` and ZIP archives to keep these
large files out of version control. Each developer must download and extract
the dataset locally before training or evaluation.

1. Obtain the [Snow-Covered Roads Dataset](https://github.com/modraa-pixels/Snow-Covered-Roads-Dataset)
   as a ZIP archive (GitHub **Code > Download ZIP**).
2. Create a `data` folder at this project's root, beside `train.py` and
   `README.md`.
3. Extract the archive into `data/`. The extracted folder should be named
   `Snow-Covered-Roads-Dataset-main`. If the extraction tool adds another
   enclosing folder, move the dataset folder so the layout matches below.
   Keep the original `train` and `test` splits and class folder names.

```text
snowy-roads/
├── README.md
├── train.py
└── data/
    └── Snow-Covered-Roads-Dataset-main/
        └── dataset/
            ├── train/
            │   ├── clear/
            │   ├── light/
            │   ├── medium/
            │   └── plowed/
            └── test/
                ├── clear/
                ├── light/
                ├── medium/
                └── plowed/
```

Each class folder must contain its corresponding image files. The default
dataset root is `data/Snow-Covered-Roads-Dataset-main/dataset`, relative to
the project root. Run this PowerShell check from the project root; all eight
rows should show `Exists` as `True` and an image count greater than zero:

```powershell
foreach ($split in 'train', 'test') {
    foreach ($label in 'clear', 'light', 'medium', 'plowed') {
        $folder = "data/Snow-Covered-Roads-Dataset-main/dataset/$split/$label"
        [pscustomobject]@{
            Folder = $folder
            Exists = Test-Path -LiteralPath $folder -PathType Container
            Images = @(Get-ChildItem -LiteralPath $folder -File -ErrorAction SilentlyContinue |
                Where-Object { $_.Extension -in '.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp', '.ppm', '.pgm' }).Count
        }
    }
}
```

If a folder is missing, check for an extra nested extraction folder. If you
store the dataset elsewhere, pass `--data "path/to/dataset"` to both
`train.py` and `evaluate.py`; that path must contain `train/` and `test/`.
Do not commit images or distribute them without verifying usage terms.

### Run commands and find outputs

Run these commands in PowerShell from the project root, using the existing
`.venv`. Prepare the dataset before training or evaluation.

#### 1. Run automated tests

```powershell
.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
```

Expected output: a pytest pass/fail summary. Tests do not establish model
accuracy. A temporary-folder cleanup permission error has occurred after
successful tests in the current environment.

#### 2. Optional training smoke test

```powershell
.venv\Scripts\python.exe train.py --epochs 1 --limit 64 --output models/smoke.pt
```

Expected output: batch progress, epoch metrics, checkpoint-save messages,
and a final graph-save message.

Files: `models/smoke.pt`, `models/smoke.split.json`,
`models/smoke.history.json`, and `graphs/smoke.loss.png`.

This checks the training pipeline on a small subset. Do not use this
checkpoint for held-out evaluation or dashboard suggestions.

#### 3. Train the classifier

```powershell
.venv\Scripts\python.exe train.py --epochs 3
```

Expected output: device and image counts, batch progress, training and
validation metrics per epoch, and checkpoint-save messages when validation
loss improves.

Files:

- `models/classifier.pt`: checkpoint with the lowest validation loss.
- `models/classifier.split.json`: training and validation image membership.
- `models/classifier.history.json`: metrics for every completed epoch.
- `graphs/classifier.loss.png`: both loss curves across the entire run.

The checkpoint saves when validation loss improves. History saves after
each epoch. The complete loss graph saves once after the final epoch.

#### 4. Evaluate the best checkpoint

```powershell
.venv\Scripts\python.exe evaluate.py
```

Expected output: test-batch progress, final test loss and accuracy, followed
by a JSON metrics report and a graph-save message. Metric values depend on
the checkpoint and test data.

Files saved after all test batches finish:

- `reports/evaluation.json`: loss, accuracy, macro F1, per-class metrics,
  confusion matrix, majority baseline, and CPU pipeline timing.
- `graphs/evaluation.loss.png`: loss for each test batch.

Confusion-matrix rows are actual classes; columns are predicted classes.
The confusion matrix is stored in JSON, not as a separate image.
Evaluation does not update the model. Its loss graph shows variation
across test batches, not learning progress.

#### 5. Predict one image

Replace the example image path with an existing JPEG or PNG of at most 10 MB.

```powershell
.venv\Scripts\python.exe inference.py "path\to\photo.jpg"
```

Expected output: JSON in the terminal containing `predicted_class`,
`class_scores`, `model_version`, `label_mapping_version`, `uncertain`,
and `limitation`. No prediction file is saved automatically.

The class is one of `clear`, `light`, `medium`, or `plowed`.
`uncertain` is null until a validation-selected threshold is configured.
Scores are not calibrated guarantees of correctness.

#### 6. Run the dashboard

```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```

Expected output: a local URL, normally `http://127.0.0.1:8501`.
Press Ctrl+C in the terminal to stop the server.

The dashboard loads `models/classifier.pt`. Submitting a report saves
`uploads/<record_id>.png` and `reviews/<record_id>.json`.
Staff decisions update the JSON record while preserving the original
model suggestion. Previewing an image does not submit a report.

#### Preserve separate experiments

Default paths overwrite generated outputs on later runs. Use matching
custom checkpoint and report paths to preserve another experiment:

```powershell
.venv\Scripts\python.exe train.py --epochs 3 --output models/experiment-02.pt
.venv\Scripts\python.exe evaluate.py --model models/experiment-02.pt --output reports/experiment-02.json
.venv\Scripts\python.exe inference.py "path\to\photo.jpg" --model models/experiment-02.pt
```

Training metadata and graphs use the checkpoint's filename stem. Graphs are saved in `graphs/`.
Evaluation graphs use the report's filename stem. The dashboard continues
to use `models/classifier.pt`.

Use `--help` with any script to inspect its supported options.
The trained checkpoint, its training history, loss graph, split manifest,
and `reports/evaluation.json` are included in Git. Other generated model
artifacts and reports, uploads, and reviews are excluded from Git.

Training reserves 20 percent of training camera IDs for validation with seed
42. It never moves source images or trains on the test set. It freezes the
pretrained backbone, including batch normalization statistics, and trains
the four-output head. The lowest-validation-loss checkpoint is saved with
its class order and preprocessing identifier. Split manifests and training
history accompany the checkpoint. Reusing an output path overwrites those
generated artifacts; use a new output name to preserve an experiment.

After the final epoch, training saves a graph of training and validation loss
to `graphs/classifier.loss.png`. A custom `--output` checkpoint path produces
a matching `.loss.png` file in `graphs/`. The graph includes every
completed epoch and is generated once, after training finishes.

The deterministic ImageNet resize, center crop, and normalization are shared
across training, evaluation, and inference. The center crop is not a detected
road region and may omit relevant content. Cached pretrained weights are
used when present; torchvision otherwise attempts to download them.

Evaluation reports a confusion matrix (true rows, predicted columns),
accuracy, macro F1, class precision/recall, a training-majority baseline,
and CPU pipeline time including decoding and preprocessing. Smoke-test
models are explicitly rejected by held-out evaluation.

After all test batches finish, evaluation saves a graph of each batch's loss
to `graphs/evaluation.loss.png`. The final JSON report stays in `reports/`. A custom
`--output` report path produces a matching `.loss.png` file in `graphs/`. This graph shows
variation across test batches; the model does not learn during evaluation.

## Limitations and next steps

- The supplied train/test split can share cameras and neighboring frames.
  Grouping validation cameras does not repair existing test leakage.
- Exact/near-duplicate audit, complete image decoding audit, and source
  usage-term verification remain necessary before reporting reliable results.
- Training labels were generated automatically and may contain errors.
- Citizen phone photos differ from traffic-camera training images.
- Model scores are not calibrated probabilities of correctness.
- No uncertainty threshold has been selected; inference returns uncertain=null.
  Select a threshold using validation evidence before integrating queue rules.
- No automatic unsuitable-image detector exists. Staff can mark photos unsuitable.
- Browser layout and keyboard accessibility still require visual/manual review.
- Four-class predictions cannot establish road safety, snow depth, or urgency.

The trained checkpoint and the supporting evidence listed above are included
in the repository; other model artifacts and generated reports remain local
and excluded from Git.
The source dataset is described in data/Snow-Covered-Roads-Dataset-main/README.md.
