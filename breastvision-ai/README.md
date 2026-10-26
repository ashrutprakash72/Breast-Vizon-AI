# BreastVision AI

An educational computer-vision project that classifies BUSI breast ultrasound images into **benign**, **malignant**, and **normal** dataset classes. It uses EfficientNet-B0 transfer learning, class-weighted loss, staged fine-tuning, early stopping, a held-out test evaluation, and Grad-CAM visualization in Streamlit.

> **Medical safety:** This is a research/portfolio prototype, not a medical device. It must not be used for diagnosis, screening, treatment decisions, or reassurance. Model outputs and Grad-CAM maps are not clinical evidence.

## Features

- Dataset validation and exclusion of BUSI mask files from classification.
- Exact-pixel duplicate detection and removal before splitting.
- Reproducible stratified train/validation/test manifest.
- EfficientNet-B0 ImageNet transfer learning and mixed precision on CUDA.
- Class-weighted cross-entropy, AdamW, learning-rate reduction, and early stopping.
- Held-out test metrics, classification report, confusion matrix, and multiclass ROC-AUC when defined.
- Streamlit upload interface with class scores and Grad-CAM overlay.

## Dataset

Use the BUSI dataset by Al-Dhabyani et al. The original collection contains 780 images across benign, malignant, and normal categories, with masks stored alongside source images. Dataset counts may vary by mirror or cleanup. Do not assume there is an official train/validation/test split.

- Dataset mirror: https://huggingface.co/datasets/SamIndia/BUSI
- Dataset paper: Al-Dhabyani W, Gomaa M, Khaled H, Fahmy A. *Dataset of breast ultrasound images*. Data in Brief. 2020;28:104863. https://doi.org/10.1016/j.dib.2019.104863

Download the dataset and arrange source images as follows. Mask images can remain in the folders; the preparation script excludes filenames containing `mask`.

```text
data/BUSI/
├── benign/
├── malignant/
└── normal/
```

Each folder should contain the original images, not only masks. Folder names must be lowercase as shown.

## Environment setup (Windows 11)

Use Python 3.11 (recommended) and run commands in PowerShell from this project folder.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install PyTorch with the CUDA build supported by your NVIDIA driver by following the official selector at https://pytorch.org/get-started/locally/. For example, choose Windows, Pip, Python, and a CUDA option supported by your driver. Then install remaining dependencies:

```powershell
pip install -r requirements.txt
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

If `torch.cuda.is_available()` is `False`, reinstall PyTorch using the official selector. The code will fall back to CPU, but training will be slower.

## Run the pipeline

### 1. Prepare a split manifest

```powershell
python prepare_data.py --data-dir data/BUSI --output artifacts/splits.csv --seed 42
```

The default split is 70% train, 15% validation, and 15% test. The script removes exact duplicate pixel arrays and stops if an exact duplicate has conflicting labels. Inspect `artifacts/splits.csv` and class counts before training.

### 2. Train

Start conservatively on a 4 GB GPU:

```powershell
python train.py --epochs 20 --batch-size 8 --image-size 224 --patience 5 --warmup-epochs 3
```

If you get CUDA out-of-memory errors, stop the run and retry with `--batch-size 4`. Reduce `--image-size` to `192` only if needed, and use the same size when interpreting the saved checkpoint (the checkpoint stores the size). Pretrained weights download once from Torchvision. Internet access may be required on first run.

The best checkpoint is selected by validation loss and saved to `artifacts/best_model.pth`. Training history is saved to `artifacts/training_history.csv`.

### 3. Evaluate once on the held-out test set

```powershell
python evaluate.py --manifest artifacts/splits.csv --checkpoint artifacts/best_model.pth
```

Outputs include:

- `artifacts/test_metrics.json`
- `artifacts/test_predictions.csv`
- `artifacts/confusion_matrix.png`

Use validation results for model choices. Avoid repeatedly tuning against the test set; doing so turns it into another validation set.

### 4. Launch Streamlit

```powershell
streamlit run app.py
```

Open the local URL printed in the terminal. The application requires `artifacts/best_model.pth`.

## Metrics to report

Report the test-set **macro F1**, **balanced accuracy**, per-class precision/recall/F1, confusion matrix, and one-vs-rest macro ROC-AUC if it can be computed. For a medical-image task, inspect malignant-class recall/sensitivity and false negatives explicitly. Do not report accuracy alone. Never claim clinical utility from a small, single-source dataset.

## Important limitations and advanced research checklist

1. **Patient-level split:** the commonly shared BUSI folder release does not provide a reliable patient ID per image. The included script removes exact pixel duplicates before splitting, but it cannot guarantee patient-level independence or detect every near-duplicate/related view. This can inflate metrics. For stronger claims, obtain patient IDs and split by patient, or evaluate on an independent external dataset.
2. **Dataset quality:** published audits report duplicates, mislabeled/non-breast images, and annotation overlays in BUSI. Inspect the source, document exclusions, and do not silently present results as clinically validated.
3. **Small and imbalanced data:** use class-wise metrics and confidence intervals / repeated splits in a research extension. Avoid aggressive augmentation that changes clinically meaningful anatomy.
4. **Domain shift:** different scanners, clinics, populations, and image preprocessing may change performance.
5. **Calibration:** softmax outputs are not automatically calibrated probabilities.
6. **Explainability:** Grad-CAM is a rough visualization of model sensitivity, not proof that the model reasons correctly or that a highlighted region is a tumor.
7. **Data rights:** review the dataset's license and attribution requirements before redistribution or commercial use. Do not upload the full dataset to GitHub.

## GitHub publishing

Do **not** commit `data/`, trained `.pth` weights, or generated test data unless you have a clear reason and rights to redistribute them. This repository's `.gitignore` excludes them. Include your own plots only after training and evaluation, and clearly label them as your results.

```powershell
git init
git add .
git status
git commit -m "Build BreastVision AI ultrasound classifier"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/breastvision-ai.git
git push -u origin main
```

Create an empty repository on GitHub first and replace `YOUR_USERNAME` with your username. Never commit API keys or private information.

## Suggested repository layout

```text
breastvision-ai/
├── app.py
├── prepare_data.py
├── train.py
├── evaluate.py
├── requirements.txt
├── README.md
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── engine.py
│   ├── evaluate.py
│   ├── gradcam.py
│   └── model.py
├── data/                 # local only; not committed
└── artifacts/            # generated locally; not committed
```

## Reproducibility checklist

- Record the Python, PyTorch, torchvision, CUDA/driver versions.
- Keep `artifacts/splits.csv` locally and record its hash for repeatable runs.
- Record seed, class counts, hyperparameters, and exact checkpoint used for evaluation.
- Keep the test set untouched during architecture/hyperparameter selection.
- For a strong portfolio write-up, compare a small CNN baseline against EfficientNet-B0 using the same splits and metrics.

## Citation

If you use BUSI, cite:

Al-Dhabyani W, Gomaa M, Khaled H, Fahmy A. Dataset of breast ultrasound images. *Data in Brief*. 2020;28:104863. https://doi.org/10.1016/j.dib.2019.104863
