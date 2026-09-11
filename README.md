# Human Activity Recognition

This project classifies six daily activities from smartphone accelerometer and
gyroscope measurements in the UCI Human Activity Recognition dataset.

## Models

- Logistic Regression: interpretable baseline
- Random Forest: nonlinear classical model with feature importance
- 1D CNN: deep-learning model for raw sensor windows

The classical pipeline uses the dataset's 561 engineered features. The CNN uses
128-step windows with nine inertial-signal channels. The official subject-based
train/test split is retained to avoid leakage between people.

## Model performance

All results below were measured on the official held-out test split.

| Model | Input representation | Accuracy | Macro precision | Macro recall | Macro F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| **Logistic Regression** | 561 engineered features | **95.55%** | **95.82%** | **95.42%** | **95.54%** |
| Random Forest | 561 engineered features | 92.64% | 92.79% | 92.29% | 92.43% |
| 1D CNN | Raw signals (`128 × 9`) | 89.58% | 90.08% | 89.73% | 89.56% |

Logistic Regression achieved the strongest accuracy and macro F1. The supplied
UCI HAR features already contain informative time- and frequency-domain
measurements, allowing a linear classifier to perform particularly well. The
compact 1D CNN learns directly from raw sensor windows and therefore avoids
manual feature engineering, but achieved lower accuracy in this experiment.

This is a comparison of two related pipelines rather than a perfectly
like-for-like algorithm comparison: Logistic Regression and Random Forest use
engineered features, while the CNN uses raw inertial signals.

## Project structure

```text
app.py                     Streamlit dashboard and prediction interface
data/raw/                  Place the unzipped UCI HAR Dataset here
data/processed/            Generated intermediate data (gitignored)
models/                    Saved fitted models
reports/figures/           Evaluation plots
src/data_loader.py         Dataset loading and validation
src/preprocessing.py       Reusable preprocessing
src/train_classical.py     Baseline training and model selection
src/train_cnn.py           1D CNN training
src/evaluate.py            Metrics and confusion matrix
src/predict.py             Classical-model inference
tests/                     Small pipeline tests
```

## Setup

Use Python 3.10 or 3.11.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-cnn.txt
```

Download and unzip the UCI HAR Dataset so this file exists:

```text
data/raw/UCI HAR Dataset/train/X_train.txt
```

Dataset source: https://www.kaggle.com/datasets/drsaeedmohsen/ucihar-dataset

## Run

Launch the Streamlit dashboard:

```powershell
streamlit run app.py
```

Train and compare the classical models:

```powershell
python -m src.train_classical --data-dir "data/raw/UCI HAR Dataset"
```

Train the CNN:

```powershell
python -m src.train_cnn --data-dir "data/raw/UCI HAR Dataset"
```

Predict one or more feature rows from a text or CSV file:

```powershell
python -m src.predict --model models/best_classical.joblib --input sample.txt
```

Run tests:

```powershell
pytest
```

## Evaluation

The training scripts report accuracy, macro precision, macro recall, and macro
F1. They also save classification reports and confusion matrices under
`reports/`. The Streamlit dashboard presents these results interactively and
supports predictions from both engineered features and raw sensor windows.

