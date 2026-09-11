"""Interactive Streamlit dashboard for the UCI HAR project."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "data" / "raw" / "UCI HAR Dataset"
REPORT_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "models"

ACTIVITIES = {
    1: "Walking",
    2: "Walking upstairs",
    3: "Walking downstairs",
    4: "Sitting",
    5: "Standing",
    6: "Laying",
}


st.set_page_config(
    page_title="HAR Model Lab",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root { --ink: #102a43; --teal: #007f78; --cyan: #18a9a0; }
      .stApp { background: linear-gradient(145deg, #f4fbfa 0%, #f8fafc 55%, #eef4fb 100%); }
      [data-testid="stSidebar"] { background: #102a38; }
      [data-testid="stSidebar"] * { color: #f3fbfa !important; }
      [data-testid="stMetric"] {
        background: rgba(255,255,255,.86); border: 1px solid #d8e8e6;
        border-radius: 14px; padding: 14px 16px; box-shadow: 0 8px 28px rgba(16,42,56,.06);
      }
      .har-kicker { color: #007f78; font-size: .82rem; font-weight: 750; letter-spacing: .12em; text-transform: uppercase; }
      .har-title { color: #102a38; font-size: clamp(2rem, 5vw, 3.4rem); line-height: 1.02; font-weight: 800; margin: .3rem 0 .7rem; }
      .har-copy { color: #49636d; font-size: 1.04rem; max-width: 760px; line-height: 1.65; }
      .har-card { background: rgba(255,255,255,.9); border: 1px solid #d8e8e6; border-radius: 16px; padding: 18px; }
      .har-pill { display:inline-block; padding:6px 11px; border-radius:999px; margin:3px; background:#dff6f3; color:#075e59; font-size:.86rem; font-weight:650; }
      h1, h2, h3 { color: #102a38 !important; }
      .stButton > button { border-radius: 10px; border: 0; background: #007f78; color: white; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_metrics() -> dict[str, dict]:
    files = {
        "Logistic Regression": REPORT_DIR / "logistic_regression_metrics.json",
        "Random Forest": REPORT_DIR / "random_forest_metrics.json",
        "1D CNN": REPORT_DIR / "cnn_1d_metrics.json",
    }
    metrics = {}
    for name, path in files.items():
        if path.exists():
            metrics[name] = json.loads(path.read_text(encoding="utf-8"))["summary"]
    return metrics


@st.cache_data
def load_dataset_summary() -> tuple[pd.DataFrame, pd.DataFrame]:
    parts = []
    subjects = []
    for split in ("train", "test"):
        y = np.loadtxt(DATASET_DIR / split / f"y_{split}.txt", dtype=int)
        person = np.loadtxt(DATASET_DIR / split / f"subject_{split}.txt", dtype=int)
        parts.append(pd.DataFrame({"Activity": [ACTIVITIES[value] for value in y], "Split": split.title()}))
        subjects.append(pd.DataFrame({"Subject": person, "Split": split.title()}))
    return pd.concat(parts, ignore_index=True), pd.concat(subjects, ignore_index=True)


@st.cache_resource
def load_classical_model():
    return joblib.load(MODEL_DIR / "best_classical.joblib")


@st.cache_resource
def load_cnn_model():
    from tensorflow import keras

    return keras.models.load_model(MODEL_DIR / "cnn_1d.keras")


def read_numeric_upload(uploaded_file) -> np.ndarray:
    raw = uploaded_file.getvalue()
    for delimiter in (None, ",", ";"):
        try:
            values = np.loadtxt(raw.splitlines(), delimiter=delimiter, dtype=np.float32)
            return np.asarray(values, dtype=np.float32)
        except (TypeError, ValueError):
            continue
    raise ValueError("The file could not be read as whitespace-, comma-, or semicolon-separated numbers.")


def header(title: str, copy: str) -> None:
    st.markdown('<div class="har-kicker">Human activity recognition</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="har-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="har-copy">{copy}</div>', unsafe_allow_html=True)
    st.write("")


with st.sidebar:
    st.markdown("## HAR Model Lab")
    st.caption("UCI smartphone sensor classification")
    page = st.radio(
        "Navigate",
        ["Overview", "Data explorer", "Model comparison", "Predict activity"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("7,352 training samples · 2,947 test samples · 6 activities")


metrics = load_metrics()

if page == "Overview":
    header(
        "Understand movement through sensor data.",
        "Explore a subject-independent HAR experiment that compares interpretable machine learning with a compact neural network.",
    )
    cols = st.columns(4)
    cols[0].metric("Train samples", "7,352")
    cols[1].metric("Test samples", "2,947")
    cols[2].metric("Engineered features", "561")
    cols[3].metric("Raw signal shape", "128 × 9")

    st.subheader("Recognized activities")
    st.markdown(
        "".join(f'<span class="har-pill">{activity}</span>' for activity in ACTIVITIES.values()),
        unsafe_allow_html=True,
    )
    st.write("")
    left, right = st.columns([1.25, 1])
    with left:
        st.subheader("Two complementary pipelines")
        st.markdown(
            """
            <div class="har-card">
              <b>Engineered-feature pipeline</b><br>
              561 time- and frequency-domain measurements → scaling → Logistic Regression or Random Forest.<br><br>
              <b>Raw-signal pipeline</b><br>
              Nine 128-step inertial channels → Conv1D → pooling → Conv1D → global pooling → six-class softmax.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.subheader("Evaluation design")
        st.info(
            "The official split keeps different participants in training and testing, preventing the model from memorizing a person's motion pattern."
        )

elif page == "Data explorer":
    header("Inspect the dataset.", "Compare activity balance and confirm that train and test participants remain separate.")
    try:
        activity_data, subject_data = load_dataset_summary()
    except FileNotFoundError:
        st.error(f"Dataset not found at {DATASET_DIR}")
        st.stop()
    left, right = st.columns([1.4, 1])
    with left:
        st.subheader("Samples by activity")
        counts = activity_data.groupby(["Activity", "Split"]).size().unstack(fill_value=0)
        st.bar_chart(counts, color=["#007f78", "#ff9f43"])
    with right:
        st.subheader("Subject separation")
        summary = subject_data.groupby("Split")["Subject"].agg(["nunique", "min", "max"])
        summary.columns = ["Unique subjects", "First ID", "Last ID"]
        st.dataframe(summary, width="stretch")
        overlap = set(subject_data.loc[subject_data.Split == "Train", "Subject"]) & set(
            subject_data.loc[subject_data.Split == "Test", "Subject"]
        )
        st.success("No subject overlap" if not overlap else f"Overlap detected: {sorted(overlap)}")
    st.subheader("Class counts")
    st.dataframe(counts.assign(Total=counts.sum(axis=1)), width="stretch")

elif page == "Model comparison":
    header("Compare model behavior.", "Review held-out performance and inspect where each approach confuses similar activities.")
    if not metrics:
        st.warning("No evaluation reports are available yet.")
        st.stop()
    frame = pd.DataFrame(metrics).T.rename(
        columns={
            "accuracy": "Accuracy",
            "macro_precision": "Macro precision",
            "macro_recall": "Macro recall",
            "macro_f1": "Macro F1",
        }
    )
    best = frame["Macro F1"].idxmax()
    cols = st.columns(3)
    for column, (name, row) in zip(cols, frame.iterrows()):
        column.metric(name, f"{row['Accuracy']:.2%}", "Best overall" if name == best else None)
    st.subheader("Held-out test metrics")
    st.dataframe(frame.style.format("{:.2%}").highlight_max(axis=0, color="#c8eee9"), width="stretch")
    st.bar_chart(frame[["Accuracy", "Macro F1"]], color=["#007f78", "#ff9f43"])

    st.subheader("Confusion matrix")
    selected = st.selectbox("Model", list(metrics))
    image_names = {
        "Logistic Regression": "logistic_regression_confusion_matrix.png",
        "Random Forest": "random_forest_confusion_matrix.png",
        "1D CNN": "cnn_1d_confusion_matrix.png",
    }
    image_path = REPORT_DIR / "figures" / image_names[selected]
    if image_path.exists():
        st.image(str(image_path), width="stretch")
    if selected == "1D CNN":
        st.caption("The CNN uses raw signals rather than the 561 engineered features; its simpler representation trades some accuracy for an end-to-end learning pipeline.")

else:
    header("Predict an activity.", "Upload one or more numeric sensor samples and run either the best classical model or the 1D CNN.")
    model_choice = st.segmented_control(
        "Prediction model", ["Logistic Regression", "1D CNN"], default="Logistic Regression"
    )
    if model_choice == "Logistic Regression":
        st.markdown("Upload a `.txt` or `.csv` file with **561 feature columns per row**.")
        uploaded = st.file_uploader("Feature file", type=["txt", "csv"], key="classical-upload")
        if uploaded:
            try:
                rows = np.atleast_2d(read_numeric_upload(uploaded))
                if rows.shape[1] != 561:
                    st.error(f"Expected 561 columns, but found {rows.shape[1]}.")
                else:
                    row_index = st.number_input("Sample row", 1, len(rows), 1) - 1
                    model = load_classical_model()
                    probabilities = model.predict_proba(rows[[row_index]])[0]
                    predicted = int(model.classes_[int(np.argmax(probabilities))])
                    st.success(f"Predicted activity: **{ACTIVITIES[predicted]}**")
                    probability_frame = pd.DataFrame(
                        {"Confidence": probabilities},
                        index=[ACTIVITIES[int(label)] for label in model.classes_],
                    ).sort_values("Confidence", ascending=False)
                    st.bar_chart(probability_frame, color="#007f78")
                    st.dataframe(probability_frame.style.format("{:.2%}"), width="stretch")
            except Exception as exc:
                st.error(f"Could not process the file: {exc}")
    else:
        st.markdown("Upload a `.txt` or `.csv` file containing exactly **128 rows × 9 signal columns**.")
        uploaded = st.file_uploader("Raw sensor window", type=["txt", "csv"], key="cnn-upload")
        if uploaded:
            try:
                window = np.asarray(read_numeric_upload(uploaded), dtype=np.float32)
                if window.shape != (128, 9):
                    st.error(f"Expected shape (128, 9), but found {window.shape}.")
                else:
                    model = load_cnn_model()
                    probabilities = model.predict(window[np.newaxis, ...], verbose=0)[0]
                    predicted = int(np.argmax(probabilities)) + 1
                    st.success(f"Predicted activity: **{ACTIVITIES[predicted]}**")
                    probability_frame = pd.DataFrame(
                        {"Confidence": probabilities}, index=list(ACTIVITIES.values())
                    ).sort_values("Confidence", ascending=False)
                    st.bar_chart(probability_frame, color="#ff9f43")
                    st.dataframe(probability_frame.style.format("{:.2%}"), width="stretch")
            except Exception as exc:
                st.error(f"Could not process the file: {exc}")


