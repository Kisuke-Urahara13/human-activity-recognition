"""Train, compare, and save intermediate-level classical HAR models."""

import argparse
import json
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.data_loader import load_feature_split
from src.evaluate import save_evaluation
from src.preprocessing import scaled_pipeline


def train_models(data_dir: str | Path, output_dir: str | Path = "models") -> dict:
    train = load_feature_split(data_dir, "train")
    test = load_feature_split(data_dir, "test")
    models = {
        "logistic_regression": scaled_pipeline(
            LogisticRegression(max_iter=1500, solver="lbfgs", random_state=42)
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=42
        ),
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    fitted = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(train.X, train.y)
        fitted[name] = model
        results[name] = save_evaluation(test.y, model.predict(test.X), name, "reports")
        print(f"  accuracy={results[name]['accuracy']:.4f}, macro_f1={results[name]['macro_f1']:.4f}")

    best_name = max(results, key=lambda name: results[name]["macro_f1"])
    joblib.dump(fitted[best_name], output_dir / "best_classical.joblib")
    with (output_dir / "model_metadata.json").open("w", encoding="utf-8") as file:
        json.dump({"best_model": best_name, "results": results}, file, indent=2)
    print(f"Saved best model: {best_name}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Path to the unzipped UCI HAR Dataset")
    parser.add_argument("--output-dir", default="models")
    args = parser.parse_args()
    train_models(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()

