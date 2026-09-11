"""Predict activities from one or more rows of 561 engineered features."""

import argparse

import joblib
import numpy as np

from src.data_loader import ACTIVITY_NAMES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/best_classical.joblib")
    parser.add_argument("--input", required=True, help="Whitespace-delimited or CSV feature file")
    args = parser.parse_args()
    model = joblib.load(args.model)
    try:
        rows = np.loadtxt(args.input, delimiter=",")
    except ValueError:
        rows = np.loadtxt(args.input)
    rows = np.atleast_2d(rows)
    for index, label in enumerate(model.predict(rows), start=1):
        print(f"Row {index}: {ACTIVITY_NAMES[int(label)]}")


if __name__ == "__main__":
    main()

