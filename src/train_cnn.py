"""Train an approachable 1D CNN on the nine raw UCI HAR signals."""

import argparse
from pathlib import Path

import numpy as np

from src.data_loader import load_raw_split
from src.evaluate import save_evaluation


def build_model(input_shape, class_count: int = 6):
    try:
        from tensorflow import keras
    except ImportError as exc:
        raise RuntimeError("Install requirements-cnn.txt to train the CNN") from exc
    return keras.Sequential(
        [
            keras.layers.Input(shape=input_shape),
            keras.layers.Conv1D(64, 5, activation="relu", padding="same"),
            keras.layers.MaxPooling1D(2),
            keras.layers.Conv1D(128, 3, activation="relu", padding="same"),
            keras.layers.GlobalAveragePooling1D(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(class_count, activation="softmax"),
        ]
    )


def train_cnn(data_dir: str | Path, epochs: int = 30, output_dir: str | Path = "models"):
    from tensorflow import keras

    train = load_raw_split(data_dir, "train")
    test = load_raw_split(data_dir, "test")
    model = build_model(train.X.shape[1:])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    callbacks = [keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
    model.fit(
        train.X,
        train.y - 1,
        validation_split=0.2,
        epochs=epochs,
        batch_size=64,
        callbacks=callbacks,
        verbose=2,
    )
    predictions = np.argmax(model.predict(test.X, verbose=0), axis=1) + 1
    metrics = save_evaluation(test.y, predictions, "cnn_1d", "reports")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir / "cnn_1d.keras")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--output-dir", default="models")
    args = parser.parse_args()
    print(train_cnn(args.data_dir, args.epochs, args.output_dir))


if __name__ == "__main__":
    main()

