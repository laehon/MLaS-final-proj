"""
Run inference with a trained volleyball action classifier.

Does not run training or evaluation. Requires a model file produced by model.py
(default: models/volleyball_classifier.joblib).

Usage:
    python predict.py data/bump_1.csv
    python predict.py path/to/window.csv --model models/volleyball_classifier.joblib
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np

from features import extract_features, load_imu_window_from_csv

DEFAULT_MODEL_PATH = Path("models/volleyball_classifier.joblib")


def load_classifier(model_path: Path | str = DEFAULT_MODEL_PATH) -> dict:
    """Load the saved classifier artifact from disk."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Model file not found: {path.resolve()}\n"
            "Train first with: python model.py"
        )
    return joblib.load(path)


def predict_imu_window(imu_window: np.ndarray, artifact: dict) -> str:
    """
    Predict the action label for one IMU window.

    Args:
        imu_window: Shape (n_samples, 6).
        artifact: Dict loaded via load_classifier().

    Returns:
        Action name, e.g. "bump", "spike", "set", "serve".
    """
    sample_rate = artifact.get("sample_rate_hz", 100)
    features = extract_features(imu_window, sample_rate_hz=sample_rate).reshape(1, -1)
    label_id = artifact["pipeline"].predict(features)[0]
    return artifact["label_encoder"].inverse_transform([label_id])[0]


def predict_csv(csv_path: Path | str, artifact: dict) -> str:
    """Load a CSV repetition and return the predicted label."""
    imu_window = load_imu_window_from_csv(csv_path)
    return predict_imu_window(imu_window, artifact)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict volleyball action from an IMU CSV window."
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="CSV file with columns ax, ay, az, gx, gy, gz",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to saved model (default: {DEFAULT_MODEL_PATH})",
    )
    args = parser.parse_args()

    artifact = load_classifier(args.model)
    label = predict_csv(args.csv_path, artifact)
    print(f"File:   {args.csv_path}")
    print(f"Model:  {args.model.resolve()}")
    print(f"Predicted action: {label}")


if __name__ == "__main__":
    main()
