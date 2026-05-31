"""
Shared IMU feature extraction for training and inference.

Used by model.py (training) and predict.py (deployment).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

IMU_COLUMNS = ["ax", "ay", "az", "gx", "gy", "gz"]
SAMPLE_RATE_HZ = 100
MIN_SAMPLES_PER_REP = 50

N_TIME_FEATURES_PER_AXIS = 5
N_FREQ_FEATURES_PER_AXIS = 4
N_FEATURES_PER_AXIS = N_TIME_FEATURES_PER_AXIS + N_FREQ_FEATURES_PER_AXIS
N_FEATURES = len(IMU_COLUMNS) * N_FEATURES_PER_AXIS


def _time_domain_features(axis: np.ndarray) -> np.ndarray:
    sma = np.sum(np.abs(axis)) / len(axis)
    return np.array([np.mean(axis), np.std(axis), np.min(axis), np.max(axis), sma])


def _frequency_domain_features(axis: np.ndarray, sample_rate_hz: float) -> np.ndarray:
    centered = axis - np.mean(axis)
    spectrum = np.fft.rfft(centered)
    magnitudes = np.abs(spectrum)
    freqs = np.fft.rfftfreq(len(centered), d=1.0 / sample_rate_hz)

    dominant_idx = int(np.argmax(magnitudes[1:])) + 1 if len(magnitudes) > 1 else 0
    dominant_freq = freqs[dominant_idx]

    return np.array(
        [
            dominant_freq,
            np.sum(magnitudes**2),
            np.mean(magnitudes),
            np.std(magnitudes),
        ]
    )


def extract_features(
    imu_window: np.ndarray,
    sample_rate_hz: float = SAMPLE_RATE_HZ,
) -> np.ndarray:
    """
    Build one fixed-size feature vector for a single repetition.

    Args:
        imu_window: Array of shape (n_samples, 6) for ax..gz.

    Returns:
        1D array of shape (N_FEATURES,).
    """
    if imu_window.ndim != 2 or imu_window.shape[1] != len(IMU_COLUMNS):
        raise ValueError(
            f"Expected imu_window shape (n_samples, {len(IMU_COLUMNS)}), "
            f"got {imu_window.shape}"
        )

    features = []
    for col_idx in range(len(IMU_COLUMNS)):
        axis = imu_window[:, col_idx]
        features.append(_time_domain_features(axis))
        features.append(_frequency_domain_features(axis, sample_rate_hz))

    return np.concatenate(features)


def load_imu_window_from_csv(csv_path: Path | str) -> np.ndarray:
    """Load one repetition window from a CSV file."""
    path = Path(csv_path)
    df = pd.read_csv(path, on_bad_lines="skip")

    missing = set(IMU_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")

    for col in IMU_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=IMU_COLUMNS)
    if len(df) < MIN_SAMPLES_PER_REP:
        raise ValueError(
            f"{path.name} has only {len(df)} valid samples "
            f"(need >= {MIN_SAMPLES_PER_REP})"
        )

    return df[IMU_COLUMNS].to_numpy(dtype=np.float64)
