"""
Volleyball IMU action classifier — training pipeline.

Classify four actions (bump, spike, set, serve) from wrist-mounted MPU-6050
data using hand-crafted time and frequency features and sklearn classifiers.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
    cross_val_predict,
    cross_validate,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from features import (
    IMU_COLUMNS,
    N_FEATURES,
    SAMPLE_RATE_HZ,
    extract_features,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("./data")
CLASSES = ["bump", "spike", "set", "serve"]
RANDOM_STATE = 42
# Standalone model file for inference (see predict.py). Not mixed with eval plots.
MODEL_SAVE_PATH = Path("models/volleyball_classifier.joblib")
ARTIFACTS_DIR = Path("./artifacts")
N_CV_FOLDS = 5
N_GRIDSEARCH_FOLDS = 3
# Parallel workers can interleave stdout on Windows; keep at 1 for clean console output.
N_JOBS = 1
TEST_SIZE = 0.2

FILENAME_PATTERN = re.compile(r"^(bump|spike|set|serve)_\d+\.csv$", re.IGNORECASE)
MIN_SAMPLES_PER_REP = 50

def _section(title: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}\n{title}\n{bar}", flush=True)

def load_dataset(data_dir: Path | str) -> list[tuple[np.ndarray, str]]:
    """Load all CSV files and attach labels."""
    data_path = Path(data_dir)
    if not data_path.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    reps: list[tuple[np.ndarray, str]] = []

    for csv_path in sorted(data_path.glob("*.csv")):
        label = csv_path.stem.split("_")[0].lower()

        if label not in CLASSES:
            logger.warning("Skipping unknown label '%s' in %s", label, csv_path.name)
            continue

        try:
            df = pd.read_csv(csv_path, on_bad_lines="skip")
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
            logger.warning("Skipping corrupt file %s: %s", csv_path.name, exc)
            continue

        missing = set(IMU_COLUMNS) - set(df.columns)
        if missing:
            logger.warning("Skipping %s: missing columns %s", csv_path.name, missing)
            continue

        for col in IMU_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=IMU_COLUMNS)
        if len(df) < MIN_SAMPLES_PER_REP:
            logger.warning(
                "Skipping %s: only %d valid samples (need >= %d)",
                csv_path.name,
                len(df),
                MIN_SAMPLES_PER_REP,
            )
            continue

        imu_window = df[IMU_COLUMNS].to_numpy(dtype=np.float64)
        reps.append((imu_window, label))

    if not reps:
        raise ValueError(f"No valid repetitions found in {data_path}")

    logger.info("Loaded %d repetitions from %s", len(reps), data_path)
    return reps


def inspect_dataset(reps: list[tuple[np.ndarray, str]]) -> None:
    """Print class balance and per-repetition window length statistics."""
    labels = [label for _, label in reps]
    lengths = [window.shape[0] for window, _ in reps]

    counts = pd.Series(labels).value_counts().reindex(CLASSES, fill_value=0)
    _section("Dataset summary")
    print("Repetitions per class:", flush=True)
    print(counts.to_string(), flush=True)

    length_stats = pd.Series(lengths).describe()
    print("\nWindow length (samples) statistics:", flush=True)
    print(length_stats.to_string(), flush=True)

    median_len = int(length_stats["50%"])
    short_reps = [length for length in lengths if length < 0.9 * median_len]
    if short_reps:
        print(
            f"\nWarning: {len(short_reps)} repetition(s) shorter than 90% of "
            f"median length ({median_len} samples)."
        )


def build_feature_matrix(
    reps: list[tuple[np.ndarray, str]],
) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """
    Apply extract_features to every repetition.

    Returns:
        X: Feature matrix of shape (n_reps, N_FEATURES).
        y: Encoded integer labels.
        label_encoder: Fitted LabelEncoder mapping integers to class names.
    """
    X = np.vstack([extract_features(window) for window, _ in reps])
    raw_labels = np.array([label for _, label in reps])

    label_encoder = LabelEncoder()
    label_encoder.fit(CLASSES)
    y = label_encoder.transform(raw_labels)

    logger.info("Feature matrix shape: %s (%d features per rep)", X.shape, N_FEATURES)
    return X, y, label_encoder


def make_pipeline(estimator) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", estimator),
        ]
    )


def get_models() -> dict[str, object]:
    """Return the four proposal classifiers (unfitted)."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(random_state=RANDOM_STATE),
        "svm": SVC(random_state=RANDOM_STATE),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=500,
            random_state=RANDOM_STATE,
        ),
    }


def get_param_grid(model_name: str) -> dict[str, list]:
    """Hyperparameter grid for GridSearchCV (Pipeline param names use clf__ prefix)."""
    grids = {
        "logistic_regression": {
            "clf__C": [0.1, 1.0, 10.0],
            "clf__solver": ["lbfgs"],
        },
        "random_forest": {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [None, 10, 20],
        },
        "svm": {
            "clf__C": [0.1, 1.0, 10.0],
            "clf__kernel": ["rbf", "linear"],
            "clf__gamma": ["scale", "auto"],
        },
        "mlp": {
            "clf__hidden_layer_sizes": [(32, 16), (64, 32)],
            "clf__alpha": [1e-4, 1e-3],
        },
    }
    if model_name not in grids:
        raise ValueError(f"No param grid defined for model: {model_name}")
    return grids[model_name]


def evaluate_with_cv(
    X: np.ndarray,
    y: np.ndarray,
    models: dict[str, object],
    n_splits: int = N_CV_FOLDS,
) -> tuple[dict[str, dict[str, float]], str]:
    """
    Compare all models with k-fold cross-validation.

    Returns:
        scores: Per-model mean/std test accuracy.
        best_name: Model with highest mean accuracy.
    """
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scores: dict[str, dict[str, float]] = {}

    _section(f"{n_splits}-fold stratified cross-validation (model comparison)")
    header = f"{'Model':<22} {'Mean acc':>10} {'Std':>8}"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    rows = []
    for name, estimator in models.items():
        pipeline = make_pipeline(estimator)
        result = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring="accuracy",
            return_train_score=False,
            n_jobs=N_JOBS,
        )
        mean_acc = float(np.mean(result["test_score"]))
        std_acc = float(np.std(result["test_score"]))
        scores[name] = {"mean_accuracy": mean_acc, "std_accuracy": std_acc}
        rows.append(f"{name:<22} {mean_acc:>10.4f} {std_acc:>8.4f}")

    print("\n".join(rows), flush=True)

    best_name = max(scores, key=lambda name: scores[name]["mean_accuracy"])
    print(f"\nBest model by mean CV accuracy: {best_name}", flush=True)
    return scores, best_name


def tune_best_model(
    X: np.ndarray,
    y: np.ndarray,
    model,
    param_grid: dict[str, list],
    cv: int = N_GRIDSEARCH_FOLDS,
) -> Pipeline:
    """Run GridSearchCV on the best-performing model from CV comparison."""
    pipeline = make_pipeline(model)
    search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        refit=True,
        n_jobs=N_JOBS,
    )
    search.fit(X, y)
    _section("Hyperparameter tuning (GridSearchCV)")
    print(f"Best hyperparameters: {search.best_params_}", flush=True)
    print(f"Best GridSearchCV accuracy: {search.best_score_:.4f}", flush=True)
    return search.best_estimator_


def report_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    title: str = "Confusion matrix",
) -> float:
    """Print metrics and display a confusion matrix."""
    accuracy = accuracy_score(y_true, y_pred)
    _section(title)
    print(f"Overall accuracy: {accuracy:.4f}\n", flush=True)
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
    )
    print(report, flush=True)

    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    bump_idx = class_names.index("bump")
    set_idx = class_names.index("set")
    bump_as_set = cm[bump_idx, set_idx]
    set_as_bump = cm[set_idx, bump_idx]
    print(
        f"Set vs bump confusion: {bump_as_set} bump->set, "
        f"{set_as_bump} set->bump",
        flush=True,
    )
    return accuracy


def evaluate_final_model(
    pipeline: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str],
    n_splits: int = N_CV_FOLDS,
) -> Pipeline:
    """Evaluate on a held-out test set, then return a fitted pipeline."""
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    y_pred_oof = cross_val_predict(pipeline, X_train, y_train, cv=cv, n_jobs=N_JOBS)
    report_results(
        y_train,
        y_pred_oof,
        class_names,
        title=f"Train set: out-of-fold predictions ({n_splits}-fold CV)",
    )

    pipeline.fit(X_train, y_train)

    y_pred_test = pipeline.predict(X_test)
    report_results(
        y_test,
        y_pred_test,
        class_names,
        title="Test set: held-out evaluation",
    )
    return pipeline


def save_classifier(
    pipeline: Pipeline,
    label_encoder: LabelEncoder,
    path: Path | str = MODEL_SAVE_PATH,
) -> Path:
    """
    Save the trained classifier to a standalone file for inference.

    Use predict.py to load this file without running training:
        python predict.py data/bump_1.csv
    """
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "pipeline": pipeline,
        "label_encoder": label_encoder,
        "classes": list(label_encoder.classes_),
        "imu_columns": IMU_COLUMNS,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "n_features": N_FEATURES,
    }
    joblib.dump(artifact, save_path)
    print(f"Saved classifier to {save_path.resolve()}", flush=True)
    print("Run inference with: python predict.py <path/to/window.csv>", flush=True)
    return save_path


def main() -> None:
    reps = load_dataset(DATA_DIR)
    inspect_dataset(reps)

    X, y, label_encoder = build_feature_matrix(reps)
    class_names = list(label_encoder.classes_)

    _section("Train/test split")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(
        f"Train size: {len(y_train)} reps | Test size: {len(y_test)} reps "
        f"(test_size={TEST_SIZE})",
        flush=True,
    )

    models = get_models()
    scores, best_name = evaluate_with_cv(X_train, y_train, models, n_splits=N_CV_FOLDS)

    param_grid = get_param_grid(best_name)
    tuned_pipeline = tune_best_model(
        X_train,
        y_train,
        models[best_name],
        param_grid,
        cv=N_GRIDSEARCH_FOLDS,
    )

    fitted_pipeline = evaluate_final_model(
        tuned_pipeline,
        X_train,
        y_train,
        X_test,
        y_test,
        class_names,
        n_splits=N_CV_FOLDS,
    )

    save_classifier(fitted_pipeline, label_encoder)

    _section("Training summary")
    for name, result in scores.items():
        marker = " <-- best" if name == best_name else ""
        print(
            f"  {name}: {result['mean_accuracy']:.4f} "
            f"(+/- {result['std_accuracy']:.4f}){marker}",
            flush=True,
        )
    print(f"Selected model: {best_name}", flush=True)


if __name__ == "__main__":
    # Required on Windows when using joblib/sklearn multiprocessing.
    import multiprocessing

    multiprocessing.freeze_support()
    main()
