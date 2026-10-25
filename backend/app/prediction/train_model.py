import json
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from statistics import mean, median
from typing import Any

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app.core.config import DATA_DIR
from app.core.database import Database, database
from app.prediction.risk_estimator import estimate_risk

FEATURE_COLUMNS = (
    "cpu_current",
    "cpu_mean",
    "cpu_trend",
    "memory_current_mb",
    "memory_growth_mb_s",
    "memory_acceleration_mb_s2",
    "missed_heartbeats",
    "abnormal_samples",
    "responsive",
    "recent_failures",
    "recent_restarts",
    "history_samples",
)
DEFAULT_MODEL_PATH = DATA_DIR / "risk_model.joblib"
DEFAULT_HORIZON_S = 30
HANG_SAMPLE_COUNT = 3


def _failure_times(measurements: list[dict[str, Any]]) -> list[float]:
    events: list[float] = []
    stale_streak = 0
    stale_start: float | None = None
    for sample in measurements:
        timestamp = float(sample["timestamp"])
        if sample["process_state"] == "failed":
            events.append(timestamp)
        if not bool(sample["responsive"]):
            if stale_streak == 0:
                stale_start = timestamp
            stale_streak += 1
            if stale_streak == HANG_SAMPLE_COUNT and stale_start is not None:
                events.append(stale_start)
        else:
            stale_streak = 0
            stale_start = None
    return sorted(set(events))


def build_labeled_dataset(
    db: Database = database,
    horizon_s: int = DEFAULT_HORIZON_S,
) -> list[dict[str, Any]]:
    """Build run-grouped labels from past-only stored features and future outcomes."""
    process_rows = db.query(
        """SELECT p.id FROM managed_processes p
        WHERE NOT EXISTS (
            SELECT 1 FROM decisions d
            WHERE d.process_id=p.id AND d.reason LIKE '%Worker error:%'
        )
        AND NOT EXISTS (
            SELECT 1 FROM recovery_attempts r
            WHERE r.process_id=p.id AND r.reason LIKE '%Replacement worker error:%'
        )
        ORDER BY p.id"""
    )
    dataset: list[dict[str, Any]] = []
    for process in process_rows:
        process_id = process["id"]
        measurements = db.query(
            "SELECT * FROM measurements WHERE process_id=? ORDER BY timestamp",
            (process_id,),
        )
        predictions = db.query(
            "SELECT timestamp,risk_score,features_json FROM predictions "
            "WHERE process_id=? ORDER BY timestamp",
            (process_id,),
        )
        if len(measurements) < 2 or not predictions:
            continue
        events = _failure_times(measurements)
        measurement_by_time = {
            float(sample["timestamp"]): sample for sample in measurements
        }
        end_time = float(measurements[-1]["timestamp"])
        for prediction in predictions:
            timestamp = float(prediction["timestamp"])
            matching_measurement = measurement_by_time.get(timestamp)
            if matching_measurement is not None and (
                matching_measurement["process_state"] == "failed"
                or not bool(matching_measurement["responsive"])
            ):
                continue
            event_time = next((event for event in events if event > timestamp), None)
            if event_time is not None and event_time <= timestamp + horizon_s:
                label = 1
                target_event = event_time
            elif end_time >= timestamp + horizon_s:
                label = 0
                target_event = None
            else:
                continue
            features = json.loads(prediction["features_json"])
            dataset.append({
                "process_id": process_id,
                "timestamp": timestamp,
                "features": {
                    name: float(features.get(name, 0.0))
                    for name in FEATURE_COLUMNS
                },
                "baseline_risk_score": estimate_risk(
                    process_id,
                    {
                        name: float(features.get(name, 0.0))
                        for name in FEATURE_COLUMNS
                    },
                    horizon_s,
                    0.35,
                    0.65,
                ).risk_score,
                "label": label,
                "event_time": target_event,
            })
    return dataset


def _safe_metrics(labels: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    matrix = confusion_matrix(labels, predicted, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    return {
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
        "confusion_matrix": {
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
        },
    }


def _non_overlapping_indexes(rows: list[dict[str, Any]], horizon_s: int) -> np.ndarray:
    by_run: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        by_run.setdefault(str(row["process_id"]), []).append(index)
    selected: list[int] = []
    for indexes in by_run.values():
        indexes.sort(key=lambda index: float(rows[index]["timestamp"]))
        cursor = 0
        while cursor < len(indexes):
            block_start = float(rows[indexes[cursor]]["timestamp"])
            block_end = block_start + horizon_s
            block_stop = cursor
            while (
                block_stop + 1 < len(indexes)
                and float(rows[indexes[block_stop + 1]]["timestamp"]) < block_end
            ):
                block_stop += 1
            selected.append(indexes[block_stop])
            selected_timestamp = float(rows[indexes[block_stop]]["timestamp"])
            cursor = block_stop + 1
            while (
                cursor < len(indexes)
                and float(rows[indexes[cursor]]["timestamp"])
                < selected_timestamp + horizon_s
            ):
                cursor += 1
    return np.asarray(selected, dtype=int)


def _calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for index in range(bins):
        selected = (
            (probabilities >= edges[index])
            & (
                probabilities <= edges[index + 1]
                if index == bins - 1
                else probabilities < edges[index + 1]
            )
        )
        if np.any(selected):
            error += float(np.mean(selected)) * abs(
                float(np.mean(probabilities[selected]))
                - float(np.mean(labels[selected]))
            )
    return error


def _fit_group_calibrated_model(
    matrix: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    random_state: int,
) -> CalibratedClassifierCV:
    split_count = min(2, len(np.unique(groups)))
    if split_count < 2:
        raise ValueError("At least two workload runs are required for calibration.")
    splitter = StratifiedGroupKFold(
        n_splits=split_count, shuffle=True, random_state=random_state
    )
    folds = [
        (train, test)
        for train, test in splitter.split(matrix, labels, groups)
        if len(np.unique(labels[train])) == 2
        and len(np.unique(labels[test])) == 2
    ]
    if len(folds) != split_count:
        raise ValueError(
            "Could not form group-disjoint calibration folds containing both classes. "
            "Collect more positive and negative workload runs."
        )
    return CalibratedClassifierCV(
        estimator=make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=random_state
            ),
        ),
        method="sigmoid",
        cv=folds,
        ensemble=False,
    ).fit(matrix, labels)


def train_and_evaluate(
    db: Database = database,
    *,
    horizon_s: int = DEFAULT_HORIZON_S,
    classification_threshold: float = 0.65,
    model_path: Path = DEFAULT_MODEL_PATH,
    random_state: int = 42,
) -> dict[str, Any]:
    if horizon_s < 1:
        raise ValueError("horizon_s must be positive.")
    if not 0.0 < classification_threshold < 1.0:
        raise ValueError("classification_threshold must be between zero and one.")
    rows = build_labeled_dataset(db, horizon_s)
    groups = np.asarray([row["process_id"] for row in rows])
    labels = np.asarray([row["label"] for row in rows], dtype=int)
    unique_groups = np.unique(groups)
    if len(unique_groups) < 8:
        raise ValueError(
            f"At least eight complete workload runs are needed for repeated grouped "
            f"evaluation; "
            f"found {len(unique_groups)}."
        )
    if len(np.unique(labels)) != 2:
        raise ValueError("The labelled runs must contain both positive and negative windows.")
    positive_groups = {
        str(group) for group, label in zip(groups, labels, strict=True) if label == 1
    }
    negative_groups = {
        str(group) for group, label in zip(groups, labels, strict=True) if label == 0
    }
    if len(positive_groups) < 2 or len(negative_groups) < 2:
        raise ValueError(
            "Positive and negative windows must each occur in at least two separate "
            "workload runs for grouped evaluation."
        )

    matrix = np.asarray([
        [row["features"][name] for name in FEATURE_COLUMNS] for row in rows
    ], dtype=float)
    non_overlapping = _non_overlapping_indexes(rows, horizon_s)
    splitter = GroupShuffleSplit(
        n_splits=100, test_size=0.25, random_state=random_state
    )
    heldout_probabilities: list[float] = []
    heldout_labels: list[int] = []
    heldout_baseline_scores: list[float] = []
    heldout_baseline_labels: list[int] = []
    split_metrics: list[dict[str, Any]] = []
    split_records: list[dict[str, Any]] = []
    lead_times: list[float] = []
    evaluated_runs: set[str] = set()
    valid_splits = 0
    for train_indexes, test_indexes in splitter.split(
        matrix, labels, groups
    ):
        if (
            len(np.unique(labels[train_indexes])) != 2
            or len(np.unique(labels[test_indexes])) != 2
        ):
            continue
        test_groups = set(groups[test_indexes])
        evaluation_indexes = np.asarray([
            index for index in non_overlapping
            if groups[index] in test_groups
        ], dtype=int)
        evaluation_labels = labels[evaluation_indexes]
        if len(np.unique(evaluation_labels)) != 2:
            continue
        model = _fit_group_calibrated_model(
            matrix[train_indexes],
            labels[train_indexes],
            groups[train_indexes],
            random_state,
        )
        probabilities = model.predict_proba(matrix[evaluation_indexes])[:, 1]
        predicted = (probabilities >= classification_threshold).astype(int)
        split_result = _safe_metrics(evaluation_labels, predicted)
        baseline_scores = np.asarray([
            rows[index]["baseline_risk_score"] for index in evaluation_indexes
        ], dtype=float)
        baseline_predicted = (baseline_scores >= classification_threshold).astype(int)
        split_metrics.append(split_result)
        split_records.append({
            "train_runs": sorted(str(group) for group in set(groups[train_indexes])),
            "test_runs": sorted(str(group) for group in test_groups),
            "model_metrics": split_result,
            "rule_based_baseline_metrics": _safe_metrics(
                evaluation_labels, baseline_predicted
            ),
        })
        heldout_probabilities.extend(float(value) for value in probabilities)
        heldout_labels.extend(int(value) for value in evaluation_labels)
        heldout_baseline_scores.extend(float(value) for value in baseline_scores)
        heldout_baseline_labels.extend(int(value) for value in evaluation_labels)
        evaluated_runs.update(str(groups[index]) for index in evaluation_indexes)
        for row, prediction in zip(
            (rows[index] for index in evaluation_indexes), predicted, strict=True
        ):
            if row["label"] == 1 and prediction == 1 and row["event_time"] is not None:
                lead_times.append(float(row["event_time"]) - float(row["timestamp"]))
        valid_splits += 1
        if valid_splits >= 30:
            break
    if valid_splits < 5:
        raise ValueError(
            "Fewer than five valid grouped holdouts had both classes and non-overlapping "
            "test windows. Collect more varied positive and negative workload runs."
        )

    final_model = _fit_group_calibrated_model(
        matrix,
        labels,
        groups,
        random_state,
    )
    probabilities = np.asarray(heldout_probabilities)
    evaluation_labels = np.asarray(heldout_labels)
    predicted = (probabilities >= classification_threshold).astype(int)
    metrics = _safe_metrics(evaluation_labels, predicted)
    metrics["brier_score"] = float(np.mean((probabilities - evaluation_labels) ** 2))
    metrics["expected_calibration_error_10_bins"] = _calibration_error(
        evaluation_labels, probabilities
    )
    metric_names = (
        "precision", "recall", "f1", "false_positive_rate"
    )
    metrics["grouped_holdout_mean"] = {
        name: mean(float(split[name]) for split in split_metrics)
        for name in metric_names
        if all(split[name] is not None for split in split_metrics)
    }
    metrics["grouped_holdout_std"] = {
        name: float(np.std([float(split[name]) for split in split_metrics]))
        for name in metric_names
        if all(split[name] is not None for split in split_metrics)
    }
    baseline_split_metrics = [
        split["rule_based_baseline_metrics"] for split in split_records
    ]
    metrics["rule_based_baseline_grouped_holdout_mean"] = {
        name: mean(float(split[name]) for split in baseline_split_metrics)
        for name in metric_names
        if all(split[name] is not None for split in baseline_split_metrics)
    }
    metrics["rule_based_baseline_grouped_holdout_std"] = {
        name: float(np.std([float(split[name]) for split in baseline_split_metrics]))
        for name in metric_names
        if all(split[name] is not None for split in baseline_split_metrics)
    }
    metrics["valid_holdout_count"] = valid_splits
    metrics["non_overlapping_test_window_observations"] = len(heldout_labels)
    metrics["evaluated_run_count"] = len(evaluated_runs)
    # The baseline uses the exact same held-out windows and repeated splits.
    baseline_scores = np.asarray(heldout_baseline_scores, dtype=float)
    baseline_labels = np.asarray(heldout_baseline_labels, dtype=int)
    baseline_metrics = _safe_metrics(
        baseline_labels,
        (baseline_scores >= classification_threshold).astype(int),
    )
    train_groups = sorted(str(group) for group in unique_groups)
    test_groups = sorted(evaluated_runs)
    report = {
        "model": "logistic_regression",
        "model_path": str(model_path),
        "horizon_s": horizon_s,
        "dataset_rows": len(rows),
        "run_count": len(unique_groups),
        "positive_run_count": len({
            row["process_id"] for row in rows if row["label"] == 1
        }),
        "negative_run_count": len({
            row["process_id"] for row in rows if row["label"] == 0
        }),
        "positive_windows": int(labels.sum()),
        "negative_windows": int((labels == 0).sum()),
        "final_model_fit_run_count": len(train_groups),
        "cross_validation_test_run_count": len(test_groups),
        "cross_validation_test_runs": test_groups,
        "holdout_splits": split_records,
        "metrics": {
            **metrics,
            "test_sample_count": int(len(heldout_labels)),
            "mean_early_warning_lead_time_s": mean(lead_times) if lead_times else None,
            "median_early_warning_lead_time_s": median(lead_times) if lead_times else None,
            "rule_based_baseline": baseline_metrics,
            "classification_threshold": classification_threshold,
            "probability_calibration": (
                "Sigmoid calibration is fitted only within group-disjoint training folds. "
                "Brier score and 10-bin expected calibration error are measured on "
                "repeated held-out runs; calibration quality is not assumed to generalize."
            ),
        },
        "label_definition": (
            f"Positive when a recorded exit or three consecutive unresponsive samples "
            f"begins within the next {horizon_s} seconds; negative only when the run "
            "has complete observation through that horizon. Other windows are censored."
        ),
        "split_method": (
            "Repeated GroupShuffleSplit by complete managed-process run ID; "
            "test metrics use non-overlapping horizon windows; seed "
            f"{random_state}."
        ),
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": final_model,
        "feature_columns": FEATURE_COLUMNS,
        "horizon_s": horizon_s,
        "model_name": "logistic_regression_sigmoid_group_calibrated",
        "evaluation": report["metrics"],
        "training_report": report,
        "final_fit_process_ids": train_groups,
        "cross_validation_test_process_ids": test_groups,
    }, model_path)
    return report


@lru_cache(maxsize=4)
def _load_trained_model(model_path: str, modified_at: int) -> Mapping[str, Any]:
    del modified_at
    payload = joblib.load(model_path)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError("Saved prediction model file has an unsupported format.")
    return payload


def load_trained_model(model_path: Path = DEFAULT_MODEL_PATH) -> Mapping[str, Any] | None:
    if not model_path.exists():
        return None
    return _load_trained_model(str(model_path), model_path.stat().st_mtime_ns)
