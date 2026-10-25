from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class Prediction:
    process_id: str
    timestamp: str
    risk_score: float
    category: str
    confidence: float
    horizon_s: int
    model_name: str
    features: dict[str, float]
    contributors: list[str]
    recommended_action: str
    confidence_kind: str = "history_coverage_not_probability"


def estimate_risk(
    process_id: str,
    features: dict[str, float],
    horizon_s: int,
    low_threshold: float,
    high_threshold: float,
    failed: bool = False,
    minimum_confidence: float = 0.5,
    trained_model: dict[str, Any] | None = None,
) -> Prediction:
    if failed:
        score = 1.0
        contributors = ["process has exited"]
        model_name = "confirmed_failure"
    elif trained_model is not None:
        columns = trained_model["feature_columns"]
        estimator = trained_model["model"]
        values = [[float(features.get(column, 0.0)) for column in columns]]
        probabilities = estimator.predict_proba(values)[0]
        classes = list(estimator.classes_)
        score = float(probabilities[classes.index(1)])
        base_estimator = estimator
        calibrated_classifiers = getattr(estimator, "calibrated_classifiers_", None)
        if calibrated_classifiers:
            base_estimator = calibrated_classifiers[0].estimator
        transformed = base_estimator.named_steps["standardscaler"].transform(values)[0]
        coefficients = base_estimator.named_steps["logisticregression"].coef_[0]
        contributors = [
            columns[index] for index in sorted(
                range(len(columns)),
                key=lambda index: abs(float(transformed[index] * coefficients[index])),
                reverse=True,
            )[:3]
        ]
        model_name = trained_model.get(
            "model_name", "logistic_regression_sigmoid_group_calibrated"
        )
    else:
        factors = {
            "cpu level and upward trend": min(
                0.18, max(0.0, (features["cpu_mean"] - 65.0) / 35.0 * 0.12)
                + max(0.0, features["cpu_trend"]) / 100.0 * 0.06
            ),
            "sustained memory growth": min(
                0.24, max(0.0, features["memory_growth_mb_s"]) * 0.08
            ),
            "accelerating memory growth": min(
                0.12, max(0.0, features["memory_acceleration_mb_s2"]) * 0.08
            ),
            "missed heartbeats": min(
                0.55, features["missed_heartbeats"] * 0.18
            ),
            "abnormal behaviour duration": min(
                0.24, features["abnormal_samples"] * 0.035
            ),
            "recent failures and recovery attempts": min(
                0.20,
                features.get("recent_failures", 0.0) * 0.12
                + features.get("recent_restarts", 0.0) * 0.04,
            ),
        }
        score = min(0.99, sum(factors.values()))
        contributors = [
            label for label, weight in sorted(
                factors.items(), key=lambda item: item[1], reverse=True
            ) if weight > 0.01
        ][:3]
        model_name = "rule_based_v1 (uncalibrated risk score)"
    count = features.get("history_samples", 0.0)
    confidence = min(0.9, 0.25 + count * 0.065)
    if not contributors:
        contributors = ["no sustained abnormal trend observed"]
    category = (
        "Critical" if score >= 0.85 else
        "High" if score >= high_threshold else
        "Moderate" if score >= low_threshold else "Low"
    )
    return Prediction(
        process_id=process_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        risk_score=round(score, 4),
        category=category,
        confidence=round(confidence, 3),
        horizon_s=horizon_s,
        model_name=model_name,
        features=features,
        contributors=contributors,
        recommended_action=(
            "restart" if failed else
            "recheck" if score >= low_threshold or confidence < minimum_confidence else
            "continue monitoring"
        ),
        confidence_kind="history_coverage_not_probability",
    )


def prediction_record(prediction: Prediction) -> dict[str, object]:
    return asdict(prediction)
