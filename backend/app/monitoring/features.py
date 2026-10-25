from collections.abc import Sequence
from statistics import mean
from typing import Any


def extract_features(samples: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Extract features from measurements at or before the current sample."""
    recent = [sample for sample in samples[-10:] if sample["process_state"] != "stopped"]
    if not recent:
        return {
            "cpu_current": 0.0,
            "cpu_mean": 0.0,
            "cpu_trend": 0.0,
            "memory_current_mb": 0.0,
            "memory_growth_mb_s": 0.0,
            "memory_acceleration_mb_s2": 0.0,
            "missed_heartbeats": 0.0,
            "abnormal_samples": 0.0,
            "responsive": 0.0,
            "recent_failures": 0.0,
            "recent_restarts": 0.0,
            "history_samples": 0.0,
        }
    cpu_values = [float(item["cpu_percent"] or 0.0) for item in recent]
    memory_values = [float(item["memory_mb"] or 0.0) for item in recent]
    growth_values = [float(item["memory_growth_mb_s"] or 0.0) for item in recent]
    cpu_trend = (
        (mean(cpu_values[len(cpu_values) // 2:]) - mean(cpu_values[:len(cpu_values) // 2]))
        if len(cpu_values) >= 4 else 0.0
    )
    acceleration = (
        growth_values[-1] - growth_values[0]
        if len(growth_values) >= 3 else 0.0
    )
    return {
        "cpu_current": cpu_values[-1],
        "cpu_mean": mean(cpu_values),
        "cpu_trend": cpu_trend,
        "memory_current_mb": memory_values[-1],
        "memory_growth_mb_s": growth_values[-1],
        "memory_acceleration_mb_s2": acceleration,
        "missed_heartbeats": float(sum(not bool(item["responsive"]) for item in recent)),
        "abnormal_samples": float(sum(
            float(item["cpu_percent"] or 0.0) >= 85.0
            or float(item["memory_growth_mb_s"] or 0.0) >= 2.0
            or not bool(item["responsive"])
            for item in recent
        )),
        "responsive": float(bool(recent[-1]["responsive"])),
        "recent_failures": float(max(
            int(item.get("recent_failure_count", 0) or 0) for item in recent
        )),
        "recent_restarts": float(max(
            int(item.get("recent_restart_count", 0) or 0) for item in recent
        )),
        "history_samples": float(len(recent)),
    }
