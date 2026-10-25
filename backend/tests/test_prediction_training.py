import json
import tempfile
import unittest
from pathlib import Path

from app.core.database import Database
from app.prediction.train_model import (
    FEATURE_COLUMNS,
    build_labeled_dataset,
    load_trained_model,
    _non_overlapping_indexes,
    train_and_evaluate,
)
from app.prediction.risk_estimator import estimate_risk


class GroupedModelTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.temp.name) / "training.sqlite3")
        self.db.initialize()

    def tearDown(self):
        self.temp.cleanup()

    def _create_runs(self):
        for run_index in range(8):
            process_id = f"run-{run_index}"
            crash_run = run_index < 4
            self.db.execute(
                """INSERT INTO managed_processes
                (id,name,workload_type,pid,status,policy,config_json,started_at)
                VALUES(?,?,?,?,?,?,?,?)""",
                (process_id, process_id, "crash" if crash_run else "healthy",
                 1000 + run_index, "stopped", "preemptguard", "{}",
                 "2026-01-01T00:00:00+00:00"),
            )
            for tick in range(13):
                timestamp = float(tick * 5)
                failed = crash_run and tick >= 8
                value = 1.0 if crash_run else 0.0
                self.db.execute(
                    """INSERT INTO measurements
                    (process_id,timestamp,process_state,responsive)
                    VALUES(?,?,?,?)""",
                    (process_id, timestamp, "failed" if failed else "running", 0 if failed else 1),
                )
                if failed:
                    continue
                features = {
                    name: 0.0 for name in FEATURE_COLUMNS
                }
                features.update({
                    "cpu_mean": 90.0 * value,
                    "memory_growth_mb_s": 3.0 * value,
                    "abnormal_samples": 5.0 * value,
                    "responsive": 1.0,
                    "history_samples": 10.0,
                })
                self.db.execute(
                    """INSERT INTO predictions
                    (process_id,timestamp,risk_score,category,confidence,horizon_s,
                    model_name,features_json,contributors_json,recommended_action)
                    VALUES(?,?,?,?,?,?,?,?,?,?)""",
                    (
                        process_id, timestamp, 0.8 * value, "High", 0.8,
                        30, "rule", json.dumps(features), "[]", "recheck",
                    ),
                )

    def test_labels_use_future_failure_but_features_remain_saved_past_features(self):
        self._create_runs()
        rows = build_labeled_dataset(self.db, horizon_s=40)
        crash_rows = [row for row in rows if row["process_id"] == "run-0"]
        healthy_rows = [row for row in rows if row["process_id"] == "run-7"]
        self.assertTrue(crash_rows)
        self.assertTrue(healthy_rows)
        self.assertTrue(all(row["label"] == 1 for row in crash_rows))
        self.assertTrue(all(row["label"] == 0 for row in healthy_rows))
        self.assertEqual(crash_rows[0]["features"]["cpu_mean"], 90.0)

    def test_post_failure_and_unresponsive_samples_are_censored(self):
        process_id = "censored-run"
        self.db.execute(
            """INSERT INTO managed_processes
            (id,name,workload_type,pid,status,policy,config_json,started_at)
            VALUES(?,?,?,?,?,?,?,?)""",
            (process_id, process_id, "crash", 1001, "stopped",
             "fixed", "{}", "2026-01-01T00:00:00+00:00"),
        )
        for timestamp, state, responsive in (
            (0.0, "running", 1),
            (10.0, "failed", 0),
            (20.0, "running", 1),
            (60.0, "running", 1),
        ):
            self.db.execute(
                """INSERT INTO measurements
                (process_id,timestamp,process_state,responsive)
                VALUES(?,?,?,?)""",
                (process_id, timestamp, state, responsive),
            )
            features = {name: 0.0 for name in FEATURE_COLUMNS}
            features["responsive"] = float(responsive)
            self.db.execute(
                """INSERT INTO predictions
                (process_id,timestamp,risk_score,category,confidence,horizon_s,
                model_name,features_json,contributors_json,recommended_action)
                VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (process_id, timestamp, 0.8, "High", 0.8, 30, "rule",
                 json.dumps(features), "[]", "recheck"),
            )
        rows = build_labeled_dataset(self.db, horizon_s=30)
        run_rows = [row for row in rows if row["process_id"] == process_id]
        self.assertEqual([row["timestamp"] for row in run_rows], [0.0, 20.0])
        self.assertEqual(run_rows[0]["label"], 1)
        self.assertEqual(run_rows[1]["label"], 0)

    def test_training_splits_by_complete_run_and_reports_metrics(self):
        self._create_runs()
        model_path = Path(self.temp.name) / "risk-model.joblib"
        result = train_and_evaluate(
            self.db, horizon_s=40, model_path=model_path, random_state=42
        )
        self.assertTrue(model_path.exists())
        self.assertEqual(len(result["holdout_splits"]), result["metrics"]["valid_holdout_count"])
        for split in result["holdout_splits"]:
            self.assertTrue(set(split["train_runs"]).isdisjoint(split["test_runs"]))
        self.assertIn("Repeated GroupShuffleSplit", result["split_method"])
        self.assertGreaterEqual(result["metrics"]["valid_holdout_count"], 5)
        for key in (
            "precision", "recall", "f1", "false_positive_rate",
            "brier_score", "expected_calibration_error_10_bins",
            "grouped_holdout_mean", "rule_based_baseline",
        ):
            self.assertIn(key, result["metrics"])
        self.assertIn("not assumed to generalize", result["metrics"]["probability_calibration"])
        saved_model = load_trained_model(model_path)
        self.assertIn("evaluation", saved_model)
        self.assertEqual(saved_model["training_report"]["run_count"], 8)
        prediction = estimate_risk(
            "current-run",
            {name: 0.0 for name in FEATURE_COLUMNS},
            40, 0.35, 0.65,
            trained_model=dict(saved_model),
        )
        self.assertEqual(
            prediction.model_name,
            "logistic_regression_sigmoid_group_calibrated",
        )

    def test_evaluation_windows_do_not_overlap_within_each_run(self):
        rows = [
            {"process_id": "a", "timestamp": float(tick)}
            for tick in range(0, 101, 5)
        ] + [
            {"process_id": "b", "timestamp": float(tick)}
            for tick in range(0, 61, 10)
        ]
        selected = _non_overlapping_indexes(rows, horizon_s=30)
        for process_id in ("a", "b"):
            times = sorted(
                rows[index]["timestamp"]
                for index in selected
                if rows[index]["process_id"] == process_id
            )
            self.assertTrue(all(
                right - left >= 30
                for left, right in zip(times, times[1:])
            ))


if __name__ == "__main__":
    unittest.main()
