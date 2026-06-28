import numpy as np
from sklearn.ensemble import RandomForestClassifier

from MetaSleepGuard.rejection.calibration import expected_calibration_error, multiclass_brier_score
from MetaSleepGuard.rejection.coverage_risk import coverage_risk_curve
from MetaSleepGuard.rejection.rejector import ActiveRejector, apply_rejection
from MetaSleepGuard.models.evaluate import evaluate_probabilities
from MetaSleepGuard.models.train_xgb import fit_with_subject_calibration


def test_active_rejection_rules():
    rejector = ActiveRejector(confidence_threshold=0.6)
    classes = ["W", "N1", "N2"]
    accepted = rejector.decide([0.7, 0.2, 0.1], classes, "A", True)
    rejected_quality = rejector.decide([0.9, 0.05, 0.05], classes, "D", False)
    rejected_conf = rejector.decide([0.4, 0.35, 0.25], classes, "A", True)
    assert accepted.accepted and accepted.stage == "W"
    assert not rejected_quality.accepted and rejected_quality.stage == "暂不判定"
    assert not rejected_conf.accepted


def test_calibration_metrics_and_curve():
    y = np.array([0, 1, 1])
    p = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
    assert expected_calibration_error(y, p) >= 0
    assert multiclass_brier_score(y, p) >= 0
    curve = coverage_risk_curve(y, p, thresholds=[0.0, 0.5, 0.9])
    assert len(curve) == 3


def test_probability_metrics_include_rejection_summary():
    metrics = evaluate_probabilities(
        np.array([0, 1, 0]),
        np.array([[0.9, 0.1], [0.45, 0.55], [0.4, 0.6]]),
    )
    assert 0.0 <= metrics["ece"] <= 1.0
    assert len(metrics["coverage_risk_curve"]) == 21
    assert "after_macro_f1" in metrics["rejection"]


def test_calibration_falls_back_when_subject_lacks_classes():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(8, 3))
    subjects = np.repeat(["s0", "s1", "s2", "s3"], 2)
    y = np.array([0, 1, 0, 2, 1, 2, 0, 1])
    model, status = fit_with_subject_calibration(
        RandomForestClassifier(n_estimators=5, random_state=1),
        X,
        y,
        subjects,
        np.ones(y.size, dtype=bool),
        random_state=42,
    )
    assert not status["applied"]
    assert status["reason"] == "calibration_subjects_do_not_cover_all_classes"
    assert hasattr(model, "predict_proba")


def test_apply_rejection_accepts_numpy_quality_arrays():
    decisions = apply_rejection(
        np.array([[0.8, 0.2], [0.4, 0.6]]),
        ["W", "N1"],
        quality_grades=np.array(["A", "D"]),
        reliability=np.array([True, False]),
    )
    assert decisions[0].accepted
    assert not decisions[1].accepted
