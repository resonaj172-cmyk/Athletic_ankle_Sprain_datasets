import os
import json
import pytest
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from pathlib import Path

# Add project root to sys.path
import sys
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
import config
from app.models_db import db, User, Athlete, SessionRecord
from app.ml.loader import ModelLoader, ModelNotFoundError
from app.ml.features import compute_history_summary, validate_session_inputs
from app.ml.predict import predict_session, explain_prediction, compute_what_if
from seed_demo import DEMO_SESSIONS, seed_demo_data
from app.app import create_app


@pytest.fixture
def app_client():
    app = create_app()
    app.config["TESTING"] = True
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    app.config["WTF_CSRF_ENABLED"] = False

    with app.app_context():
        db.create_all()
        seed_demo_data(app)
        with app.test_client() as client:
            yield client, app


# -------------------------------------------------------------
# A. PIPELINE OUTPUTS EXIST
# -------------------------------------------------------------
def test_pipeline_outputs_exist():
    """Verify all 3 model folders, metrics.csv, wall_time.csv, and selected_model.json exist."""
    model_keys = ["logistic_regression", "random_forest", "xgboost"]
    for key in model_keys:
        model_dir = config.MODELS_DIR / key
        assert model_dir.exists(), f"Directory missing: {model_dir}"
        assert (model_dir / "metrics.csv").exists(), f"metrics.csv missing in {key}"
        assert (model_dir / "wall_time.csv").exists(), f"wall_time.csv missing in {key}"
        assert (model_dir / "best_params.json").exists(), f"best_params.json missing in {key}"
        assert (model_dir / "model_config.json").exists(), f"model_config.json missing in {key}"

        if key == "xgboost":
            assert (model_dir / "model.json").exists(), "XGBoost model.json missing"
        else:
            assert (model_dir / "model.joblib").exists(), f"{key} model.joblib missing"

    selected_json = config.MODELS_DIR / "selected_model.json"
    assert selected_json.exists(), "selected_model.json missing"

    meta = json.loads(selected_json.read_text(encoding="utf-8"))
    assert "model_name" in meta
    assert "metrics" in meta

    # Check that ModelLoader loads the selected model
    model, cfg, loaded_meta = ModelLoader.load_selected_model(force_reload=True)
    assert model is not None
    assert cfg["model_name"] == meta["model_name"]


# -------------------------------------------------------------
# B. NO-LEAKAGE TEST
# -------------------------------------------------------------
def test_no_leakage():
    """Verify X contains only 7 features, no target, and train/test share 0 identical rows."""
    train_df = pd.read_csv(config.PROCESSED_DATA_DIR / "train.csv")
    test_df = pd.read_csv(config.PROCESSED_DATA_DIR / "test.csv")

    assert config.TARGET_COLUMN not in config.FEATURE_COLUMNS
    assert config.CLASS_COLUMN not in config.FEATURE_COLUMNS
    assert len(config.FEATURE_COLUMNS) == 7

    # Check identical rows
    overlap = pd.merge(train_df[config.FEATURE_COLUMNS], test_df[config.FEATURE_COLUMNS], how="inner")
    assert len(overlap) == 0, f"Leakage detected: {len(overlap)} identical rows across split!"


# -------------------------------------------------------------
# C. REPRODUCIBILITY TEST
# -------------------------------------------------------------
def test_reproducibility():
    """Reloading saved models reproduces saved test accuracy within 1e-6."""
    test_df = pd.read_csv(config.PROCESSED_DATA_DIR / "test.csv")
    X_test = test_df[config.FEATURE_COLUMNS]
    y_test = test_df[config.CLASS_COLUMN]

    # 1. Logistic Regression
    lr_model = joblib.load(config.MODELS_DIR / "logistic_regression" / "model.joblib")
    lr_metrics = pd.read_csv(config.MODELS_DIR / "logistic_regression" / "metrics.csv")
    lr_acc = float((lr_model.predict(X_test) == y_test).mean())
    assert abs(lr_acc - float(lr_metrics.loc[0, "test_accuracy"])) < 1e-6

    # 2. Random Forest
    rf_model = joblib.load(config.MODELS_DIR / "random_forest" / "model.joblib")
    rf_metrics = pd.read_csv(config.MODELS_DIR / "random_forest" / "metrics.csv")
    rf_acc = float((rf_model.predict(X_test) == y_test).mean())
    assert abs(rf_acc - float(rf_metrics.loc[0, "test_accuracy"])) < 1e-6

    # 3. XGBoost
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(str(config.MODELS_DIR / "xgboost" / "model.json"))
    xgb_metrics = pd.read_csv(config.MODELS_DIR / "xgboost" / "metrics.csv")
    xgb_acc = float((xgb_model.predict(X_test) == y_test).mean())
    assert abs(xgb_acc - float(xgb_metrics.loc[0, "test_accuracy"])) < 1e-6


# -------------------------------------------------------------
# D. SELECTION TEST
# -------------------------------------------------------------
def test_selection_rule():
    """Verify selected model matches selection rule against model_comparison.csv."""
    comp_df = pd.read_csv(config.REPORTS_DIR / "model_comparison.csv")
    meta = json.loads((config.MODELS_DIR / "selected_model.json").read_text(encoding="utf-8"))

    # Highest macro-F1 among candidate pool
    sorted_df = comp_df.sort_values(by=["macro_f1", "roc_auc"], ascending=[False, False]).reset_index(drop=True)
    expected_winner = sorted_df.iloc[0]["model_name"]
    assert meta["model_name"] == expected_winner


# -------------------------------------------------------------
# E. ACCURACY GATE TEST
# -------------------------------------------------------------
def test_accuracy_gate_report():
    """Reports each model's accuracy and whether meets_target >= 0.90 is accurately recorded."""
    comp_df = pd.read_csv(config.REPORTS_DIR / "model_comparison.csv")
    for _, r in comp_df.iterrows():
        expected_meets = bool(r["accuracy"] >= 0.90)
        assert bool(r["meets_target"]) == expected_meets, f"meets_target mismatch for {r['model_name']}"


# -------------------------------------------------------------
# F. HISTORY SUMMARY TEST (TABLE IN SECTION 9 REPRODUCED WITHIN 0.001)
# -------------------------------------------------------------
def test_history_summary_reproduction():
    """Reproduce Section 9 history summary table for session 15 within 0.001 tolerance."""
    s15_load = 560.0
    summary = compute_history_summary(DEMO_SESSIONS, s15_load)

    # 7-session and 14-session load sums and means
    assert abs(summary["load_sum_7"] - 5510.0) < 0.001
    assert abs(summary["load_mean_7"] - 787.1429) < 0.001
    assert abs(summary["load_sum_14"] - 10160.0) < 0.001
    assert abs(summary["load_mean_14"] - 725.7143) < 0.001

    # Differences & Ratios
    assert abs(summary["load_minus_mean_7"] - (-227.1429)) < 0.001
    assert abs(summary["load_minus_mean_14"] - (-165.7143)) < 0.001
    assert abs(summary["load_ratio_7"] - 0.7114) < 0.001
    assert abs(summary["load_ratio_14"] - 0.7717) < 0.001

    # Means of other variables
    assert abs(summary["training_intensity_mean_7"] - 7.0000) < 0.001
    assert abs(summary["training_intensity_mean_14"] - 6.7143) < 0.001
    assert abs(summary["heart_rate_mean_7"] - 74.5714) < 0.001
    assert abs(summary["heart_rate_mean_14"] - 74.0000) < 0.001  # 13 present
    assert abs(summary["sleep_quality_mean_7"] - 5.5286) < 0.001
    assert abs(summary["sleep_quality_mean_14"] - 5.8143) < 0.001
    assert abs(summary["recovery_score_mean_7"] - 50.7143) < 0.001
    assert abs(summary["recovery_score_mean_14"] - 53.5000) < 0.001
    assert abs(summary["stress_level_mean_7"] - 0.4814) < 0.001
    assert abs(summary["stress_level_mean_14"] - 0.4521) < 0.001


# -------------------------------------------------------------
# G. LEAKAGE IN HISTORY TEST
# -------------------------------------------------------------
def test_leakage_in_history():
    """Verify current session is excluded from window; session 16 uses 9-15 and 2-15."""
    # Build 15 sessions
    sessions_15 = list(DEMO_SESSIONS)
    s15 = {"session_number": 15, "training_duration": 70.0, "training_intensity": 8.0, "training_load": 560.0, "heart_rate": 78.0, "sleep_quality": 5.5, "recovery_score": 45.0, "stress_level": 0.75}
    sessions_15.append(s15)

    s16_load = 600.0
    summary_16 = compute_history_summary(sessions_15, s16_load)

    # Window 7 for session 16 must be sessions 9-15
    window_7_sessions = sessions_15[-7:]
    assert window_7_sessions[0]["session_number"] == 9
    assert window_7_sessions[-1]["session_number"] == 15
    expected_load_sum_7 = sum(s["training_load"] for s in window_7_sessions)
    assert abs(summary_16["load_sum_7"] - expected_load_sum_7) < 1e-4

    # Window 14 for session 16 must be sessions 2-15
    window_14_sessions = sessions_15[-14:]
    assert window_14_sessions[0]["session_number"] == 2
    assert window_14_sessions[-1]["session_number"] == 15
    expected_load_sum_14 = sum(s["training_load"] for s in window_14_sessions)
    assert abs(summary_16["load_sum_14"] - expected_load_sum_14) < 1e-4


# -------------------------------------------------------------
# H. NO-PREDICTION TEST (< 15 SAVED WITHOUT PREDICTION, >= 15 WITH)
# -------------------------------------------------------------
def test_no_prediction_before_15(app_client):
    client, app = app_client
    with app.app_context():
        user = User.query.filter_by(email="demo@athlete.app").first()
        sessions = SessionRecord.query.filter_by(athlete_id=user.athlete.athlete_id).order_by(SessionRecord.session_number.asc()).all()

        assert len(sessions) == 14
        for s in sessions:
            assert s.predicted_class is None
            assert s.prob_low is None
            assert s.source == "demo_seed"

        # Now test session 15 prediction inference
        s15_data = {
            "training_duration": 70.0,
            "training_intensity": 8.0,
            "training_load": 560.0,
            "heart_rate": 78.0,
            "sleep_quality": 5.5,
            "recovery_score": 45.0,
            "stress_level": 0.75,
        }
        pred = predict_session(s15_data)
        assert pred["predicted_class"] in ["Low", "Moderate", "High"]
        assert len(pred["probabilities_raw"]) == 3
        assert abs(sum(pred["probabilities_raw"]) - 1.0) < 1e-4


# -------------------------------------------------------------
# I. MISSING-VALUE HANDLING TEST
# -------------------------------------------------------------
def test_missing_values():
    """Blank heart rate / sleep / intensity stored as NULL and passed as NaN without errors."""
    feat_with_nan = {
        "training_duration": 80.0,
        "training_intensity": np.nan,
        "training_load": 500.0,
        "heart_rate": np.nan,
        "sleep_quality": np.nan,
        "recovery_score": 50.0,
        "stress_level": 0.45,
    }
    pred = predict_session(feat_with_nan)
    assert pred["predicted_class"] in ["Low", "Moderate", "High"]

    # Test history mean with fewer than 3 values returns None
    sparse_sessions = [
        {"training_load": 100, "heart_rate": None},
        {"training_load": 100, "heart_rate": 70.0},
        {"training_load": 100, "heart_rate": None},
        {"training_load": 100, "heart_rate": 72.0},
        {"training_load": 100, "heart_rate": None},
        {"training_load": 100, "heart_rate": None},
        {"training_load": 100, "heart_rate": None},
    ]
    summ = compute_history_summary(sparse_sessions, 100.0)
    assert summ["heart_rate_mean_7"] is None  # Only 2 non-missing HR values


# -------------------------------------------------------------
# J. EXPLANATION TEST (FOR EACH OF THE THREE MODELS)
# -------------------------------------------------------------
def test_explanation_all_three_models():
    """Verify explanation returns top factors for session 15 for all 3 models."""
    s15 = {"training_duration": 70.0, "training_intensity": 8.0, "training_load": 560.0, "heart_rate": 78.0, "sleep_quality": 5.5, "recovery_score": 45.0, "stress_level": 0.75}

    models_info = [
        ("logistic_regression", joblib.load(config.MODELS_DIR / "logistic_regression" / "model.joblib")),
        ("random_forest", joblib.load(config.MODELS_DIR / "random_forest" / "model.joblib")),
        ("xgboost", xgb.XGBClassifier()),
    ]
    models_info[2][1].load_model(str(config.MODELS_DIR / "xgboost" / "model.json"))

    for key, mdl in models_info:
        cfg = json.loads((config.MODELS_DIR / key / "model_config.json").read_text(encoding="utf-8"))
        meta = {"model_name": cfg["model_name"]}
        expl = explain_prediction(s15, model=mdl, model_config=cfg, selected_meta=meta)
        assert len(expl["top_towards"]) <= 5
        assert len(expl["top_against"]) <= 3
        assert len(expl["all_factors"]) == 7


# -------------------------------------------------------------
# K. AUTH / OWNERSHIP
# -------------------------------------------------------------
def test_auth_and_ownership(app_client):
    client, app = app_client
    with app.app_context():
        demo_user = User.query.filter_by(email="demo@athlete.app").first()
        assert demo_user.password_hash.startswith("$2") or demo_user.password_hash.startswith("$argon")

    # Unauthenticated access redirects to login
    resp = client.get("/dashboard", follow_redirects=False)
    assert resp.status_code == 302
    assert "/login" in resp.headers["Location"]


# -------------------------------------------------------------
# L. UI CHECKS
# -------------------------------------------------------------
def test_ui_compliance(app_client):
    client, app = app_client
    # Get login page to initiate session & CSRF token
    resp_get = client.get("/login")
    with client.session_transaction() as sess:
        csrf_tok = sess.get("csrf_token", "")

    # Log in as demo user with valid CSRF token
    resp = client.post(
        "/login",
        data={"email": "demo@athlete.app", "password": "Demo@1234", "csrf_token": csrf_tok},
        follow_redirects=True,
    )
    assert resp.status_code == 200

    html = resp.data.decode("utf-8")
    # Navbar order check
    assert "Dashboard" in html
    assert "Add Session" in html
    assert "History" in html
    assert "Athlete Details" in html
    assert "AI Assistant" in html
    assert "Logout" in html

    # Forbidden strings check on dashboard
    assert "Prototype. Not medical advice." not in html
    assert "batch size" not in html.lower()
    assert "epochs" not in html.lower()
