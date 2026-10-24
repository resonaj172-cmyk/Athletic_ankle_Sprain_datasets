import json
from pathlib import Path
from typing import Dict, Any, List

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config
from app.ml.loader import ModelLoader
from app.ml.features import compute_history_summary
from app.ml.predict import explain_prediction
from app.ml.labels import LABELS_CONFIG


def build_assistant_context(athlete_id: int, db_sessions: List[Any]) -> Dict[str, Any]:
    """
    Builds sanitized, privacy-safe context for the AI Assistant.
    NO athlete names, emails, or credentials are included.
    """
    meta = ModelLoader.get_selected_metadata() or {}
    model_name = meta.get("model_name", "Selected Model")
    metrics = meta.get("metrics", {})
    acc_pct = round(metrics.get("accuracy", 0.0) * 100, 1)
    f1_val = round(metrics.get("macro_f1", 0.0), 3)

    # 1. Stored sessions summary (sanitized, numbers only)
    sessions_data = []
    for s in db_sessions:
        sessions_data.append({
            "session_number": s.session_number,
            "training_duration": s.training_duration,
            "training_intensity": s.training_intensity,
            "training_load": s.training_load,
            "heart_rate": s.heart_rate,
            "sleep_quality": s.sleep_quality,
            "recovery_score": s.recovery_score,
            "stress_level": s.stress_level,
            "predicted_class": s.predicted_class,
            "prob_low": s.prob_low,
            "prob_moderate": s.prob_moderate,
            "prob_high": s.prob_high,
            "source": s.source,
        })

    # 2. Latest session details
    latest_session = sessions_data[-1] if sessions_data else None
    latest_explanation = None
    workload_summary = None

    if latest_session:
        earlier = sessions_data[:-1]
        workload_summary = compute_history_summary(earlier, latest_session["training_load"])

        if latest_session.get("predicted_class"):
            feat_input = {
                "training_duration": latest_session["training_duration"],
                "training_intensity": latest_session["training_intensity"],
                "training_load": latest_session["training_load"],
                "heart_rate": latest_session["heart_rate"],
                "sleep_quality": latest_session["sleep_quality"],
                "recovery_score": latest_session["recovery_score"],
                "stress_level": latest_session["stress_level"],
            }
            try:
                latest_explanation = explain_prediction(feat_input)
            except Exception:
                pass

    # 3. Model Information & Limitations
    limitations = [
        f"Test accuracy of {model_name} is {acc_pct}% (Macro F1 {f1_val}) measured on held-out test data.",
        "Predictions use strictly today's 7 session values; previous sessions are context only and do not affect prediction.",
        "Sessions are tracked by session sequence, not calendar dates.",
        "Classes (Low, Moderate, High) are project-defined thresholds (split at 47.26 and 68.89), not clinical diagnostic categories.",
        "Predictions provide monitoring support only and are NOT medical advice or injury diagnoses.",
    ]

    context = {
        "model_name": model_name,
        "accuracy_pct": acc_pct,
        "macro_f1": f1_val,
        "total_sessions_count": len(sessions_data),
        "sessions_history": sessions_data[-14:],  # up to last 14 sessions
        "latest_session": latest_session,
        "latest_explanation": latest_explanation,
        "workload_context": workload_summary,
        "limitations": limitations,
        "labels_rules": LABELS_CONFIG,
    }

    return context
