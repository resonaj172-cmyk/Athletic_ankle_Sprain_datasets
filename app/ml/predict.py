import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config
from app.ml.loader import ModelLoader


FRIENDLY_NAMES = {
    "training_duration": "Training Duration of this session",
    "training_intensity": "Training Intensity (RPE)",
    "training_load": "Training Load of this session",
    "heart_rate": "Heart Rate during training",
    "sleep_quality": "Sleep Quality prior to session",
    "recovery_score": "Recovery Score",
    "stress_level": "Perceived Stress Level",
}


def predict_session(
    feature_dict: Dict[str, Any],
    model=None,
    model_config=None,
    selected_meta=None,
) -> Dict[str, Any]:
    """
    Runs model inference on exactly 7 features.
    Returns:
        dict with class_name, probabilities_pct, probabilities_raw, model_name, and input_values.
    """
    if model is None or model_config is None:
        model, model_config, selected_meta = ModelLoader.load_selected_model()

    model_name = model_config.get("model_name", "Selected Model")
    feature_order = model_config.get("feature_columns", config.FEATURE_COLUMNS)

    # Build single-row DataFrame with exact 7 features in exact order
    row_data = {}
    for col in feature_order:
        val = feature_dict.get(col)
        row_data[col] = float(val) if val is not None and not (isinstance(val, float) and np.isnan(val)) else np.nan

    df_input = pd.DataFrame([row_data], columns=feature_order)

    # Predict probabilities
    raw_probs = model.predict_proba(df_input)[0]

    assert len(raw_probs) == 3, f"Expected 3 class probabilities, got {len(raw_probs)}"
    assert abs(sum(raw_probs) - 1.0) < 1e-4, f"Probabilities do not sum to 1.0: {sum(raw_probs)}"

    pred_idx = int(np.argmax(raw_probs))
    pred_class = config.CLASS_NAMES[pred_idx]

    probs_pct = {
        "Low": round(float(raw_probs[0] * 100), 1),
        "Moderate": round(float(raw_probs[1] * 100), 1),
        "High": round(float(raw_probs[2] * 100), 1),
    }

    return {
        "predicted_class": pred_class,
        "predicted_index": pred_idx,
        "probabilities_pct": probs_pct,
        "probabilities_raw": [float(p) for p in raw_probs],
        "model_name": model_name,
        "input_values": row_data,
    }


def generate_fatigue_reasons(
    feature_dict: Dict[str, Any],
    predicted_class: str,
    top_towards: List[Dict[str, Any]],
    workload: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Generates human-readable, evidence-based fatigue reasons comparing session-wise metrics,
    7-session and 14-session historical baselines, and model feature impacts.
    """
    reasons = []

    # 1. Primary Model Feature Impact Reasons
    for factor in top_towards[:3]:
        feat = factor["feature"]
        name = factor["friendly_name"]
        val = factor["actual_value"]
        delta_pct = factor["delta_pct"]

        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue

        if feat == "training_load":
            reasons.append({
                "title": f"Training Load Impact ({val:.0f} a.u.)",
                "description": f"Session training load contributed +{delta_pct}% toward classifying fatigue as {predicted_class}.",
                "icon": "⚡",
                "category": "Session Intensity",
            })
        elif feat == "recovery_score":
            reasons.append({
                "title": f"Recovery Readiness Score ({val:.0f}/100)",
                "description": f"Current recovery readiness state added +{delta_pct}% confidence to the {predicted_class} fatigue prediction.",
                "icon": "🔋",
                "category": "Recovery",
            })
        elif feat == "sleep_quality":
            reasons.append({
                "title": f"Sleep Quality Prior to Session ({val:.1f}/10)",
                "description": f"Pre-session sleep rating increased the {predicted_class} fatigue likelihood by +{delta_pct}%.",
                "icon": "🌙",
                "category": "Sleep",
            })
        elif feat == "stress_level":
            reasons.append({
                "title": f"Perceived Stress Level ({val:.2f})",
                "description": f"Elevated physiological/mental stress level drove +{delta_pct}% toward {predicted_class} fatigue.",
                "icon": "🧠",
                "category": "Stress",
            })
        elif feat == "training_duration":
            reasons.append({
                "title": f"Session Duration ({val:.0f} min)",
                "description": f"Training duration increased fatigue classification likelihood by +{delta_pct}%.",
                "icon": "⏱️",
                "category": "Volume",
            })
        elif feat == "training_intensity":
            reasons.append({
                "title": f"Training Intensity (RPE {val:.1f})",
                "description": f"Perceived effort (RPE) added +{delta_pct}% toward {predicted_class} fatigue state.",
                "icon": "🔥",
                "category": "Intensity",
            })
        elif feat == "heart_rate":
            reasons.append({
                "title": f"Heart Rate Response ({val:.0f} bpm)",
                "description": f"Session heart rate reading contributed +{delta_pct}% to the fatigue evaluation.",
                "icon": "❤️",
                "category": "Physiology",
            })

    # 2. Historical Baseline Comparison Reasons (7-Session & 14-Session)
    if workload:
        # 7-Session Workload Comparison
        if workload.get("has_window_7"):
            ratio_7 = workload.get("load_ratio_7")
            mean_7 = workload.get("load_mean_7")
            diff_7 = workload.get("load_minus_mean_7")
            if ratio_7 is not None and mean_7 is not None:
                if ratio_7 >= 1.2:
                    reasons.append({
                        "title": f"Acute Workload Spike (+{round((ratio_7 - 1.0) * 100, 1)}% vs 7-Session Baseline)",
                        "description": f"Today's load ({feature_dict.get('training_load', 0):.0f} a.u.) is {ratio_7:.2f}x your 7-session rolling mean ({mean_7:.1f} a.u.), placing acute strain on recovery.",
                        "icon": "📈",
                        "category": "7-Session Baseline",
                    })
                elif ratio_7 <= 0.8:
                    reasons.append({
                        "title": "Deload Session (-" + f"{round((1.0 - ratio_7) * 100, 1)}% vs 7-Session Baseline)",
                        "description": f"Today's load is significantly below your 7-session average ({mean_7:.1f} a.u.), favoring recovery.",
                        "icon": "📉",
                        "category": "7-Session Baseline",
                    })

            # Sleep comparison vs 7-session average
            sq_diff_7 = workload.get("sleep_quality_diff_7")
            sq_mean_7 = workload.get("sleep_quality_mean_7")
            if sq_diff_7 is not None and sq_mean_7 is not None and sq_diff_7 <= -1.0:
                reasons.append({
                    "title": f"Sleep Deficit vs 7-Session Average ({sq_diff_7:.1f} pts)",
                    "description": f"Today's sleep quality ({feature_dict.get('sleep_quality', 0):.1f}) is below your 7-session baseline of {sq_mean_7:.1f}/10.",
                    "icon": "⚠️",
                    "category": "7-Session Baseline",
                })

        # 14-Session Workload Comparison
        if workload.get("has_window_14"):
            ratio_14 = workload.get("load_ratio_14")
            mean_14 = workload.get("load_mean_14")
            if ratio_14 is not None and mean_14 is not None:
                if ratio_14 >= 1.25:
                    reasons.append({
                        "title": f"Chronic Workload Elevation ({ratio_14:.2f}x vs 14-Session Baseline)",
                        "description": f"Today's training load exceeds your 14-session chronic average of {mean_14:.1f} a.u. by {round((ratio_14 - 1.0) * 100, 1)}%.",
                        "icon": "🏋️",
                        "category": "14-Session Baseline",
                    })

            # Recovery score comparison vs 14-session average
            rec_diff_14 = workload.get("recovery_score_diff_14")
            rec_mean_14 = workload.get("recovery_score_mean_14")
            if rec_diff_14 is not None and rec_mean_14 is not None and rec_diff_14 <= -10.0:
                reasons.append({
                    "title": f"Recovery Suppression vs 14-Session Average ({rec_diff_14:.0f} pts)",
                    "description": f"Recovery readiness ({feature_dict.get('recovery_score', 0):.0f}/100) is reduced compared to your 14-session average of {rec_mean_14:.1f}/100.",
                    "icon": "🔋",
                    "category": "14-Session Baseline",
                })

            # Stress comparison vs 14-session average
            st_diff_14 = workload.get("stress_level_diff_14")
            st_mean_14 = workload.get("stress_level_mean_14")
            if st_diff_14 is not None and st_mean_14 is not None and st_diff_14 >= 0.15:
                reasons.append({
                    "title": f"Elevated Stress vs 14-Session Baseline (+{st_diff_14:.2f})",
                    "description": f"Perceived stress level ({feature_dict.get('stress_level', 0):.2f}) is noticeably higher than your 14-session baseline average ({st_mean_14:.2f}).",
                    "icon": "🧠",
                    "category": "14-Session Baseline",
                })

    return reasons


def explain_prediction(
    feature_dict: Dict[str, Any],
    model=None,
    model_config=None,
    selected_meta=None,
    workload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Model-agnostic explainability via median baseline replacement:
    For each of the 7 features:
        delta = prob_real - prob_with_median
    Top 5 toward predicted class, top 3 against predicted class.
    Also returns structured fatigue reasons incorporating 7 and 14 session comparisons.
    """
    if model is None or model_config is None:
        model, model_config, selected_meta = ModelLoader.load_selected_model()

    base_pred = predict_session(feature_dict, model=model, model_config=model_config, selected_meta=selected_meta)
    pred_idx = base_pred["predicted_index"]
    pred_class = base_pred["predicted_class"]
    base_prob = base_pred["probabilities_raw"][pred_idx]

    medians = model_config.get("training_medians", {})
    feature_order = model_config.get("feature_columns", config.FEATURE_COLUMNS)

    factors = []

    for feat in feature_order:
        current_val = feature_dict.get(feat)
        median_val = medians.get(feat, 0.0)

        # Create counterfactual input where this feature is replaced by median
        cf_dict = dict(feature_dict)
        cf_dict[feat] = median_val

        cf_pred = predict_session(cf_dict, model=model, model_config=model_config, selected_meta=selected_meta)
        cf_prob = cf_pred["probabilities_raw"][pred_idx]

        # Delta: how much keeping the real value shifted probability toward predicted class
        delta = base_prob - cf_prob

        factors.append({
            "feature": feat,
            "friendly_name": FRIENDLY_NAMES.get(feat, feat),
            "actual_value": current_val,
            "median_value": median_val,
            "delta_prob": float(delta),
            "delta_pct": round(float(delta * 100), 1),
            "pushed_toward": delta > 0,
        })

    # Sort factors
    towards = [f for f in factors if f["delta_prob"] > 0]
    towards.sort(key=lambda x: x["delta_prob"], reverse=True)
    top_towards = towards[:5]

    against = [f for f in factors if f["delta_prob"] < 0]
    against.sort(key=lambda x: x["delta_prob"])
    top_against = against[:3]

    # Get top permutation feature dynamically
    top_perm_feature = "training_load"
    model_key = model_config.get("model_key", "logistic_regression")
    perm_file = config.MODELS_DIR / model_key / "permutation_importance.csv"
    if perm_file.exists():
        try:
            perm_df = pd.read_csv(perm_file)
            if not perm_df.empty:
                top_perm_feature = perm_df.iloc[0]["feature"]
        except Exception:
            pass

    top_perm_friendly = FRIENDLY_NAMES.get(top_perm_feature, top_perm_feature)
    importance_sentence = (
        f"Based on global test evaluations, {top_perm_friendly} is the most influential overall factor for this model."
    )

    fatigue_reasons = generate_fatigue_reasons(feature_dict, pred_class, top_towards, workload=workload)
    primary_cause = generate_primary_cause_analysis(feature_dict, pred_class, top_towards, factors, workload=workload)

    return {
        "predicted_class": pred_class,
        "base_prob_pct": round(base_prob * 100, 1),
        "top_towards": top_towards,
        "top_against": top_against,
        "all_factors": factors,
        "importance_sentence": importance_sentence,
        "top_permutation_feature": top_perm_feature,
        "fatigue_reasons": fatigue_reasons,
        "primary_cause": primary_cause,
    }


def generate_primary_cause_analysis(
    feature_dict: Dict[str, Any],
    predicted_class: str,
    top_towards: List[Dict[str, Any]],
    all_factors: List[Dict[str, Any]],
    workload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Identifies the primary feature that caused/affected the fatigue prediction,
    compares it with previous session (#N-1), resting/contrast baseline session,
    and rolling 7-week (7-session) and 14-week (14-session) baseline averages.
    """
    units = {
        "training_duration": "min",
        "training_intensity": "RPE",
        "training_load": "a.u.",
        "heart_rate": "bpm",
        "sleep_quality": "/10",
        "recovery_score": "/100",
        "stress_level": "index",
    }

    top_factor = top_towards[0] if top_towards else (all_factors[0] if all_factors else None)
    if not top_factor:
        return {}

    feat_key = top_factor["feature"]
    feat_name = FRIENDLY_NAMES.get(feat_key, feat_key)
    curr_val = top_factor["actual_value"]
    unit = units.get(feat_key, "")
    delta_pct = top_factor.get("delta_pct", 0.0)

    prev_session = workload.get("previous_session") if workload else None
    prev_num = prev_session.get("session_number") if prev_session else None
    prev_val = prev_session.get(feat_key) if prev_session else None
    prev_diff = (curr_val - prev_val) if (curr_val is not None and prev_val is not None) else None
    prev_pct = ((prev_diff / abs(prev_val)) * 100.0) if (prev_diff is not None and prev_val and prev_val != 0) else None

    # 7-week / 7-session mean
    m7 = (workload.get(f"{feat_key}_mean_7") if feat_key != "training_load" else workload.get("load_mean_7")) if workload else None
    d7 = (curr_val - m7) if (curr_val is not None and m7 is not None) else None
    pct7 = ((d7 / abs(m7)) * 100.0) if (d7 is not None and m7 and m7 != 0) else None

    # 14-week / 14-session mean
    m14 = (workload.get(f"{feat_key}_mean_14") if feat_key != "training_load" else workload.get("load_mean_14")) if workload else None
    d14 = (curr_val - m14) if (curr_val is not None and m14 is not None) else None
    pct14 = ((d14 / abs(m14)) * 100.0) if (d14 is not None and m14 and m14 != 0) else None


    # Contrast session search
    earlier_sessions = workload.get("earlier_sessions", []) if workload else []
    contrast_session = None
    if earlier_sessions:
        valid_earlier = [s for s in earlier_sessions if s.get(feat_key) is not None]
        if valid_earlier:
            if feat_key in ["training_load", "training_duration", "training_intensity", "stress_level"]:
                contrast_session = min(valid_earlier, key=lambda x: x[feat_key])
            else:
                contrast_session = max(valid_earlier, key=lambda x: x[feat_key])

    contrast_num = contrast_session.get("session_number") if contrast_session else None
    contrast_val = contrast_session.get(feat_key) if contrast_session else None
    contrast_diff = (curr_val - contrast_val) if (curr_val is not None and contrast_val is not None) else None
    contrast_pct = ((contrast_diff / abs(contrast_val)) * 100.0) if (contrast_diff is not None and contrast_val and contrast_val != 0) else None

    # Secondary factor
    sec_factor = top_towards[1] if len(top_towards) > 1 else None

    return {
        "feature_key": feat_key,
        "feature_name": feat_name,
        "actual_value": curr_val,
        "unit": unit,
        "predicted_class": predicted_class,
        "delta_pct": delta_pct,
        "prev_session_num": prev_num,
        "prev_session_val": prev_val,
        "prev_session_diff": prev_diff,
        "prev_session_pct": prev_pct,
        "mean_7": m7,
        "diff_7": d7,
        "pct_7": pct7,
        "mean_14": m14,
        "diff_14": d14,
        "pct_14": pct14,
        "contrast_session_num": contrast_num,
        "contrast_session_val": contrast_val,
        "contrast_session_diff": contrast_diff,
        "contrast_session_pct": contrast_pct,
        "secondary_feature": sec_factor,
    }



def compute_what_if(
    base_feature_dict: Dict[str, Any],
    new_sleep: Optional[float] = None,
    new_recovery: Optional[float] = None,
    new_stress: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Re-predicts with user-modified lifestyle variables (sleep, recovery, stress).
    Also searches for the smallest single change that lowers fatigue class.
    """
    model, model_config, selected_meta = ModelLoader.load_selected_model()

    base_pred = predict_session(base_feature_dict, model=model, model_config=model_config)
    current_idx = base_pred["predicted_index"]
    current_class = base_pred["predicted_class"]

    # Modified prediction
    mod_dict = dict(base_feature_dict)
    if new_sleep is not None:
        mod_dict["sleep_quality"] = float(new_sleep)
    if new_recovery is not None:
        mod_dict["recovery_score"] = float(new_recovery)
    if new_stress is not None:
        mod_dict["stress_level"] = float(new_stress)

    mod_pred = predict_session(mod_dict, model=model, model_config=model_config)

    # Search for smallest single change to lower fatigue class
    # Only applicable if current class is High (2) or Moderate (1)
    target_max_idx = current_idx - 1 if current_idx > 0 else 0
    single_change_recommendations = []

    if current_idx > 0:
        # 1. Test Sleep Quality (increase from current to 10.0)
        curr_s = base_feature_dict.get("sleep_quality")
        curr_s_val = float(curr_s) if curr_s is not None and not np.isnan(curr_s) else 5.5
        sleep_grid = np.linspace(curr_s_val, 10.0, 60)
        for s_val in sleep_grid:
            temp_dict = dict(base_feature_dict)
            temp_dict["sleep_quality"] = float(s_val)
            res = predict_session(temp_dict, model=model, model_config=model_config)
            if res["predicted_index"] <= target_max_idx:
                diff = s_val - curr_s_val
                single_change_recommendations.append({
                    "factor": "sleep_quality",
                    "label": "Sleep Quality",
                    "current": round(curr_s_val, 1),
                    "needed": round(float(s_val), 1),
                    "change": f"+{round(diff, 1)} points",
                    "target_class": res["predicted_class"],
                    "abs_effort": diff / (10.0 - 1.16),
                })
                break

        # 2. Test Recovery Score (increase from current to 98.0)
        curr_r = base_feature_dict.get("recovery_score")
        curr_r_val = float(curr_r) if curr_r is not None and not np.isnan(curr_r) else 50.0
        rec_grid = np.linspace(curr_r_val, 98.0, 60)
        for r_val in rec_grid:
            temp_dict = dict(base_feature_dict)
            temp_dict["recovery_score"] = float(r_val)
            res = predict_session(temp_dict, model=model, model_config=model_config)
            if res["predicted_index"] <= target_max_idx:
                diff = r_val - curr_r_val
                single_change_recommendations.append({
                    "factor": "recovery_score",
                    "label": "Recovery Score",
                    "current": round(curr_r_val, 1),
                    "needed": round(float(r_val), 1),
                    "change": f"+{round(diff, 1)} points",
                    "target_class": res["predicted_class"],
                    "abs_effort": diff / (98.0 - 8.9),
                })
                break

        # 3. Test Stress Level (decrease from current to 0.10)
        curr_st = base_feature_dict.get("stress_level")
        curr_st_val = float(curr_st) if curr_st is not None and not np.isnan(curr_st) else 0.50
        stress_grid = np.linspace(curr_st_val, 0.10, 60)
        for st_val in stress_grid:
            temp_dict = dict(base_feature_dict)
            temp_dict["stress_level"] = float(st_val)
            res = predict_session(temp_dict, model=model, model_config=model_config)
            if res["predicted_index"] <= target_max_idx:
                diff = curr_st_val - st_val
                single_change_recommendations.append({
                    "factor": "stress_level",
                    "label": "Stress Level",
                    "current": round(curr_st_val, 2),
                    "needed": round(float(st_val), 2),
                    "change": f"-{round(diff, 2)}",
                    "target_class": res["predicted_class"],
                    "abs_effort": diff / (0.95 - 0.10),
                })
                break

    # Pick the recommendation with smallest relative effort
    best_single_change = None
    if single_change_recommendations:
        single_change_recommendations.sort(key=lambda x: x["abs_effort"])
        best_single_change = single_change_recommendations[0]

    return {
        "base_prediction": base_pred,
        "modified_prediction": mod_pred,
        "best_single_change": best_single_change,
        "all_single_changes": single_change_recommendations,
    }
