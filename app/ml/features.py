import math
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any

# Field specifications: Hard bounds (reject), Training range (warn only), Required flag
FIELD_SPECS = {
    "training_duration": {
        "label": "Training Duration",
        "unit": "min",
        "required": True,
        "hard_min": 1.0,
        "hard_max": 600.0,
        "train_min": 30.0,
        "train_max": 180.0,
    },
    "training_intensity": {
        "label": "Training Intensity",
        "unit": "RPE (0-10)",
        "required": False,
        "hard_min": 0.0,
        "hard_max": 10.0,
        "train_min": 2.0,
        "train_max": 10.0,
    },
    "training_load": {
        "label": "Training Load",
        "unit": "a.u.",
        "required": True,
        "hard_min": 0.0,
        "hard_max": 10000.0,
        "train_min": 150.0,
        "train_max": 2632.64,
    },
    "heart_rate": {
        "label": "Heart Rate",
        "unit": "bpm",
        "required": False,
        "hard_min": 30.0,
        "hard_max": 220.0,
        "train_min": 40.0,
        "train_max": 160.0,
    },
    "sleep_quality": {
        "label": "Sleep Quality",
        "unit": "scale 0-10",
        "required": False,
        "hard_min": 0.0,
        "hard_max": 10.0,
        "train_min": 1.16,
        "train_max": 10.0,
    },
    "recovery_score": {
        "label": "Recovery Score",
        "unit": "scale 0-100",
        "required": True,
        "hard_min": 0.0,
        "hard_max": 100.0,
        "train_min": 8.90,
        "train_max": 98.0,
    },
    "stress_level": {
        "label": "Stress Level",
        "unit": "scale 0.0-1.0",
        "required": True,
        "hard_min": 0.0,
        "hard_max": 1.0,
        "train_min": 0.10,
        "train_max": 0.95,
    },
}


def validate_session_inputs(form_data: Dict[str, Any]) -> Tuple[Dict[str, float], List[str], List[str]]:
    """
    Validates form data against hard limits and training ranges.
    Returns:
        parsed_values: dict with exact float values (or np.nan for blank optional fields)
        errors: list of error strings (hard validation failures)
        warnings: list of warning strings (values outside training distribution)
    """
    parsed = {}
    errors = []
    warnings = []

    for field_name, spec in FIELD_SPECS.items():
        raw_val = form_data.get(field_name)

        if raw_val is None or str(raw_val).strip() == "":
            if spec["required"]:
                errors.append(f"{spec['label']} is required.")
                parsed[field_name] = np.nan
            else:
                parsed[field_name] = np.nan
            continue

        try:
            val = float(raw_val)
        except (ValueError, TypeError):
            errors.append(f"Invalid numeric value for {spec['label']}.")
            parsed[field_name] = np.nan
            continue

        # Hard bounds check
        if val < spec["hard_min"] or val > spec["hard_max"]:
            errors.append(
                f"{spec['label']} ({val}) is outside permissible limits [{spec['hard_min']} - {spec['hard_max']}]."
            )
            parsed[field_name] = val
            continue

        # Soft bounds (training range) check
        if val < spec["train_min"] or val > spec["train_max"]:
            warnings.append(
                f"{spec['label']} ({val}) is outside the training range [{spec['train_min']} - {spec['train_max']}]; prediction may be less reliable."
            )

        parsed[field_name] = val

    return parsed, errors, warnings


def compute_history_summary(
    earlier_sessions: List[Dict[str, Any]],
    current_training_load: float,
    windows: Tuple[int, int] = (7, 14),
    current_session_dict: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Computes display-only historical workload context using only earlier sessions.
    Current session is strictly excluded from earlier_sessions.

    Rules:
    - For w in (7, 14), last_w = previous w earlier sessions.
    - load_sum_w and load_mean_w are None unless all w sessions exist.
    - load_minus_mean_w = current_training_load - load_mean_w.
    - load_ratio_w = current_training_load / load_mean_w.
    - For other variables: mean over last_w ignoring None/NaN; None if fewer than 3 values present.
    - Also includes previous session metrics and comprehensive 7/14 session comparison breakdown.
    """
    summary = {}
    n_sessions = len(earlier_sessions)

    # Store previous session (Session #N-1)
    prev_session = earlier_sessions[-1] if n_sessions > 0 else None
    summary["previous_session"] = prev_session
    summary["earlier_sessions"] = earlier_sessions

    for w in windows:
        w_key = str(w)
        window_slice = earlier_sessions[-w:] if n_sessions >= w else earlier_sessions
        if not window_slice:
            summary[f"has_window_{w_key}"] = False
            summary[f"load_sum_{w_key}"] = None
            summary[f"load_mean_{w_key}"] = None
            summary[f"load_minus_mean_{w_key}"] = None
            summary[f"load_ratio_{w_key}"] = None
            for var in ["training_duration", "training_intensity", "heart_rate", "sleep_quality", "recovery_score", "stress_level"]:
                summary[f"{var}_mean_{w_key}"] = None
                summary[f"{var}_diff_{w_key}"] = None
            continue

        summary[f"has_window_{w_key}"] = True

        # Load sum and mean
        loads = [s["training_load"] for s in window_slice if s.get("training_load") is not None and not (isinstance(s["training_load"], float) and math.isnan(s["training_load"]))]
        if len(loads) > 0:
            l_sum = float(sum(loads))
            l_mean = float(l_sum / len(loads))
            summary[f"load_sum_{w_key}"] = l_sum
            summary[f"load_mean_{w_key}"] = l_mean
            summary[f"load_minus_mean_{w_key}"] = float(current_training_load - l_mean)
            summary[f"load_ratio_{w_key}"] = float(current_training_load / l_mean) if l_mean > 0 else None
        else:
            summary[f"load_sum_{w_key}"] = None
            summary[f"load_mean_{w_key}"] = None
            summary[f"load_minus_mean_{w_key}"] = None
            summary[f"load_ratio_{w_key}"] = None

        # Other metrics averages (computed from available observations in the window; minimum min(3, window_size) required)
        min_required = min(3, len(window_slice))
        for var in ["training_duration", "training_intensity", "heart_rate", "sleep_quality", "recovery_score", "stress_level"]:
            vals = [
                s[var] for s in window_slice
                if s.get(var) is not None and not (isinstance(s[var], float) and math.isnan(s[var]))
            ]
            if len(vals) >= min_required:
                m_val = float(sum(vals) / len(vals))
                summary[f"{var}_mean_{w_key}"] = m_val
                if current_session_dict and current_session_dict.get(var) is not None:
                    curr = current_session_dict[var]
                    if not (isinstance(curr, float) and math.isnan(curr)):
                        summary[f"{var}_diff_{w_key}"] = float(curr) - m_val
                    else:
                        summary[f"{var}_diff_{w_key}"] = None
                else:
                    summary[f"{var}_diff_{w_key}"] = None
            else:
                summary[f"{var}_mean_{w_key}"] = None
                summary[f"{var}_diff_{w_key}"] = None

    # Compute detailed parameter comparisons if current session dict is available
    param_comparisons = []
    all_vars = [
        ("training_load", "Training Load", "a.u."),
        ("training_duration", "Training Duration", "min"),
        ("training_intensity", "Training Intensity", "RPE"),
        ("heart_rate", "Heart Rate", "bpm"),
        ("sleep_quality", "Sleep Quality", "/10"),
        ("recovery_score", "Recovery Score", "/100"),
        ("stress_level", "Stress Level", "index"),
    ]

    for var_key, label, unit in all_vars:
        curr_val = None
        if current_session_dict:
            val = current_session_dict.get(var_key)
            if val is not None and not (isinstance(val, float) and math.isnan(val)):
                curr_val = float(val)

        prev_val = None
        prev_diff = None
        prev_pct = None
        if prev_session:
            pv = prev_session.get(var_key)
            if pv is not None and not (isinstance(pv, float) and math.isnan(pv)):
                prev_val = float(pv)
                if curr_val is not None:
                    prev_diff = curr_val - prev_val
                    if prev_val != 0:
                        prev_pct = (prev_diff / abs(prev_val)) * 100.0

        # Window 7 stats
        m7 = summary.get(f"{var_key}_mean_7") if var_key != "training_load" else summary.get("load_mean_7")
        d7 = summary.get(f"{var_key}_diff_7") if var_key != "training_load" else summary.get("load_minus_mean_7")
        pct7 = None
        if curr_val is not None and m7 is not None and m7 != 0:
            pct7 = ((curr_val - m7) / abs(m7)) * 100.0

        # Window 14 stats
        m14 = summary.get(f"{var_key}_mean_14") if var_key != "training_load" else summary.get("load_mean_14")
        d14 = summary.get(f"{var_key}_diff_14") if var_key != "training_load" else summary.get("load_minus_mean_14")
        pct14 = None
        if curr_val is not None and m14 is not None and m14 != 0:
            pct14 = ((curr_val - m14) / abs(m14)) * 100.0

        param_comparisons.append({
            "key": var_key,
            "label": label,
            "unit": unit,
            "current_val": curr_val,
            "prev_val": prev_val,
            "prev_diff": prev_diff,
            "prev_pct": prev_pct,
            "mean_7": m7,
            "diff_7": d7,
            "pct_7": pct7,
            "mean_14": m14,
            "diff_14": d14,
            "pct_14": pct14,
        })

    summary["parameter_comparisons"] = param_comparisons
    return summary

