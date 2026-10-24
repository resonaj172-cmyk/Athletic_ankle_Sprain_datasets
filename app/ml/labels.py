"""
State labels for sleep, recovery, and stress.
Fixed rules for display only (not machine learning).
Equal-width thirds of each variable's observed range in the processed dataset.
"""

LABELS_CONFIG = {
    "sleep_quality": {
        "name": "Sleep Quality",
        "unit": "/ 10",
        "poor_max": 4.0,       # below 4.0 = Poor
        "moderate_max": 7.0,   # 4.0 to < 7.0 = Moderate, >= 7.0 = Good
        "poor_label": "Poor",
        "moderate_label": "Moderate",
        "good_label": "Good",
    },
    "recovery_score": {
        "name": "Recovery Score",
        "unit": "/ 100",
        "poor_max": 40.0,      # below 40.0 = Poor
        "moderate_max": 70.0,  # 40.0 to < 70.0 = Moderate, >= 70.0 = Good
        "poor_label": "Poor",
        "moderate_label": "Moderate",
        "good_label": "Good",
    },
    "stress_level": {
        "name": "Stress Level",
        "unit": "(0-1 scale)",
        "low_max": 0.40,       # below 0.40 = Low (Good)
        "moderate_max": 0.67,  # 0.40 to < 0.67 = Moderate, >= 0.67 = High (Poor)
        "poor_label": "High",
        "moderate_label": "Moderate",
        "good_label": "Low",
    },
}


def get_sleep_label(val):
    if val is None or val == "" or str(val).lower() == "nan":
        return "Not provided", "secondary"
    try:
        v = float(val)
    except (ValueError, TypeError):
        return "Not provided", "secondary"
    if v < 4.0:
        return "Poor", "danger"
    elif v < 7.0:
        return "Moderate", "warning"
    else:
        return "Good", "success"


def get_recovery_label(val):
    if val is None or val == "" or str(val).lower() == "nan":
        return "Not provided", "secondary"
    try:
        v = float(val)
    except (ValueError, TypeError):
        return "Not provided", "secondary"
    if v < 40.0:
        return "Poor", "danger"
    elif v < 70.0:
        return "Moderate", "warning"
    else:
        return "Good", "success"


def get_stress_label(val):
    if val is None or val == "" or str(val).lower() == "nan":
        return "Not provided", "secondary"
    try:
        v = float(val)
    except (ValueError, TypeError):
        return "Not provided", "secondary"
    if v >= 0.67:
        return "High", "danger"
    elif v >= 0.40:
        return "Moderate", "warning"
    else:
        return "Low", "success"


def get_all_state_labels(sleep, recovery, stress):
    s_label, s_badge = get_sleep_label(sleep)
    r_label, r_badge = get_recovery_label(recovery)
    st_label, st_badge = get_stress_label(stress)

    return {
        "sleep": {"label": s_label, "badge": s_badge, "value": sleep},
        "recovery": {"label": r_label, "badge": r_badge, "value": recovery},
        "stress": {"label": st_label, "badge": st_badge, "value": stress},
    }
