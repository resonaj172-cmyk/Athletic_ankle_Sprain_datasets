"""
Feature Engineering module for Athlete Fatigue Prediction.
Contains pure transformations derived ONLY from the 7 input features:
- training_duration
- training_intensity
- training_load
- heart_rate
- sleep_quality
- recovery_score
- stress_level

All transformations are deterministic and rely only on the single row / session inputs.
"""
import numpy as np
import pandas as pd

RAW_FEATURES = [
    "training_duration",
    "training_intensity",
    "training_load",
    "heart_rate",
    "sleep_quality",
    "recovery_score",
    "stress_level",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes domain-specific interaction and non-linear features
    strictly from the 7 base input features.
    """
    df = df.copy()

    # 1. Load per minute (efficiency / density of work)
    dur = df["training_duration"].replace(0, np.nan)
    df["load_per_minute"] = df["training_load"] / dur

    # 2. Intensity x Duration (theoretical raw workload)
    df["intensity_x_duration"] = df["training_intensity"] * df["training_duration"]

    # 3. Workload vs Recovery balance
    # Higher load with lower recovery amplifies fatigue
    rec = df["recovery_score"].clip(lower=1.0)
    df["load_to_recovery_ratio"] = df["training_load"] / rec

    # 4. Recovery minus Stress index (Overall readiness index)
    # Recovery is 0-100, stress is 0.0-1.0 (scale stress to 100)
    df["readiness_balance"] = df["recovery_score"] - (df["stress_level"] * 100.0)

    # 5. Stress multiplied by Training Load (psychophysiological load)
    df["load_x_stress"] = df["training_load"] * df["stress_level"]

    # 6. Sleep-weighted recovery
    df["sleep_x_recovery"] = df["sleep_quality"] * df["recovery_score"]

    # 7. Non-linear log transform of skewed training_load
    df["log_training_load"] = np.log1p(df["training_load"].clip(lower=0))

    return df
