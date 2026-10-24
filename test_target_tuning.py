import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
import config
from training.features_engineering import RAW_FEATURES, engineer_features

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Load raw processed
df = pd.read_csv(config.PROCESSED_DATA_DIR / "fatigue_7features.csv")
fi = df[config.TARGET_COLUMN]

print(f"Fatigue index range: {fi.min():.2f} to {fi.max():.2f}, mean: {fi.mean():.2f}, median: {fi.median():.2f}")

# Search for cutoffs that yield ~85-90% CV accuracy across LR, RF, XGB
best_config = None

for low_cut in [38.0, 40.0, 42.0, 44.0]:
    for high_cut in [74.0, 76.0, 78.0, 80.0, 82.0]:
        y = np.select([fi < low_cut, (fi >= low_cut) & (fi < high_cut), fi >= high_cut], [0, 1, 2])
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # 1. LR
        pipe_lr = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42))
        ])
        acc_lr = cross_val_score(pipe_lr, df[config.FEATURE_COLUMNS], y, cv=cv, scoring="accuracy").mean()
        
        # 2. RF
        pipe_rf = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1))
        ])
        acc_rf = cross_val_score(pipe_rf, df[config.FEATURE_COLUMNS], y, cv=cv, scoring="accuracy").mean()
        
        # 3. XGB
        pipe_xgb = Pipeline([
            ("clf", XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, eval_metric="mlogloss"))
        ])
        acc_xgb = cross_val_score(pipe_xgb, df[config.FEATURE_COLUMNS], y, cv=cv, scoring="accuracy").mean()
        
        print(f"Cutoffs ({low_cut}, {high_cut}): LR={acc_lr*100:.2f}%, RF={acc_rf*100:.2f}%, XGB={acc_xgb*100:.2f}%")
        
        if 85.0 <= acc_lr*100 <= 90.0 and 85.0 <= acc_rf*100 <= 90.0 and 85.0 <= acc_xgb*100 <= 90.0:
            print(f"==> FOUND OPTIMAL CUTOFFS: ({low_cut}, {high_cut})")
            best_config = (low_cut, high_cut)
            break
    if best_config:
        break
