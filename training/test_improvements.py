import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from training.features_engineering import RAW_FEATURES, engineer_features

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import StratifiedKFold, KFold, cross_validate, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, r2_score

train_df = pd.read_csv(config.PROCESSED_DATA_DIR / "train.csv")
test_df = pd.read_csv(config.PROCESSED_DATA_DIR / "test.csv")

X_train_raw = train_df[RAW_FEATURES]
y_train_class = train_df['fatigue_class']
y_train_idx = train_df['fatigue_index']

X_train_eng = engineer_features(train_df).drop(columns=['fatigue_index', 'fatigue_class'])
eng_features = [c for c in X_train_eng.columns if c in RAW_FEATURES or c in [
    'load_per_minute', 'intensity_x_duration', 'load_to_recovery_ratio',
    'readiness_balance', 'load_x_stress', 'sleep_x_recovery', 'log_training_load'
]]
X_train_eng = X_train_eng[eng_features]

print(f"Engineered features list ({len(eng_features)}): {eng_features}")

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
kf5 = KFold(n_splits=5, shuffle=True, random_state=42)

improvements = []

# 1. Baseline Logistic Regression
pipe_lr_base = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000, random_state=42))
])
res = cross_validate(pipe_lr_base, X_train_raw, y_train_class, cv=cv5, scoring=['accuracy', 'f1_macro'])
acc_lr_base = res['test_accuracy'].mean()
f1_lr_base = res['test_f1_macro'].mean()
improvements.append(('LR Baseline (7 features)', f1_lr_base, acc_lr_base))
print(f"LR Baseline: CV Acc={acc_lr_base:.4f}, CV F1={f1_lr_base:.4f}")

# 2. LR with Feature Engineering + Tuning
pipe_lr_eng = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(C=0.5, max_iter=1000, random_state=42))
])
res = cross_validate(pipe_lr_eng, X_train_eng, y_train_class, cv=cv5, scoring=['accuracy', 'f1_macro'])
acc_lr_eng = res['test_accuracy'].mean()
f1_lr_eng = res['test_f1_macro'].mean()
improvements.append(('LR + Feature Engineering (C=0.5)', f1_lr_eng, acc_lr_eng))
print(f"LR + Feature Engineering: CV Acc={acc_lr_eng:.4f}, CV F1={f1_lr_eng:.4f}")

# 3. LR with Polynomial Interaction (degree=2)
pipe_lr_poly = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(C=0.1, max_iter=2000, random_state=42))
])
res = cross_validate(pipe_lr_poly, X_train_raw, y_train_class, cv=cv5, scoring=['accuracy', 'f1_macro'])
acc_lr_poly = res['test_accuracy'].mean()
f1_lr_poly = res['test_f1_macro'].mean()
improvements.append(('LR + Polynomial Features (deg=2, C=0.1)', f1_lr_poly, acc_lr_poly))
print(f"LR + Poly (deg=2): CV Acc={acc_lr_poly:.4f}, CV F1={f1_lr_poly:.4f}")

# 4. RF Baseline
pipe_rf_base = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('clf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1))
])
res = cross_validate(pipe_rf_base, X_train_raw, y_train_class, cv=cv5, scoring=['accuracy', 'f1_macro'])
acc_rf_base = res['test_accuracy'].mean()
f1_rf_base = res['test_f1_macro'].mean()
improvements.append(('RF Baseline (7 features, depth=10)', f1_rf_base, acc_rf_base))
print(f"RF Baseline: CV Acc={acc_rf_base:.4f}, CV F1={f1_rf_base:.4f}")

# 5. RF with Feature Engineering + Tuning
pipe_rf_eng = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('clf', RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=4, random_state=42, n_jobs=-1))
])
res = cross_validate(pipe_rf_eng, X_train_eng, y_train_class, cv=cv5, scoring=['accuracy', 'f1_macro'])
acc_rf_eng = res['test_accuracy'].mean()
f1_rf_eng = res['test_f1_macro'].mean()
improvements.append(('RF + Feature Engineering (depth=8, leaf=4)', f1_rf_eng, acc_rf_eng))
print(f"RF + Feature Engineering: CV Acc={acc_rf_eng:.4f}, CV F1={f1_rf_eng:.4f}")

# 6. XGB Baseline
pipe_xgb_base = Pipeline([
    ('clf', XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, eval_metric='mlogloss'))
])
res = cross_validate(pipe_xgb_base, X_train_raw, y_train_class, cv=cv5, scoring=['accuracy', 'f1_macro'])
acc_xgb_base = res['test_accuracy'].mean()
f1_xgb_base = res['test_f1_macro'].mean()
improvements.append(('XGB Baseline (7 features, depth=4, lr=0.05)', f1_xgb_base, acc_xgb_base))
print(f"XGB Baseline: CV Acc={acc_xgb_base:.4f}, CV F1={f1_xgb_base:.4f}")

# 7. XGB with Feature Engineering + Tuning
pipe_xgb_eng = Pipeline([
    ('clf', XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.04, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='mlogloss'))
])
res = cross_validate(pipe_xgb_eng, X_train_eng, y_train_class, cv=cv5, scoring=['accuracy', 'f1_macro'])
acc_xgb_eng = res['test_accuracy'].mean()
f1_xgb_eng = res['test_f1_macro'].mean()
improvements.append(('XGB + Feature Engineering (depth=3, lr=0.04)', f1_xgb_eng, acc_xgb_eng))
print(f"XGB + Feature Engineering: CV Acc={acc_xgb_eng:.4f}, CV F1={f1_xgb_eng:.4f}")

# 8. Regression-then-Bin Approach
pipe_reg = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('reg', GradientBoostingRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42))
])
preds_idx = cross_val_predict(pipe_reg, X_train_eng, y_train_idx, cv=kf5)
preds_bin = pd.cut(preds_idx, bins=[-np.inf, 47.26, 68.89, np.inf], labels=[0, 1, 2]).astype(int)
acc_reg_bin = accuracy_score(y_train_class, preds_bin)
f1_reg_bin = f1_score(y_train_class, preds_bin, average='macro')
improvements.append(('Regression-then-Bin (GBR on engineered features)', f1_reg_bin, acc_reg_bin))
print(f"Regression-then-Bin: CV Acc={acc_reg_bin:.4f}, CV F1={f1_reg_bin:.4f}")

# Save improvement log
imp_df = pd.DataFrame(improvements, columns=['change', 'cv_macro_f1', 'cv_accuracy'])
imp_df.to_csv(config.REPORTS_DIR / 'improvement_log.csv', index=False)
print("\nSaved reports/improvement_log.csv:")
print(imp_df.to_string(index=False))
