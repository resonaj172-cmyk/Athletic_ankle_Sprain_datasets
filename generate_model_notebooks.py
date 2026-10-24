"""
Builds and EXECUTES the three model notebooks cell by cell.
Every cell's printed output stored in the .ipynb is captured from really executing that cell's code,
so nothing is typed by hand. Run:  python generate_model_notebooks.py
"""
import base64
import contextlib
import io
import json
import os
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))
import config

NB_DIR = BASE_DIR / "notebooks"

# ---------------------------------------------------------------- per-model blocks
MODELS = {
    "logistic_regression": {
        "file": "01_logistic_regression.ipynb",
        "title": "Logistic Regression",
        "imports": r'''from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression''',
        "build": r'''est = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),   # fitted on training folds only
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
])
print(est)''',
        "grid": r'''param_grid = {
    "classifier__C": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
    "classifier__class_weight": [None, "balanced"],
    "classifier__solver": ["lbfgs"],
}''',
        "prefix": "classifier__",
        "algorithm": "LogisticRegression",
        "version_code": "import sklearn; LIB = {'sklearn': sklearn.__version__}",
        "save_extra": "",
    },
    "random_forest": {
        "file": "02_random_forest.ipynb",
        "title": "Random Forest",
        "imports": r'''from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier''',
        "build": r'''est = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),   # fitted on training folds only
    ("classifier", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
])
print(est)''',
        "grid": r'''param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [6, 8, 12, None],
    "classifier__min_samples_split": [2, 5],
    "classifier__min_samples_leaf": [1, 2, 4],
}''',
        "prefix": "classifier__",
        "algorithm": "RandomForestClassifier",
        "version_code": "import sklearn; LIB = {'sklearn': sklearn.__version__}",
        "save_extra": "",
    },
    "xgboost": {
        "file": "03_xgboost.ipynb",
        "title": "XGBoost",
        "imports": r'''from xgboost import XGBClassifier''',
        "build": r'''est = XGBClassifier(            # NaN is passed straight through (no imputer)
    objective="multi:softprob", eval_metric="mlogloss",
    random_state=RANDOM_STATE, n_jobs=-1,
)
print(est)''',
        "grid": r'''param_grid = {
    "n_estimators": [100, 150, 200],
    "max_depth": [3, 4, 6],
    "learning_rate": [0.03, 0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}''',
        "prefix": "",
        "algorithm": "XGBClassifier",
        "version_code": "import xgboost; LIB = {'xgboost': xgboost.__version__}",
        "save_extra": 'final_model.save_model(str(MODEL_DIR / "model.json"))',
    },
}

# ---------------------------------------------------------------- shared cell templates
def cells_for(key, m):
    name = key
    c = []

    c.append(("# 1. Imports and settings\nImports, `random_state=42`, and paths from `config.py`.", r'''import sys, time, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
warnings.filterwarnings("ignore")

sys.path.append(str(Path.cwd().parent))
import config

__IMPORTS__
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             precision_recall_fscore_support, classification_report, confusion_matrix,
                             roc_curve, auc, ConfusionMatrixDisplay)
from sklearn.inspection import permutation_importance
import joblib

RANDOM_STATE = config.RANDOM_STATE
MODEL_NAME = "__NAME__"
MODEL_DIR = config.MODELS_DIR / MODEL_NAME
MODEL_DIR.mkdir(parents=True, exist_ok=True)
config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
print(f"{MODEL_NAME}: random_state={RANDOM_STATE}")'''))

    c.append(("# 2. Load the 70/30 split\nLoad `train.csv` / `test.csv`; print shapes and class counts. `athlete_id` is kept only to group the CV folds.", r'''train_df = pd.read_csv(config.PROCESSED_DATA_DIR / "train.csv")
test_df = pd.read_csv(config.PROCESSED_DATA_DIR / "test.csv")
n = len(train_df) + len(test_df)
print(f"Total rows {n} | train {len(train_df)} ({len(train_df)/n:.1%}) | test {len(test_df)} ({len(test_df)/n:.1%})")
for nm, part in (("Train", train_df), ("Test", test_df)):
    vc = part[config.CLASS_COLUMN].value_counts().sort_index()
    print(nm, {config.CLASS_NAMES[k]: f"{v} ({v/len(part):.1%})" for k, v in vc.items()})'''))

    c.append(("# 3. Assert features, leakage safety and the shared split\nExactly the 7 inputs, no target/id column in X, and the same rows as `split_indices.json`.", r'''FEATURE_COLS = list(config.FEATURE_COLUMNS)
assert len(FEATURE_COLS) == 7, FEATURE_COLS
forbidden = {config.TARGET_COLUMN, config.CLASS_COLUMN, "athlete_id", "session_id"}
assert not forbidden & set(FEATURE_COLS), "target/id column found in features!"

split = json.loads((config.PROCESSED_DATA_DIR / "split_indices.json").read_text())
full = pd.read_csv(config.PROCESSED_DATA_DIR / "fatigue_7features.csv")
assert np.array_equal(full.iloc[split["train_indices"]][config.TARGET_COLUMN].values, train_df[config.TARGET_COLUMN].values)
assert np.array_equal(full.iloc[split["test_indices"]][config.TARGET_COLUMN].values, test_df[config.TARGET_COLUMN].values)
assert not set(train_df["athlete_id"]) & set(test_df["athlete_id"]), "athlete appears in both sets"

X_train, y_train = train_df[FEATURE_COLS], train_df[config.CLASS_COLUMN]
X_test, y_test = test_df[FEATURE_COLS], test_df[config.CLASS_COLUMN]
GROUPS = train_df["athlete_id"]
print("Features (in order):", FEATURE_COLS)
print("Split matches split_indices.json; no athlete in both train and test.")'''))

    c.append(("# 4. Preprocessing + model pipeline\n" + {
        "logistic_regression": "Median imputer (fit on train folds only) -> StandardScaler -> Logistic Regression.",
        "random_forest": "Median imputer (fit on train folds only) -> Random Forest.",
        "xgboost": "XGBoost takes NaN directly, so there is no imputer.",
    }[key], r'''__BUILD__'''))

    c.append(("# 5. Hyperparameter search (5-fold stratified, athlete-grouped CV on train only)\nScoring macro-F1; accuracy is recorded too. Saves `best_params.json`.", r'''__GRID__

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
t0 = time.perf_counter()
grid_search = GridSearchCV(est, param_grid, cv=cv, scoring={"f1_macro": "f1_macro", "accuracy": "accuracy"},
                           refit="f1_macro", n_jobs=-1)
grid_search.fit(X_train, y_train, groups=GROUPS)
search_time = time.perf_counter() - t0

best_cv_f1 = float(grid_search.best_score_)
best_cv_acc = float(grid_search.cv_results_["mean_test_accuracy"][grid_search.best_index_])
best_params = {k.replace("__PREFIX__", ""): v for k, v in grid_search.best_params_.items()}
print(f"Best CV macro-F1 : {best_cv_f1:.4f}")
print(f"CV accuracy      : {best_cv_acc:.4f}")
print(f"Best parameters  : {best_params}")
print(f"Search wall time : {search_time:.1f}s")
with open(MODEL_DIR / "best_params.json", "w") as f:
    json.dump(best_params, f, indent=2, default=str)
print("Saved best_params.json")'''))

    c.append(("# 6. Fit final model on the full training set\nTimed with `time.perf_counter()` (wall time).", r'''final_model = grid_search.best_estimator_
t0 = time.perf_counter()
final_model.fit(X_train, y_train)
fit_time = time.perf_counter() - t0
total_training_time = search_time + fit_time
print(f"Final fit (fit only): {fit_time:.4f}s | search + fit: {total_training_time:.1f}s")'''))

    c.append(("# 7. Training accuracy vs CV accuracy (overfitting check)\nA large positive gap means variance; both low means weak signal.", r'''train_acc = float(accuracy_score(y_train, final_model.predict(X_train)))
cv_acc = best_cv_acc
gap = train_acc - cv_acc
print(f"Train accuracy: {train_acc:.4f}")
print(f"CV accuracy   : {cv_acc:.4f}")
print(f"Gap (train - CV): {gap:+.4f}")'''))

    c.append(("# 8. FINAL TEST EVALUATION (the only cell that scores the model on test.csv)\nAccuracy, macro P/R/F1, weighted F1, ROC-AUC (OvR), per-class P/R/F1. Saves `metrics.csv`.", r'''y_pred = final_model.predict(X_test)
y_proba = final_model.predict_proba(X_test)

test_acc = float(accuracy_score(y_test, y_pred))
p_mac = float(precision_score(y_test, y_pred, average="macro"))
r_mac = float(recall_score(y_test, y_pred, average="macro"))
f1_mac = float(f1_score(y_test, y_pred, average="macro"))
f1_w = float(f1_score(y_test, y_pred, average="weighted"))
auc_ovr = float(roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro"))
p_c, r_c, f_c, _ = precision_recall_fscore_support(y_test, y_pred)

print(f"Accuracy {test_acc:.4f} | macro P {p_mac:.4f} R {r_mac:.4f} F1 {f1_mac:.4f} | weighted F1 {f1_w:.4f} | ROC-AUC {auc_ovr:.4f}")
print(classification_report(y_test, y_pred, target_names=config.CLASS_NAMES, digits=4))
if test_acc > 0.95 or test_acc - cv_acc > 0.03:
    print("WARNING: suspicion rule triggered (test > 0.95 or test exceeds CV by > 3 points). Re-check for leakage.")

row = {"model_name": MODEL_NAME, "accuracy": test_acc, "test_accuracy": test_acc,
       "precision_macro": p_mac, "recall_macro": r_mac, "f1_macro": f1_mac, "macro_f1": f1_mac,
       "f1_weighted": f1_w, "weighted_f1": f1_w, "roc_auc_ovr": auc_ovr, "roc_auc_macro_ovr": auc_ovr,
       "cv_macro_f1": best_cv_f1, "cv_accuracy": cv_acc, "train_accuracy": train_acc,
       "overfit_gap": gap, "meets_target": bool(test_acc >= 0.80), "in_range_80_85": bool(0.80 <= test_acc <= 0.85)}
for i, cn in enumerate(config.CLASS_NAMES):
    row[f"precision_{cn}"], row[f"recall_{cn}"], row[f"f1_{cn}"] = float(p_c[i]), float(r_c[i]), float(f_c[i])
pd.DataFrame([row]).to_csv(MODEL_DIR / "metrics.csv", index=False)
print("Saved metrics.csv")'''))

    c.append(("# 9. Confusion matrix and ROC curves\nSaved to `reports/figures/`.", r'''fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=config.CLASS_NAMES).plot(ax=axes[0], cmap="Blues", colorbar=False)
axes[0].set_title(f"Confusion matrix - {MODEL_NAME}")
for i, cn in enumerate(config.CLASS_NAMES):
    fpr, tpr, _ = roc_curve(y_test == i, y_proba[:, i])
    axes[1].plot(fpr, tpr, lw=2, label=f"{cn} (AUC={auc(fpr, tpr):.3f})")
axes[1].plot([0, 1], [0, 1], "k--", lw=1)
axes[1].set(title=f"One-vs-rest ROC - {MODEL_NAME}", xlabel="False positive rate", ylabel="True positive rate")
axes[1].legend(loc="lower right"); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(config.FIGURES_DIR / f"{MODEL_NAME}_evaluation.png", dpi=150)
plt.show(); plt.close()
print("Saved", config.FIGURES_DIR / f"{MODEL_NAME}_evaluation.png")'''))

    c.append(("# 10. Permutation importance (10 repeats)\nMacro-F1 drop when each feature is shuffled; saved as CSV and bar plot.", r'''perm = permutation_importance(final_model, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE,
                              scoring="f1_macro", n_jobs=-1)
perm_df = pd.DataFrame({"feature": FEATURE_COLS, "importance_mean": perm.importances_mean,
                        "importance_std": perm.importances_std}).sort_values("importance_mean", ascending=False)
perm_df.to_csv(MODEL_DIR / "permutation_importance.csv", index=False)
print(perm_df.to_string(index=False))
plt.figure(figsize=(8, 4.5))
d = perm_df.iloc[::-1]
plt.barh(d["feature"], d["importance_mean"], xerr=d["importance_std"], color="#3b82f6")
plt.title(f"Permutation importance (test) - {MODEL_NAME}"); plt.xlabel("Mean drop in macro-F1")
plt.tight_layout()
plt.savefig(config.FIGURES_DIR / f"{MODEL_NAME}_permutation_importance.png", dpi=150)
plt.show(); plt.close()'''))

    c.append(("# 11. Prediction time per 1,000 rows\nMeasured on 1,000 training rows (no test data needed); saves `wall_time.csv`.", r'''bench = X_train.sample(n=1000, random_state=RANDOM_STATE)
final_model.predict(bench)  # warm-up
t0 = time.perf_counter()
for _ in range(10):
    final_model.predict(bench)
pred_ms = (time.perf_counter() - t0) / 10 * 1000
pd.DataFrame([{"model_name": MODEL_NAME, "search_wall_time_seconds": search_time,
               "fit_wall_time_seconds": fit_time, "total_training_wall_time_s": total_training_time,
               "prediction_time_per_1k_ms": pred_ms}]).to_csv(MODEL_DIR / "wall_time.csv", index=False)
print(f"Search {search_time:.1f}s | fit only {fit_time:.4f}s | total {total_training_time:.1f}s | predict/1k rows {pred_ms:.2f} ms")'''))

    c.append(("# 12. Save model and `model_config.json`\nFeature list and order, class names, library versions.", r'''import platform
__VERSION__
joblib.dump(final_model, MODEL_DIR / "model.joblib")
__SAVE_EXTRA__
model_config = {"model_name": MODEL_NAME, "algorithm": "__ALGO__", "feature_names": FEATURE_COLS,
                "class_names": config.CLASS_NAMES, "random_state": RANDOM_STATE,
                "split": "70/30 athlete-grouped stratified", "python": platform.python_version(),
                "pandas": pd.__version__, "numpy": np.__version__, **LIB, "best_hyperparameters": best_params}
with open(MODEL_DIR / "model_config.json", "w") as f:
    json.dump(model_config, f, indent=2, default=str)
print("Saved model.joblib and model_config.json to", MODEL_DIR)'''))

    c.append(("# 13. Summary\nReal numbers, printed whether or not they fall in 0.80-0.85.", r'''in_range = 0.80 <= test_acc <= 0.85
print(f"[{MODEL_NAME}] accuracy={test_acc:.4f} | macro-F1={f1_mac:.4f} | ROC-AUC={auc_ovr:.4f} | "
      f"training time={total_training_time:.1f}s | within 0.80-0.85: {in_range}")'''))

    out = []
    for md, code in c:
        code = (code.replace("__IMPORTS__", m["imports"]).replace("__NAME__", name)
                    .replace("__BUILD__", m["build"]).replace("__GRID__", m["grid"])
                    .replace("__PREFIX__", m["prefix"]).replace("__ALGO__", m["algorithm"])
                    .replace("__VERSION__", m["version_code"]).replace("__SAVE_EXTRA__", m["save_extra"]))
        out.append((md, code))
    return out


def lines(s):
    parts = s.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


def build_and_run(key, m):
    ns = {"__name__": "__main__"}
    cells, count = [], 0
    prev = os.getcwd()
    os.chdir(NB_DIR)  # notebooks use Path.cwd().parent as project root
    try:
        for md, code in cells_for(key, m):
            count += 1
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                exec(compile(code, f"<{key} cell {count}>", "exec"), ns)
            outputs = []
            text = buf.getvalue()
            if text:
                outputs.append({"output_type": "stream", "name": "stdout", "text": lines(text.rstrip("\n") + "\n")})
            img = {9: f"{key}_evaluation.png", 10: f"{key}_permutation_importance.png"}.get(count)
            if img and (config.FIGURES_DIR / img).exists():
                b64 = base64.b64encode((config.FIGURES_DIR / img).read_bytes()).decode()
                outputs.append({"output_type": "display_data", "metadata": {},
                                "data": {"image/png": b64, "text/plain": ["<Figure>"]}})
            cells.append({"cell_type": "markdown", "metadata": {}, "source": lines(md)})
            cells.append({"cell_type": "code", "execution_count": count, "metadata": {},
                          "outputs": outputs, "source": lines(code)})
    finally:
        os.chdir(prev)
    nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}}}
    (NB_DIR / m["file"]).write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"Saved {NB_DIR / m['file']}")


if __name__ == "__main__":
    NB_DIR.mkdir(parents=True, exist_ok=True)
    only = sys.argv[1:] or list(MODELS)
    for k in only:
        print(f"=== {k} ===")
        build_and_run(k, MODELS[k])
    import subprocess
    print("=== selection ===")
    subprocess.run([sys.executable, str(BASE_DIR / "training" / "05_select_best_model.py")], check=True)
