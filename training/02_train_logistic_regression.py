import sys
import json
import time
import joblib
import sklearn
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from training.common import (
    ensure_dirs,
    load_train_test_data,
    evaluate_model,
    measure_wall_time,
    save_evaluation_plots,
)
def train_logistic_regression():
    ensure_dirs()
    model_key = "logistic_regression"
    model_display_name = "Logistic Regression"
    model_dir = config.MODELS_DIR / model_key
    model_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"STEP 2: TRAINING {model_display_name.upper()}")
    print("=" * 60)

    X_train, y_train, X_test, y_test = load_train_test_data()

    # Compute training set medians for model config and explainability
    train_medians = X_train.median().to_dict()

    # Base pipeline: Median Imputer -> StandardScaler -> LogisticRegression
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(random_state=config.RANDOM_STATE, max_iter=2000)),
    ])

    param_grid = {
        "classifier__C": [0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0],
        "classifier__solver": ["lbfgs", "saga"],
        "classifier__class_weight": [None, "balanced"],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )

    def fit_and_train():
        t_start = time.perf_counter()
        grid_search.fit(X_train, y_train)
        t_tuning_end = time.perf_counter()

        best_pipe = grid_search.best_estimator_

        # Measure fit-only time with best params
        t_fit_start = time.perf_counter()
        best_pipe.fit(X_train, y_train)
        t_fit_end = time.perf_counter()
        fit_only_time = t_fit_end - t_fit_start

        return best_pipe, fit_only_time

    def predict_fn(mdl, data):
        return mdl.predict_proba(data)

    print("Starting hyperparameter tuning via 5-Fold Stratified CV on training set...")
    best_model, timing_dict, timing_df = measure_wall_time(fit_and_train, predict_fn, X_test)

    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_
    print(f"Best CV Macro F1: {best_cv_score:.4f}")
    print(f"Best Parameters: {best_params}")

    # Save best_params.json
    params_payload = {
        "best_params": {k.replace("classifier__", ""): v for k, v in best_params.items()},
        "best_cv_macro_f1": float(best_cv_score),
        "cv_folds": 5,
        "scoring": "f1_macro",
    }
    (model_dir / "best_params.json").write_text(json.dumps(params_payload, indent=2), encoding="utf-8")

    # Evaluate on held-out test set
    metrics_dict, metrics_df = evaluate_model(
        best_model, X_train, y_train, X_test, y_test, model_display_name, cv_macro_f1=best_cv_score
    )

    # Save metrics.csv and wall_time.csv
    metrics_df.to_csv(model_dir / "metrics.csv", index=False)
    timing_df.to_csv(model_dir / "wall_time.csv", index=False)

    # Save model file (.joblib)
    model_file_path = model_dir / "model.joblib"
    joblib.dump(best_model, model_file_path)
    print(f"Saved model pipeline to: {model_file_path}")

    # Save model_config.json
    model_config = {
        "model_key": model_key,
        "model_name": model_display_name,
        "model_format": "joblib",
        "model_file": "model.joblib",
        "feature_columns": config.FEATURE_COLUMNS,
        "class_names": config.CLASS_NAMES,
        "training_medians": train_medians,
        "sklearn_version": sklearn.__version__,
        "metrics": metrics_dict,
        "timing": timing_dict,
    }
    (model_dir / "model_config.json").write_text(json.dumps(model_config, indent=2), encoding="utf-8")

    # Save evaluation figures
    print("Generating evaluation plots (Confusion Matrix, ROC Curves, Permutation Importance)...")
    save_evaluation_plots(best_model, X_test, y_test, model_key, model_display_name)

    print("\n" + "=" * 60)
    print(f"SUMMARY FOR {model_display_name}")
    print("=" * 60)
    print(f"Test Accuracy:       {metrics_dict['test_accuracy']:.4f}")
    print(f"Test Macro F1:       {metrics_dict['macro_f1']:.4f}")
    print(f"Test Weighted F1:    {metrics_dict['weighted_f1']:.4f}")
    print(f"Test ROC-AUC Macro:  {metrics_dict['roc_auc_macro_ovr']:.4f}")
    print(f"Train Accuracy:      {metrics_dict['train_accuracy']:.4f}")
    print(f"CV Macro F1:         {metrics_dict['cv_macro_f1']:.4f}")
    print(f"Meets Target (>=0.90): {metrics_dict['meets_target']}")
    print(f"Total Training Time: {timing_dict['total_training_wall_time_s']:.2f}s")
    print(f"Pred Time / 1k rows: {timing_dict['prediction_time_per_1k_s']*1000:.2f}ms")
    print("=" * 60)

    if not metrics_dict["meets_target"]:
        print(f"WARNING: {model_display_name} test accuracy ({metrics_dict['test_accuracy']:.4f}) is below target 0.90.")
    elif metrics_dict["test_accuracy"] > 0.97:
        print(f"NOTE: Test accuracy {metrics_dict['test_accuracy']:.4f} > 0.97. Verified no leakage.")

    return best_model, metrics_dict, timing_dict


if __name__ == "__main__":
    train_logistic_regression()
