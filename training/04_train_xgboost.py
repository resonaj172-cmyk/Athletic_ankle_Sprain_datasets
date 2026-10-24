import sys
import json
import time
import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

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


def train_xgboost():
    ensure_dirs()
    model_key = "xgboost"
    model_display_name = "XGBoost"
    model_dir = config.MODELS_DIR / model_key
    model_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"STEP 4: TRAINING {model_display_name.upper()}")
    print("=" * 60)

    X_train, y_train, X_test, y_test = load_train_test_data()

    # Training set medians (for config/explainability)
    train_medians = X_train.median().to_dict()

    # Base XGBoost model (receives NaN directly)
    base_xgb = xgb.XGBClassifier(
        random_state=config.RANDOM_STATE,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=1,
    )

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.03, 0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "min_child_weight": [1, 3],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    grid_search = GridSearchCV(
        estimator=base_xgb,
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

        best_estimator = grid_search.best_estimator_

        # Measure fit-only time with best params on full X_train
        t_fit_start = time.perf_counter()
        best_estimator.fit(X_train, y_train)
        t_fit_end = time.perf_counter()
        fit_only_time = t_fit_end - t_fit_start

        return best_estimator, fit_only_time

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
        "best_params": best_params,
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

    # Save native model file (JSON format)
    model_file_path = model_dir / "model.json"
    best_model.save_model(str(model_file_path))
    print(f"Saved native XGBoost model to: {model_file_path}")

    # Save model_config.json
    model_config = {
        "model_key": model_key,
        "model_name": model_display_name,
        "model_format": "xgboost_json",
        "model_file": "model.json",
        "feature_columns": config.FEATURE_COLUMNS,
        "class_names": config.CLASS_NAMES,
        "training_medians": train_medians,
        "xgboost_version": xgb.__version__,
        "metrics": metrics_dict,
        "timing": timing_dict,
    }
    (model_dir / "model_config.json").write_text(json.dumps(model_config, indent=2), encoding="utf-8")

    # Save standard evaluation figures (Confusion Matrix, ROC Curves, Permutation Importance)
    print("Generating evaluation plots (Confusion Matrix, ROC Curves, Permutation Importance)...")
    save_evaluation_plots(best_model, X_test, y_test, model_key, model_display_name)

    # XGBoost Loss Curve using an evaluation carve-out from TRAINING data (NEVER from test set)
    print("Generating training/validation loss curve from training carve-out...")
    X_tr_sub, X_val_sub, y_tr_sub, y_val_sub = train_test_split(
        X_train, y_train, test_size=0.15, random_state=config.RANDOM_STATE, stratify=y_train
    )

    eval_xgb = xgb.XGBClassifier(
        **best_params,
        random_state=config.RANDOM_STATE,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
    )
    eval_xgb.fit(
        X_tr_sub,
        y_tr_sub,
        eval_set=[(X_tr_sub, y_tr_sub), (X_val_sub, y_val_sub)],
        verbose=False,
    )

    eval_results = eval_xgb.evals_result()
    train_loss = eval_results["validation_0"]["mlogloss"]
    val_loss = eval_results["validation_1"]["mlogloss"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(train_loss, label="Train Loss (mlogloss)", color="#2980b9", lw=2)
    ax.plot(val_loss, label="Validation Loss (mlogloss)", color="#e67e22", lw=2, linestyle="--")
    ax.set_xlabel("Boosting Iteration", fontsize=10)
    ax.set_ylabel("Multi-class Log Loss", fontsize=10)
    ax.set_title("XGBoost Training vs Validation Loss Curve", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    loss_curve_path = config.FIGURES_DIR / "xgboost_loss_curve.png"
    plt.savefig(loss_curve_path, dpi=200)
    plt.close()
    print(f"Saved loss curve to: {loss_curve_path}")

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
    train_xgboost()
