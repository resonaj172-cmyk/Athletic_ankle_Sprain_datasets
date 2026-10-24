import json
import time
import shutil
import zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import label_binarize

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


def ensure_dirs():
    """Ensure all required directories exist."""
    config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def get_raw_zip_and_extract():
    """
    Copy RAW_ZIP_PATH to data/raw/ and extract there without modifying original.
    Returns the path to extracted CSV file.
    """
    ensure_dirs()
    if not config.RAW_ZIP_PATH.exists():
        raise FileNotFoundError(f"Source zip file not found at: {config.RAW_ZIP_PATH}")

    dest_zip = config.RAW_DATA_DIR / config.RAW_ZIP_PATH.name
    if not dest_zip.exists() or dest_zip.stat().st_size != config.RAW_ZIP_PATH.stat().st_size:
        shutil.copy2(config.RAW_ZIP_PATH, dest_zip)

    with zipfile.ZipFile(dest_zip, "r") as z:
        z.extractall(config.RAW_DATA_DIR)
        csv_files = [f for f in z.namelist() if f.lower().endswith(".csv")]

    if not csv_files:
        raise FileNotFoundError("No CSV files found in the extracted zip archive.")

    return config.RAW_DATA_DIR / csv_files[0], csv_files


def load_train_test_data():
    """Load train.csv and test.csv, validate no target leakage, return X_train, y_train, X_test, y_test."""
    train_path = config.PROCESSED_DATA_DIR / "train.csv"
    test_path = config.PROCESSED_DATA_DIR / "test.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("train.csv or test.csv missing. Run 01_prepare_data.py first.")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    assert config.TARGET_COLUMN not in config.FEATURE_COLUMNS, "Leakage: target column in feature list!"
    assert config.CLASS_COLUMN not in config.FEATURE_COLUMNS, "Leakage: class column in feature list!"

    for col in config.FEATURE_COLUMNS:
        assert col in train_df.columns, f"Missing feature {col} in train.csv"
        assert col in test_df.columns, f"Missing feature {col} in test.csv"

    assert config.CLASS_COLUMN in train_df.columns, f"Missing target {config.CLASS_COLUMN} in train.csv"
    assert config.CLASS_COLUMN in test_df.columns, f"Missing target {config.CLASS_COLUMN} in test.csv"

    X_train = train_df[config.FEATURE_COLUMNS].copy()
    y_train = train_df[config.CLASS_COLUMN].copy()
    X_test = test_df[config.FEATURE_COLUMNS].copy()
    y_test = test_df[config.CLASS_COLUMN].copy()

    # Leakage check: verify X_train and X_test columns are EXACTLY the 7 features in exact order
    assert list(X_train.columns) == config.FEATURE_COLUMNS, "X_train feature list/order mismatch!"
    assert list(X_test.columns) == config.FEATURE_COLUMNS, "X_test feature list/order mismatch!"

    return X_train, y_train, X_test, y_test


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name, cv_macro_f1=None):
    """
    Computes comprehensive test and train metrics.
    Returns metrics dict and metrics DataFrame.
    """
    # Predictions on test set
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # Train accuracy for overfitting check
    y_train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)

    test_acc = accuracy_score(y_test, y_pred)

    # Macro & Weighted metrics
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    # Per-class metrics
    p_per_class, r_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_test, y_pred, average=None, labels=[0, 1, 2], zero_division=0
    )

    # Multi-class ROC-AUC (One-vs-Rest, macro)
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    try:
        if y_test_bin.shape[1] == 1:
            # edge case
            roc_auc_macro = roc_auc_score(y_test, y_pred_proba[:, 1])
        else:
            roc_auc_macro = roc_auc_score(y_test_bin, y_pred_proba, multi_class="ovr", average="macro")
    except Exception as e:
        roc_auc_macro = float("nan")

    meets_target = bool(test_acc >= 0.90)

    metrics_dict = {
        "model_name": model_name,
        "test_accuracy": float(test_acc),
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_weighted),
        "roc_auc_macro_ovr": float(roc_auc_macro),
        "train_accuracy": float(train_acc),
        "cv_macro_f1": float(cv_macro_f1) if cv_macro_f1 is not None else float("nan"),
        "meets_target": meets_target,
        # Per-class metrics
        "precision_low": float(p_per_class[0]),
        "recall_low": float(r_per_class[0]),
        "f1_low": float(f1_per_class[0]),
        "precision_moderate": float(p_per_class[1]),
        "recall_moderate": float(r_per_class[1]),
        "f1_moderate": float(f1_per_class[1]),
        "precision_high": float(p_per_class[2]),
        "recall_high": float(r_per_class[2]),
        "f1_high": float(f1_per_class[2]),
    }

    metrics_df = pd.DataFrame([metrics_dict])
    return metrics_dict, metrics_df


def measure_wall_time(train_fn, predict_fn, X_test, n_warmup=2, n_runs=10):
    """
    Measures training wall time and prediction time per 1,000 rows using time.perf_counter().
    """
    t0 = time.perf_counter()
    model, fit_only_time = train_fn()
    t1 = time.perf_counter()
    total_training_wall_time = t1 - t0

    # Warm-up predict
    for _ in range(n_warmup):
        _ = predict_fn(model, X_test)

    # Benchmark predict
    predict_times = []
    for _ in range(n_runs):
        tp0 = time.perf_counter()
        _ = predict_fn(model, X_test)
        tp1 = time.perf_counter()
        predict_times.append(tp1 - tp0)

    avg_pred_time_batch = np.mean(predict_times)
    n_rows = len(X_test)
    pred_time_per_1k = (avg_pred_time_batch / n_rows) * 1000.0

    timing_dict = {
        "total_training_wall_time_s": float(total_training_wall_time),
        "fit_only_time_s": float(fit_only_time) if fit_only_time is not None else float(total_training_wall_time),
        "prediction_time_per_1k_s": float(pred_time_per_1k),
    }
    timing_df = pd.DataFrame([timing_dict])
    return model, timing_dict, timing_df


def save_evaluation_plots(model, X_test, y_test, model_key, model_display_name):
    """
    Generates and saves:
    1. Confusion matrix: reports/figures/<model_key>_confusion_matrix.png
    2. ROC curves per class: reports/figures/<model_key>_roc_curves.png
    3. Permutation importance bar chart: reports/figures/<model_key>_permutation_importance.png
    Also saves permutation importance values to CSV: models/<model_key>/permutation_importance.csv
    """
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    model_dir = config.MODELS_DIR / model_key
    model_dir.mkdir(parents=True, exist_ok=True)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=config.CLASS_NAMES)
    disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)
    ax.set_title(f"Confusion Matrix - {model_display_name}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    cm_path = config.FIGURES_DIR / f"{model_key}_confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()

    # 2. ROC Curves per class
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    for i, class_name in enumerate(config.CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[i], lw=2, label=f"{class_name} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title(f"ROC Curves (One-vs-Rest) - {model_display_name}", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    roc_path = config.FIGURES_DIR / f"{model_key}_roc_curves.png"
    plt.savefig(roc_path, dpi=200)
    plt.close()

    # 3. Permutation Importance
    perm = permutation_importance(
        model, X_test, y_test, scoring="f1_macro", n_repeats=10, random_state=config.RANDOM_STATE
    )
    sorted_idx = perm.importances_mean.argsort()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        [config.FEATURE_COLUMNS[i] for i in sorted_idx],
        perm.importances_mean[sorted_idx],
        xerr=perm.importances_std[sorted_idx],
        align="center",
        color="#3498db",
        edgecolor="#2980b9",
        alpha=0.85,
    )
    ax.set_xlabel("Mean Macro F1 Decrease on Permutation", fontsize=10)
    ax.set_title(f"Permutation Feature Importance - {model_display_name}", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    perm_path = config.FIGURES_DIR / f"{model_key}_permutation_importance.png"
    plt.savefig(perm_path, dpi=200)
    plt.close()

    # Save permutation importance to CSV
    perm_df = pd.DataFrame({
        "feature": config.FEATURE_COLUMNS,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values(by="importance_mean", ascending=False)
    perm_df.to_csv(model_dir / "permutation_importance.csv", index=False)
