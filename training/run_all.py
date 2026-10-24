import sys
import subprocess
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def run_pipeline():
    print("=" * 70)
    print("ATHLETE FATIGUE PREDICTION - FULL OFFLINE TRAINING PIPELINE")
    print("=" * 70)

    stages = [
        ("00_inspect_dataset.py", "Stage 0: Inspect Dataset"),
        ("01_prepare_data.py", "Stage 1: Clean, Create Classes & Split Data"),
        ("02_train_logistic_regression.py", "Stage 2: Train Logistic Regression"),
        ("03_train_random_forest.py", "Stage 3: Train Random Forest"),
        ("04_train_xgboost.py", "Stage 4: Train XGBoost"),
        ("05_select_best_model.py", "Stage 5: Compare and Select Best Model"),
    ]

    t0_total = time.perf_counter()

    for script_name, stage_desc in stages:
        script_path = BASE_DIR / "training" / script_name
        print(f"\n>>> Running {stage_desc} ({script_name})...")
        t0 = time.perf_counter()
        res = subprocess.run([sys.executable, str(script_path)], cwd=str(BASE_DIR))
        t1 = time.perf_counter()

        if res.returncode != 0:
            print(f"\n[ERROR] Pipeline failed at {stage_desc} (exit code {res.returncode}). Stopping.")
            sys.exit(res.returncode)
        print(f">>> {stage_desc} completed in {t1 - t0:.2f}s")

    t1_total = time.perf_counter()
    print("\n" + "=" * 70)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {t1_total - t0_total:.2f}s")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()
