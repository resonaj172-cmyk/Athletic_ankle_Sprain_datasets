import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from training.common import ensure_dirs, get_raw_zip_and_extract

GROUP_COL = "athlete_id"
ID_COLS = ["athlete_id", "session_id"]  # kept ONLY as metadata for grouping; never model inputs
TEST_FRACTION_FOLDS = 3                 # 3 of 10 grouped folds = ~30% test
N_FOLDS = 10


def prepare_data():
    ensure_dirs()
    print("=" * 60)
    print("STEP 1: PREPARING DATA (70/30 athlete-grouped, class-stratified split)")
    print("=" * 60)

    csv_path, _ = get_raw_zip_and_extract()
    print(f"Reading raw data from: {csv_path}")
    raw_df = pd.read_csv(csv_path)
    raw_df.rename(columns={c: c.strip().lower() for c in raw_df.columns}, inplace=True)
    print(f"Raw dataset: {raw_df.shape[0]} rows x {raw_df.shape[1]} columns")

    for col in config.FEATURE_COLUMNS + [config.TARGET_COLUMN] + ID_COLS:
        if col not in raw_df.columns:
            raise KeyError(f"Required column '{col}' missing from raw dataset.")

    # Keep 7 features + target + id columns (ids are grouping metadata only)
    cols_to_keep = ID_COLS + config.FEATURE_COLUMNS + [config.TARGET_COLUMN]
    df = raw_df[cols_to_keep].copy()
    rows_before = len(df)

    # Remove EXACT duplicate rows BEFORE splitting
    dup_full = int(df.duplicated().sum())
    dup_feat = int(df.duplicated(subset=config.FEATURE_COLUMNS).sum())
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Exact duplicate rows removed: {dup_full} (rows {rows_before} -> {len(df)})")
    print(f"(Info) rows sharing identical 7-feature values: {dup_feat}")

    # Classes from fatigue_index with the original boundaries
    fi = df[config.TARGET_COLUMN]
    low, mod = config.CLASS_THRESHOLDS["low_max"], config.CLASS_THRESHOLDS["moderate_max"]
    df[config.CLASS_COLUMN] = np.select([fi < low, (fi >= low) & (fi < mod), fi >= mod], [0, 1, 2], default=-1).astype(int)
    assert (df[config.CLASS_COLUMN] >= 0).all()

    class_counts = df[config.CLASS_COLUMN].value_counts().sort_index().to_dict()
    class_dist = {
        config.CLASS_NAMES[k]: {"count": int(v), "percentage": float(v / len(df) * 100)}
        for k, v in class_counts.items()
    }
    p_low = float((fi < low).mean() * 100)
    p_mod = float((fi < mod).mean() * 100)

    class_config = {
        "class_names": config.CLASS_NAMES,
        "class_mapping": {0: "Low", 1: "Moderate", 2: "High"},
        "thresholds": config.CLASS_THRESHOLDS,
        "percentile_low_boundary": p_low,
        "percentile_moderate_boundary": p_mod,
        "class_distribution": class_dist,
        "feature_columns": config.FEATURE_COLUMNS,
        "target_column": config.TARGET_COLUMN,
        "class_column": config.CLASS_COLUMN,
    }
    (config.MODELS_DIR / "class_config.json").write_text(json.dumps(class_config, indent=2), encoding="utf-8")

    full_path = config.PROCESSED_DATA_DIR / "fatigue_7features.csv"
    df.to_csv(full_path, index=False)
    print(f"Saved processed dataset: {full_path}")

    # 70/30 split grouped by athlete, stratified on class (no athlete in both sets)
    groups = df[GROUP_COL].values
    y = df[config.CLASS_COLUMN].values
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    folds = [test for _, test in sgkf.split(df, y, groups)]
    test_idx = np.sort(np.concatenate(folds[:TEST_FRACTION_FOLDS]))
    train_idx = np.setdiff1d(np.arange(len(df)), test_idx)

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    overlap_groups = set(train_df[GROUP_COL]) & set(test_df[GROUP_COL])
    assert len(overlap_groups) == 0, f"Athlete leakage: {len(overlap_groups)} athletes in both sets"
    overlap_rows = pd.merge(train_df[config.FEATURE_COLUMNS], test_df[config.FEATURE_COLUMNS], how="inner")
    print(f"Athletes in both train and test: {len(overlap_groups)}")
    print(f"Identical 7-feature rows shared by train and test: {len(overlap_rows)}")

    n = len(df)
    print(f"\nTotal rows: {n}")
    print(f"Train rows: {len(train_df)} ({len(train_df)/n*100:.2f}%)  | athletes: {train_df[GROUP_COL].nunique()}")
    print(f"Test rows:  {len(test_df)} ({len(test_df)/n*100:.2f}%)  | athletes: {test_df[GROUP_COL].nunique()}")
    for name, part in (("Train", train_df), ("Test", test_df)):
        vc = part[config.CLASS_COLUMN].value_counts().sort_index()
        print(f"{name} class counts/percentages:")
        for k, v in vc.items():
            print(f"  - {config.CLASS_NAMES[k]}: {v} ({v/len(part)*100:.2f}%)")

    train_df.to_csv(config.PROCESSED_DATA_DIR / "train.csv", index=False)
    test_df.to_csv(config.PROCESSED_DATA_DIR / "test.csv", index=False)

    split_info = {
        "split": "70/30",
        "method": "StratifiedGroupKFold(n_splits=10), first 3 folds = test",
        "group_column": GROUP_COL,
        "random_state": config.RANDOM_STATE,
        "rows_before_dedup": int(rows_before),
        "exact_duplicates_removed": dup_full,
        "total_rows": int(n),
        "train_indices": [int(i) for i in train_idx],
        "test_indices": [int(i) for i in test_idx],
    }
    (config.PROCESSED_DATA_DIR / "split_indices.json").write_text(json.dumps(split_info), encoding="utf-8")
    print("Saved split_indices.json")

    for name, part in (("Train", train_df), ("Test", test_df)):
        miss = {c: int(part[c].isna().sum()) for c in config.FEATURE_COLUMNS if part[c].isna().sum() > 0}
        print(f"Missing values in {name}: {miss}")
    print("\nData preparation complete.")
    return train_df, test_df


if __name__ == "__main__":
    prepare_data()
