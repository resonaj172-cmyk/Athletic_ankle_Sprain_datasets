import sys
import zipfile
import shutil
import pandas as pd
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from training.common import ensure_dirs


def inspect_dataset():
    ensure_dirs()
    print("=" * 60)
    print("STEP 0: INSPECTING DATASET")
    print("=" * 60)

    if not config.RAW_ZIP_PATH.exists():
        msg = f"ERROR: Raw zip file not found at: {config.RAW_ZIP_PATH}"
        print(msg)
        sys.exit(1)

    # Copy zip to data/raw/
    dest_zip = config.RAW_DATA_DIR / config.RAW_ZIP_PATH.name
    print(f"Copying {config.RAW_ZIP_PATH} -> {dest_zip}")
    shutil.copy2(config.RAW_ZIP_PATH, dest_zip)

    with zipfile.ZipFile(dest_zip, "r") as z:
        file_list = z.namelist()
        print(f"Archive contains {len(file_list)} file(s): {file_list}")
        z.extractall(config.RAW_DATA_DIR)
        csv_files = [f for f in file_list if f.lower().endswith(".csv")]

    if not csv_files:
        msg = "ERROR: No CSV files found inside the archive!"
        print(msg)
        sys.exit(1)

    print(f"Found {len(csv_files)} CSV file(s): {csv_files}")
    primary_csv_path = config.RAW_DATA_DIR / csv_files[0]
    print(f"Inspecting primary CSV: {primary_csv_path.name}")

    df = pd.read_csv(primary_csv_path)

    # Normalize column names for check
    col_mapping = {c: c.strip().lower() for c in df.columns}
    normalized_cols = set(col_mapping.values())

    expected_cols = set(config.FEATURE_COLUMNS + [config.TARGET_COLUMN])
    missing_cols = expected_cols - normalized_cols

    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("DATASET INSPECTION REPORT")
    report_lines.append("=" * 60)
    report_lines.append(f"Zip source: {config.RAW_ZIP_PATH}")
    report_lines.append(f"Copied zip: {dest_zip}")
    report_lines.append(f"Files in zip: {file_list}")
    report_lines.append(f"Primary CSV: {csv_files[0]}")
    report_lines.append(f"Total Rows: {len(df)}")
    report_lines.append(f"Total Columns: {len(df.columns)}")
    report_lines.append("\nOriginal Columns & Dtypes:")
    for col, dtype in df.dtypes.items():
        missing_cnt = df[col].isna().sum()
        missing_pct = (missing_cnt / len(df)) * 100
        report_lines.append(f"  - {col}: {dtype} (missing: {missing_cnt}, {missing_pct:.2f}%)")

    report_lines.append("\nRequired 7 Features + Target Check:")
    for req in config.FEATURE_COLUMNS + [config.TARGET_COLUMN]:
        found = any(c.strip().lower() == req for c in df.columns)
        report_lines.append(f"  - {req}: {'FOUND' if found else 'MISSING'}")

    report_lines.append("\nSummary Statistics of Raw Data (Numeric):")
    report_lines.append(df.describe().to_string())

    report_text = "\n".join(report_lines)
    inspection_file = config.REPORTS_DIR / "dataset_inspection.txt"
    inspection_file.write_text(report_text, encoding="utf-8")
    print(f"Inspection report saved to: {inspection_file}")

    print("\n--- Summary Preview ---")
    print(f"Rows: {len(df)}, Cols: {len(df.columns)}")
    print(f"Missing required columns: {missing_cols}")

    if missing_cols:
        print(f"\nCRITICAL ERROR: Missing required columns: {missing_cols}")
        print("STOPPING pipeline as instructed in specification.")
        sys.exit(1)

    print("\nDataset inspection successful. All required columns are present.")
    return df


if __name__ == "__main__":
    inspect_dataset()
