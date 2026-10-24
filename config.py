import os
from pathlib import Path
from dotenv import load_dotenv

# Base project directory
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from .env if present
load_dotenv(BASE_DIR / ".env")

# Dataset Paths
RAW_ZIP_PATH_STR = os.getenv("RAW_ZIP_PATH", r"C:\Users\Resona\Downloads\fatigue datasets\archive (1).zip")
RAW_ZIP_PATH = Path(RAW_ZIP_PATH_STR)

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Models and Reports
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Application Settings
FIRST_PREDICTION_SESSION = 15
HISTORY_WINDOWS = (7, 14)

# Exact 7 Model Features and Target
FEATURE_COLUMNS = [
    "training_duration",
    "training_intensity",
    "training_load",
    "heart_rate",
    "sleep_quality",
    "recovery_score",
    "stress_level",
]

TARGET_COLUMN = "fatigue_index"
CLASS_COLUMN = "fatigue_class"

# Fatigue Class Boundaries
# 0 = Low (< 47.26), 1 = Moderate (47.26 <= x < 68.89), 2 = High (x >= 68.89)
CLASS_THRESHOLDS = {
    "low_max": 47.26,
    "moderate_max": 68.89,
}
CLASS_NAMES = ["Low", "Moderate", "High"]

# Database & Security
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-athlete-fatigue-2026")

if os.getenv("VERCEL"):
    import shutil
    tmp_instance = Path("/tmp") / "instance"
    try:
        tmp_instance.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    tmp_db = tmp_instance / "fatigue_app.db"
    src_db = BASE_DIR / "instance" / "fatigue_app.db"
    if not tmp_db.exists() and src_db.exists():
        try:
            shutil.copyfile(src_db, tmp_db)
        except Exception:
            pass
    DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{tmp_db}")
else:
    DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'instance' / 'fatigue_app.db'}")

# LLM Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "")

# Random State
RANDOM_STATE = 42

# Ensure key directories exist
for d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR, BASE_DIR / "instance"]:
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
