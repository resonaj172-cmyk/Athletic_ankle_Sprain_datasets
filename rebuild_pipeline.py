from _pytest import hookspec
import sys
import subprocess
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
print("Running Step 1: Prepare Data...")
subprocess.run([sys.executable, str(BASE_DIR / "training" / "01_prepare_data.py")], check=True)
print("\nRunning Model Notebook")
("oks Generator & Runner...")
subprocess.run([sys.executable, str(BASE_DIR / "generate_model_notebooks.py")], check=True)
print("\nALL MODELS RETRAINED AND NOTEBOOKS UPDATED SUCCESSFULLY!")
   