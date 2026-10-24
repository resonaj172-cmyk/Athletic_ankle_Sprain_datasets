import os
import sys
from pathlib import Path

# Add project root directory to Python path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# Ensure VERCEL environment variable is set
os.environ["VERCEL"] = "1"

from app.app import create_app

app = create_app()

if __name__ == "__main__":
    app.run()
