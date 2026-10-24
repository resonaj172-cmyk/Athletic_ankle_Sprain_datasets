import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from app.app import create_app

app = create_app()

if __name__ == "__main__":
    print("Starting Athlete Fatigue Monitor web application on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
