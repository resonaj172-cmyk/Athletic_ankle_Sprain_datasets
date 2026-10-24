import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from app.app import create_app
import seed_demo

app = create_app()
print("create_app OK")
seed_demo.seed_demo_data(app)
print("seed OK")
c = app.test_client()
r = c.get("/login"); print("GET /login", r.status_code)
html = r.get_data(as_text=True)
import re
m = re.search(r'name="csrf_token" value="([^"]*)"', html)
tok = m.group(1) if m else ""
r = c.post("/login", data={"email": "demo@athlete.app", "password": "Demo@1234", "csrf_token": tok}, follow_redirects=False)
print("POST /login", r.status_code, r.headers.get("Location"))
for url in ["/dashboard", "/history", "/add-session", "/athlete-details", "/assistant"]:
    try:
        r = c.get(url)
        print("GET", url, r.status_code)
        if r.status_code >= 400:
            print(r.get_data(as_text=True)[:400])
    except Exception as e:
        print("GET", url, "EXCEPTION", type(e).__name__, e)
r = c.get("/dashboard"); body = r.get_data(as_text=True)
print("chart_labels in page:", re.findall(r"const labels = (.*);", body))
print("chart_loads in page:", re.findall(r"const dataLoads = (.*);", body))
