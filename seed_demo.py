import sys
from pathlib import Path
import bcrypt
from flask import Flask

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent))
import config
from app.models_db import db, User, Athlete, SessionRecord

DEMO_SESSIONS = [
    {"session_number": 1, "training_duration": 75.0, "training_intensity": 6.0, "training_load": 520.0, "heart_rate": 70.0, "sleep_quality": 6.5, "recovery_score": 62.0, "stress_level": 0.35},
    {"session_number": 2, "training_duration": 90.0, "training_intensity": 7.0, "training_load": 690.0, "heart_rate": 74.0, "sleep_quality": 5.8, "recovery_score": 55.0, "stress_level": 0.42},
    {"session_number": 3, "training_duration": 60.0, "training_intensity": 5.0, "training_load": 410.0, "heart_rate": None, "sleep_quality": 7.2, "recovery_score": 68.0, "stress_level": 0.30},
    {"session_number": 4, "training_duration": 110.0, "training_intensity": 8.0, "training_load": 980.0, "heart_rate": 78.0, "sleep_quality": 4.9, "recovery_score": 41.0, "stress_level": 0.58},
    {"session_number": 5, "training_duration": 85.0, "training_intensity": 6.5, "training_load": 640.0, "heart_rate": 72.0, "sleep_quality": 6.0, "recovery_score": 58.0, "stress_level": 0.40},
    {"session_number": 6, "training_duration": 120.0, "training_intensity": 8.5, "training_load": 1150.0, "heart_rate": 80.0, "sleep_quality": 4.5, "recovery_score": 36.0, "stress_level": 0.66},
    {"session_number": 7, "training_duration": 45.0, "training_intensity": 4.0, "training_load": 260.0, "heart_rate": 66.0, "sleep_quality": 7.8, "recovery_score": 74.0, "stress_level": 0.25},
    {"session_number": 8, "training_duration": 95.0, "training_intensity": 7.0, "training_load": 760.0, "heart_rate": 73.0, "sleep_quality": 5.6, "recovery_score": 52.0, "stress_level": 0.45},
    {"session_number": 9, "training_duration": 100.0, "training_intensity": 7.5, "training_load": 850.0, "heart_rate": 76.0, "sleep_quality": 5.2, "recovery_score": 47.0, "stress_level": 0.52},
    {"session_number": 10, "training_duration": 70.0, "training_intensity": 5.5, "training_load": 480.0, "heart_rate": 69.0, "sleep_quality": 6.8, "recovery_score": 64.0, "stress_level": 0.33},
    {"session_number": 11, "training_duration": 130.0, "training_intensity": 8.5, "training_load": 1240.0, "heart_rate": 82.0, "sleep_quality": 4.2, "recovery_score": 33.0, "stress_level": 0.70},
    {"session_number": 12, "training_duration": 80.0, "training_intensity": 6.0, "training_load": 560.0, "heart_rate": 71.0, "sleep_quality": 6.2, "recovery_score": 60.0, "stress_level": 0.38},
    {"session_number": 13, "training_duration": 105.0, "training_intensity": 7.5, "training_load": 900.0, "heart_rate": 77.0, "sleep_quality": 5.0, "recovery_score": 45.0, "stress_level": 0.55},
    {"session_number": 14, "training_duration": 90.0, "training_intensity": 7.0, "training_load": 720.0, "heart_rate": 74.0, "sleep_quality": 5.7, "recovery_score": 54.0, "stress_level": 0.44},
]


def seed_demo_data(app=None):
    if app is None:
        app = Flask(__name__)
        app.config["SQLALCHEMY_DATABASE_URI"] = config.DATABASE_URL
        app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
        db.init_app(app)

    with app.app_context():
        db.create_all()

        demo_email = "athlete@gmail.com"
        existing_user = User.query.filter_by(email=demo_email).first()

        if existing_user:
            print(f"Athlete user '{demo_email}' already exists. Resetting sessions...")
            if existing_user.athlete:
                SessionRecord.query.filter_by(athlete_id=existing_user.athlete.athlete_id).delete()
            db.session.delete(existing_user)
            db.session.commit()

        # Hash password in @gmail.com format: athlete@gmail.com
        salt = bcrypt.gensalt()
        pw_hash = bcrypt.hashpw(b"athlete@gmail.com", salt).decode("utf-8")

        user = User(
            full_name="Alex Morgan",
            email=demo_email,
            password_hash=pw_hash,
            role="athlete",
        )
        db.session.add(user)
        db.session.flush()

        athlete = Athlete(
            user_id=user.user_id,
            age=21,
            gender="Female",
            sport_type="Soccer",
            height_cm=168.0,
            weight_kg=60.0,
        )
        db.session.add(athlete)
        db.session.flush()

        for s in DEMO_SESSIONS:
            record = SessionRecord(
                athlete_id=athlete.athlete_id,
                session_number=s["session_number"],
                training_duration=s["training_duration"],
                training_intensity=s["training_intensity"],
                training_load=s["training_load"],
                heart_rate=s["heart_rate"],
                sleep_quality=s["sleep_quality"],
                recovery_score=s["recovery_score"],
                stress_level=s["stress_level"],
                predicted_class=None,
                prob_low=None,
                prob_moderate=None,
                prob_high=None,
                model_name=None,
                source="demo_seed",
            )
            db.session.add(record)

        db.session.commit()
        print(f"Successfully created demo user '{demo_email}' (Athlete ID: {athlete.formatted_id}) with {len(DEMO_SESSIONS)} initial sessions.")
        print("Ready for session 15 prediction.")


if __name__ == "__main__":
    seed_demo_data()
