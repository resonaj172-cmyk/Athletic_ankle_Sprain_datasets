from datetime import datetime, timezone
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import UniqueConstraint

db = SQLAlchemy()


def utcnow():
    return datetime.now(timezone.utc)


class User(db.Model):
    __tablename__ = "users"

    user_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    full_name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(50), default="athlete", nullable=False)
    created_at = db.Column(db.DateTime, default=utcnow, nullable=False)

    athlete = db.relationship("Athlete", back_populates="user", uselist=False, cascade="all, delete-orphan")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __repr__(self):
        return f"<User {self.email}>"


class Athlete(db.Model):
    __tablename__ = "athletes"

    athlete_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.user_id"), unique=True, nullable=False)
    age = db.Column(db.Integer, nullable=True)
    gender = db.Column(db.String(50), nullable=True)
    sport_type = db.Column(db.String(100), nullable=True)
    height_cm = db.Column(db.Float, nullable=True)
    weight_kg = db.Column(db.Float, nullable=True)

    user = db.relationship("User", back_populates="athlete")
    sessions = db.relationship("SessionRecord", back_populates="athlete", cascade="all, delete-orphan", order_by="SessionRecord.session_number")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def formatted_id(self):
        return f"A{self.athlete_id:03d}"

    def __repr__(self):
        return f"<Athlete {self.formatted_id}>"


class SessionRecord(db.Model):
    __tablename__ = "sessions"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    athlete_id = db.Column(db.Integer, db.ForeignKey("athletes.athlete_id"), nullable=False, index=True)
    session_number = db.Column(db.Integer, nullable=False)

    # Exactly Seven Feature Values
    training_duration = db.Column(db.Float, nullable=False)
    training_intensity = db.Column(db.Float, nullable=True)  # nullable
    training_load = db.Column(db.Float, nullable=False)
    heart_rate = db.Column(db.Float, nullable=True)          # nullable
    sleep_quality = db.Column(db.Float, nullable=True)       # nullable
    recovery_score = db.Column(db.Float, nullable=False)
    stress_level = db.Column(db.Float, nullable=False)

    # Prediction outputs (NULL for session < 15)
    predicted_class = db.Column(db.String(20), nullable=True)
    prob_low = db.Column(db.Float, nullable=True)
    prob_moderate = db.Column(db.Float, nullable=True)
    prob_high = db.Column(db.Float, nullable=True)
    model_name = db.Column(db.String(100), nullable=True)

    # Meta
    source = db.Column(db.String(50), default="user", nullable=False)  # "user" or "demo_seed"
    created_at = db.Column(db.DateTime, default=utcnow, nullable=False)

    athlete = db.relationship("Athlete", back_populates="sessions")

    __table_args__ = (
        UniqueConstraint("athlete_id", "session_number", name="uq_athlete_session_num"),
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def to_dict(self):
        return {
            "id": self.id,
            "session_number": self.session_number,
            "training_duration": self.training_duration,
            "training_intensity": self.training_intensity,
            "training_load": self.training_load,
            "heart_rate": self.heart_rate,
            "sleep_quality": self.sleep_quality,
            "recovery_score": self.recovery_score,
            "stress_level": self.stress_level,
            "predicted_class": self.predicted_class,
            "prob_low": self.prob_low,
            "prob_moderate": self.prob_moderate,
            "prob_high": self.prob_high,
            "model_name": self.model_name,
            "source": self.source,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def __repr__(self):
        return f"<SessionRecord Athlete={self.athlete_id} Num={self.session_number} Class={self.predicted_class}>"
