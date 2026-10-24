import os
import secrets
import json
from pathlib import Path
from functools import wraps
import bcrypt
import pandas as pd
import numpy as np

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash,
    jsonify,
    abort,
    send_from_directory,
)

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
import sys
sys.path.append(str(BASE_DIR))
import config
from app.models_db import db, User, Athlete, SessionRecord
from app.ml.loader import ModelLoader, ModelNotFoundError
from app.ml.features import FIELD_SPECS, validate_session_inputs, compute_history_summary
from app.ml.predict import predict_session, explain_prediction, compute_what_if
from app.ml.labels import get_all_state_labels, LABELS_CONFIG
from app.assistant.context import build_assistant_context
from app.assistant.llm import ask_assistant
from seed_demo import DEMO_SESSIONS


def create_app():
    app = Flask(
        __name__,
        template_folder=str(BASE_DIR / "app" / "templates"),
        static_folder=str(BASE_DIR / "app" / "static"),
    )

    app.config["SECRET_KEY"] = config.SECRET_KEY
    app.config["SQLALCHEMY_DATABASE_URI"] = config.DATABASE_URL
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

    db.init_app(app)

    with app.app_context():
        db.create_all()

    # Rate limiting simple counter in memory for failed logins
    failed_attempts = {}

    # CSRF protection helper
    @app.before_request
    def check_csrf():
        if "csrf_token" not in session:
            session["csrf_token"] = secrets.token_hex(16)

        if request.method == "POST":
            # Exclude nothing except explicit internal webhooks if any
            token = request.form.get("csrf_token") or request.headers.get("X-CSRF-Token")
            if not token or token != session.get("csrf_token"):
                # For json requests
                if request.is_json:
                    json_token = request.get_json(silent=True) or {}
                    token = json_token.get("csrf_token")
                if not token or token != session.get("csrf_token"):
                    if request.path.startswith("/api/"):
                        return jsonify({"error": "CSRF token invalid or missing"}), 403
                    flash("Session expired or invalid form submission. Please try again.", "danger")
                    return redirect(request.referrer or url_for("login"))

    # Context processor for templates
    @app.context_processor
    def inject_globals():
        current_user = None
        current_athlete = None
        if "user_id" in session:
            current_user = db.session.get(User, session["user_id"])
            if current_user:
                current_athlete = current_user.athlete

        selected_meta = ModelLoader.get_selected_metadata()
        return {
            "current_user": current_user,
            "current_athlete": current_athlete,
            "csrf_token": session.get("csrf_token", ""),
            "selected_meta": selected_meta,
            "field_specs": FIELD_SPECS,
            "labels_config": LABELS_CONFIG,
        }

    # Auth decorator
    def login_required(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if "user_id" not in session:
                flash("Please log in to access this page.", "info")
                return redirect(url_for("login", next=request.url))
            return f(*args, **kwargs)
        return decorated_function

    def ensure_baseline_history(athlete_id: int):
        """
        Ensures that an athlete has the 14 historical baseline sessions (sessions #1 through #14).
        If baseline sessions are missing or if user created sessions without baseline:
        1. Populate sessions #1 through #14 from DEMO_SESSIONS.
        2. Renumber any non-demo user sessions to start at #15, #16... and evaluate ML prediction for them.
        """
        if not athlete_id:
            return

        baseline_count = SessionRecord.query.filter_by(athlete_id=athlete_id, source="demo_seed").count()
        if baseline_count < 14:
            user_sessions = SessionRecord.query.filter_by(athlete_id=athlete_id).filter(SessionRecord.source != "demo_seed").order_by(SessionRecord.session_number.asc()).all()

            for s in DEMO_SESSIONS:
                existing = SessionRecord.query.filter_by(athlete_id=athlete_id, session_number=s["session_number"]).first()
                if not existing:
                    rec = SessionRecord(
                        athlete_id=athlete_id,
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
                    db.session.add(rec)

            next_num = 15
            for u_rec in user_sessions:
                u_rec.session_number = next_num
                next_num += 1
                if not u_rec.predicted_class:
                    parsed_vals = {
                        "training_duration": u_rec.training_duration,
                        "training_intensity": u_rec.training_intensity if u_rec.training_intensity is not None else np.nan,
                        "training_load": u_rec.training_load,
                        "heart_rate": u_rec.heart_rate if u_rec.heart_rate is not None else np.nan,
                        "sleep_quality": u_rec.sleep_quality if u_rec.sleep_quality is not None else np.nan,
                        "recovery_score": u_rec.recovery_score,
                        "stress_level": u_rec.stress_level,
                    }
                    try:
                        pred_res = predict_session(parsed_vals)
                        u_rec.predicted_class = pred_res["predicted_class"]
                        u_rec.prob_low = pred_res["probabilities_raw"][0]
                        u_rec.prob_moderate = pred_res["probabilities_raw"][1]
                        u_rec.prob_high = pred_res["probabilities_raw"][2]
                        u_rec.model_name = pred_res["model_name"]
                    except Exception:
                        pass

            db.session.commit()

    # -------------------------------------------------------------
    # PUBLIC & AUTH ROUTES
    # -------------------------------------------------------------
    @app.route("/")
    def index():
        if "user_id" in session:
            return redirect(url_for("dashboard"))
        return render_template("landing.html")

    @app.route("/register", methods=["GET", "POST"])
    def register():
        if "user_id" in session:
            return redirect(url_for("dashboard"))

        if request.method == "POST":
            full_name = request.form.get("full_name", "").strip()
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "")
            confirm_password = request.form.get("confirm_password", "")

            if not full_name or not email or not password:
                flash("All fields are required.", "danger")
                return render_template("register.html", full_name=full_name, email=email)

            if len(password) < 8:
                flash("Password must be at least 8 characters long.", "danger")
                return render_template("register.html", full_name=full_name, email=email)

            if password != confirm_password:
                flash("Passwords do not match.", "danger")
                return render_template("register.html", full_name=full_name, email=email)

            existing_user = User.query.filter_by(email=email).first()
            if existing_user:
                flash("An account with that email already exists.", "warning")
                return render_template("register.html", full_name=full_name)

            salt = bcrypt.gensalt()
            pw_hash = bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")

            user = User(full_name=full_name, email=email, password_hash=pw_hash)
            db.session.add(user)
            db.session.flush()

            # Create athlete record & ensure baseline history
            athlete = Athlete(user_id=user.user_id)
            db.session.add(athlete)
            db.session.commit()
            ensure_baseline_history(athlete.athlete_id)

            session["user_id"] = user.user_id
            session["athlete_id"] = athlete.athlete_id
            flash("Welcome! Please complete your athlete profile details.", "success")
            return redirect(url_for("athlete_details"))

        return render_template("register.html")

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if "user_id" in session:
            return redirect(url_for("dashboard"))

        if request.method == "POST":
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "")
            ip = request.remote_addr or "unknown"

            # Simple rate limiting: 5 failed attempts per IP
            attempts = failed_attempts.get(ip, 0)
            if attempts >= 5:
                flash("Too many failed login attempts. Please wait a minute and try again.", "danger")
                return render_template("login.html", email=email)

            user = User.query.filter_by(email=email).first()
            if user and bcrypt.checkpw(password.encode("utf-8"), user.password_hash.encode("utf-8")):
                failed_attempts.pop(ip, None)
                session["user_id"] = user.user_id
                session["athlete_id"] = user.athlete.athlete_id if user.athlete else None
                flash("Logged in successfully.", "success")
                next_page = request.args.get("next")
                return redirect(next_page or url_for("dashboard"))
            else:
                failed_attempts[ip] = attempts + 1
                flash("Invalid email or password.", "danger")
                return render_template("login.html", email=email)

        return render_template("login.html")

    @app.route("/logout")
    def logout():
        session.clear()
        flash("You have been logged out.", "info")
        return redirect(url_for("index"))

    # -------------------------------------------------------------
    # ATHLETE DASHBOARD & DETAILS
    # -------------------------------------------------------------
    @app.route("/dashboard")
    @login_required
    def dashboard():
        user = db.session.get(User, session["user_id"])
        athlete = user.athlete if user else None
        if athlete:
            ensure_baseline_history(athlete.athlete_id)

        sessions_query = SessionRecord.query.filter_by(athlete_id=athlete.athlete_id).order_by(SessionRecord.session_number.asc()).all()
        session_count = len(sessions_query)

        latest_session = sessions_query[-1] if sessions_query else None
        earlier_sessions = [s.to_dict() for s in sessions_query[:-1]] if latest_session else []

        workload = None
        if latest_session:
            workload = compute_history_summary(earlier_sessions, latest_session.training_load)

        # Mini chart data: last 14 sessions training load
        chart_sessions = sessions_query[-14:]
        chart_labels = [f"#{s.session_number}" for s in chart_sessions]
        chart_loads = [s.training_load for s in chart_sessions]

        return render_template(
            "dashboard.html",
            athlete=athlete,
            session_count=session_count,
            latest_session=latest_session,
            workload=workload,
            chart_labels=json.dumps(chart_labels),
            chart_loads=json.dumps(chart_loads),
        )

    @app.route("/athlete-details", methods=["GET", "POST"])
    @login_required
    def athlete_details():
        user = db.session.get(User, session["user_id"])
        athlete = user.athlete if user else None
        if not athlete and user:
            athlete = Athlete(user_id=user.user_id)
            db.session.add(athlete)
            db.session.commit()

        if athlete:
            ensure_baseline_history(athlete.athlete_id)

        if request.method == "POST":
            try:
                age_val = request.form.get("age", "").strip()
                athlete.age = int(age_val) if age_val else None
                athlete.gender = request.form.get("gender", "").strip() or None
                athlete.sport_type = request.form.get("sport_type", "").strip() or None
                h_val = request.form.get("height_cm", "").strip()
                athlete.height_cm = float(h_val) if h_val else None
                w_val = request.form.get("weight_kg", "").strip()
                athlete.weight_kg = float(w_val) if w_val else None

                db.session.commit()
                flash("Profile details updated successfully.", "success")
            except ValueError:
                flash("Please enter valid numeric values for age, height, and weight.", "danger")

        session_count = SessionRecord.query.filter_by(athlete_id=athlete.athlete_id).count()

        return render_template(
            "athlete_details.html",
            athlete=athlete,
            session_count=session_count,
        )

    @app.route("/api/athlete/load-demo-data", methods=["POST"])
    @login_required
    def load_demo_data():
        user = db.session.get(User, session["user_id"])
        athlete = user.athlete if user else None
        if not athlete and user:
            athlete = Athlete(user_id=user.user_id)
            db.session.add(athlete)
            db.session.commit()

        # Clear existing sessions for this athlete
        SessionRecord.query.filter_by(athlete_id=athlete.athlete_id).delete()

        # Insert 14 demo sessions
        for s in DEMO_SESSIONS:
            rec = SessionRecord(
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
            db.session.add(rec)

        db.session.commit()
        flash("Loaded 14 sample history demo sessions. Your next session is session 15.", "success")
        return redirect(url_for("dashboard"))

    @app.route("/api/athlete/delete-account", methods=["POST"])
    @login_required
    def delete_account():
        user = db.session.get(User, session["user_id"])
        if user:
            db.session.delete(user)
            db.session.commit()
        session.clear()
        flash("Your account and all associated data have been permanently deleted.", "info")
        return redirect(url_for("index"))

    # -------------------------------------------------------------
    # SESSION MANAGEMENT & PREDICTIONS
    # -------------------------------------------------------------
    @app.route("/add-session", methods=["GET", "POST"])
    @login_required
    def add_session():
        user = db.session.get(User, session["user_id"])
        athlete = user.athlete
        if athlete:
            ensure_baseline_history(athlete.athlete_id)

        # Check if selected model is available
        selected_meta = ModelLoader.get_selected_metadata()
        if not selected_meta:
            flash("Selected model not found. Run training/run_all.py first.", "danger")

        if request.method == "POST":
            form_dict = request.form.to_dict()
            parsed_vals, errors, warnings = validate_session_inputs(form_dict)

            if errors:
                for err in errors:
                    flash(err, "danger")
                return render_template("add_session.html", form_vals=form_dict, warnings=warnings)

            for w in warnings:
                flash(w, "warning")

            # Determine next session number
            existing_count = SessionRecord.query.filter_by(athlete_id=athlete.athlete_id).count()
            next_session_num = existing_count + 1

            # Check if prediction threshold reached
            predicted_class = None
            prob_low = None
            prob_moderate = None
            prob_high = None
            model_name = None

            if next_session_num >= config.FIRST_PREDICTION_SESSION:
                try:
                    pred_res = predict_session(parsed_vals)
                    predicted_class = pred_res["predicted_class"]
                    prob_low = pred_res["probabilities_raw"][0]
                    prob_moderate = pred_res["probabilities_raw"][1]
                    prob_high = pred_res["probabilities_raw"][2]
                    model_name = pred_res["model_name"]
                except ModelNotFoundError as e:
                    flash(str(e), "danger")
                    return redirect(url_for("dashboard"))
                except Exception as e:
                    flash(f"Prediction inference error: {str(e)}", "danger")
                    return redirect(url_for("dashboard"))

            # Save session
            new_record = SessionRecord(
                athlete_id=athlete.athlete_id,
                session_number=next_session_num,
                training_duration=parsed_vals["training_duration"],
                training_intensity=parsed_vals["training_intensity"] if not np.isnan(parsed_vals["training_intensity"]) else None,
                training_load=parsed_vals["training_load"],
                heart_rate=parsed_vals["heart_rate"] if not np.isnan(parsed_vals["heart_rate"]) else None,
                sleep_quality=parsed_vals["sleep_quality"] if not np.isnan(parsed_vals["sleep_quality"]) else None,
                recovery_score=parsed_vals["recovery_score"],
                stress_level=parsed_vals["stress_level"],
                predicted_class=predicted_class,
                prob_low=prob_low,
                prob_moderate=prob_moderate,
                prob_high=prob_high,
                model_name=model_name,
                source="user",
            )
            db.session.add(new_record)
            db.session.commit()

            flash(f"Session #{next_session_num} saved successfully.", "success")
            return redirect(url_for("view_session", session_id=new_record.id))

        existing_count = SessionRecord.query.filter_by(athlete_id=athlete.athlete_id).count()
        next_session_num = existing_count + 1

        return render_template(
            "add_session.html",
            next_session_num=next_session_num,
            form_vals={},
            warnings=[],
        )

    @app.route("/session/<int:session_id>")
    @login_required
    def view_session(session_id):
        user = db.session.get(User, session["user_id"])
        athlete = user.athlete
        if athlete:
            ensure_baseline_history(athlete.athlete_id)

        record = db.session.get(SessionRecord, session_id)
        if not record:
            abort(404)
        # Strict ownership check
        if record.athlete_id != athlete.athlete_id:
            abort(403)

        # Fetch earlier sessions for history context
        earlier_query = SessionRecord.query.filter(
            SessionRecord.athlete_id == athlete.athlete_id,
            SessionRecord.session_number < record.session_number
        ).order_by(SessionRecord.session_number.asc()).all()

        earlier_sessions = [s.to_dict() for s in earlier_query]
        current_dict = record.to_dict()
        workload = compute_history_summary(earlier_sessions, record.training_load, current_session_dict=current_dict)

        # State labels
        state_labels = get_all_state_labels(record.sleep_quality, record.recovery_score, record.stress_level)

        # Explainability & What-If
        explanation = None
        what_if_data = None
        if record.predicted_class:
            feat_dict = {
                "training_duration": record.training_duration,
                "training_intensity": record.training_intensity,
                "training_load": record.training_load,
                "heart_rate": record.heart_rate,
                "sleep_quality": record.sleep_quality,
                "recovery_score": record.recovery_score,
                "stress_level": record.stress_level,
            }
            try:
                explanation = explain_prediction(feat_dict, workload=workload)
                what_if_data = compute_what_if(feat_dict)
            except Exception as e:
                explanation = None

        return render_template(
            "result.html",
            record=record,
            athlete=athlete,
            workload=workload,
            state_labels=state_labels,
            explanation=explanation,
            what_if_data=what_if_data,
        )

    @app.route("/session/<int:session_id>/edit", methods=["GET", "POST"])
    @login_required
    def edit_session(session_id):
        user = db.session.get(User, session["user_id"])
        athlete = user.athlete

        record = db.session.get(SessionRecord, session_id)
        if not record:
            abort(404)
        if record.athlete_id != athlete.athlete_id:
            abort(403)

        # Rule: Only latest session can be edited
        latest_record = SessionRecord.query.filter_by(athlete_id=athlete.athlete_id).order_by(SessionRecord.session_number.desc()).first()
        if latest_record.id != record.id:
            flash("Only the latest session record can be edited.", "warning")
            return redirect(url_for("history"))

        if request.method == "POST":
            form_dict = request.form.to_dict()
            parsed_vals, errors, warnings = validate_session_inputs(form_dict)

            if errors:
                for err in errors:
                    flash(err, "danger")
                return render_template("edit_session.html", record=record, form_vals=form_dict)

            for w in warnings:
                flash(w, "warning")

            record.training_duration = parsed_vals["training_duration"]
            record.training_intensity = parsed_vals["training_intensity"] if not np.isnan(parsed_vals["training_intensity"]) else None
            record.training_load = parsed_vals["training_load"]
            record.heart_rate = parsed_vals["heart_rate"] if not np.isnan(parsed_vals["heart_rate"]) else None
            record.sleep_quality = parsed_vals["sleep_quality"] if not np.isnan(parsed_vals["sleep_quality"]) else None
            record.recovery_score = parsed_vals["recovery_score"]
            record.stress_level = parsed_vals["stress_level"]

            if record.session_number >= config.FIRST_PREDICTION_SESSION:
                pred_res = predict_session(parsed_vals)
                record.predicted_class = pred_res["predicted_class"]
                record.prob_low = pred_res["probabilities_raw"][0]
                record.prob_moderate = pred_res["probabilities_raw"][1]
                record.prob_high = pred_res["probabilities_raw"][2]
                record.model_name = pred_res["model_name"]

            db.session.commit()
            flash(f"Session #{record.session_number} updated successfully.", "success")
            return redirect(url_for("view_session", session_id=record.id))

        return render_template("edit_session.html", record=record, form_vals=record.to_dict())

    @app.route("/session/<int:session_id>/delete", methods=["POST"])
    @login_required
    def delete_session(session_id):
        user = db.session.get(User, session["user_id"])
        athlete = user.athlete

        record = db.session.get(SessionRecord, session_id)
        if not record:
            abort(404)
        if record.athlete_id != athlete.athlete_id:
            abort(403)

        # Rule: Only latest session can be deleted
        latest_record = SessionRecord.query.filter_by(athlete_id=athlete.athlete_id).order_by(SessionRecord.session_number.desc()).first()
        if latest_record.id != record.id:
            flash("Only the latest session record can be deleted.", "warning")
            return redirect(url_for("history"))

        db.session.delete(record)
        db.session.commit()
        flash(f"Session #{record.session_number} has been deleted.", "info")
        return redirect(url_for("history"))

    @app.route("/history")
    @login_required
    def history():
        user = db.session.get(User, session["user_id"])
        athlete = user.athlete
        if athlete:
            ensure_baseline_history(athlete.athlete_id)

        sessions = SessionRecord.query.filter_by(athlete_id=athlete.athlete_id).order_by(SessionRecord.session_number.asc()).all()
        latest_id = sessions[-1].id if sessions else None

        # Prepare chart datasets
        session_nums = [s.session_number for s in sessions]
        load_data = [s.training_load for s in sessions]
        recovery_data = [s.recovery_score for s in sessions]
        stress_data = [s.stress_level for s in sessions]
        class_map = {"Low": 0, "Moderate": 1, "High": 2}
        class_data = [class_map.get(s.predicted_class, None) for s in sessions]

        return render_template(
            "history.html",
            sessions=sessions,
            latest_id=latest_id,
            chart_data=json.dumps({
                "labels": [f"#{n}" for n in session_nums],
                "loads": load_data,
                "recovery": recovery_data,
                "stress": stress_data,
                "classes": class_data,
            }),
        )

    # -------------------------------------------------------------
    # AI ASSISTANT & ABOUT PAGES
    # -------------------------------------------------------------
    @app.route("/assistant")
    @login_required
    def assistant():
        user = db.session.get(User, session["user_id"])
        athlete = user.athlete

        sessions_query = SessionRecord.query.filter_by(athlete_id=athlete.athlete_id).order_by(SessionRecord.session_number.asc()).all()
        context = build_assistant_context(athlete.athlete_id, sessions_query)

        return render_template("assistant.html", context=context)

    @app.route("/api/assistant", methods=["POST"])
    @login_required
    def api_assistant():
        user = db.session.get(User, session["user_id"])
        athlete = user.athlete

        data = request.get_json(silent=True) or {}
        prompt = data.get("prompt", "").strip()

        if not prompt:
            return jsonify({"error": "Prompt cannot be empty"}), 400

        sessions_query = SessionRecord.query.filter_by(athlete_id=athlete.athlete_id).order_by(SessionRecord.session_number.asc()).all()
        context = build_assistant_context(athlete.athlete_id, sessions_query)

        reply = ask_assistant(prompt, context)
        return jsonify({"reply": reply})

    @app.route("/api/what-if", methods=["POST"])
    @login_required
    def api_what_if():
        data = request.get_json(silent=True) or {}
        base_features = data.get("base_features", {})
        new_sleep = data.get("sleep_quality")
        new_recovery = data.get("recovery_score")
        new_stress = data.get("stress_level")

        try:
            res = compute_what_if(
                base_features,
                new_sleep=float(new_sleep) if new_sleep is not None else None,
                new_recovery=float(new_recovery) if new_recovery is not None else None,
                new_stress=float(new_stress) if new_stress is not None else None,
            )
            return jsonify(res)
        except Exception as e:
            return jsonify({"error": str(e)}), 400



    @app.route("/figures/<path:filename>")
    def serve_figure(filename):
        return send_from_directory(str(config.FIGURES_DIR), filename)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="127.0.0.1", port=5000, debug=True)
