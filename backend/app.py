from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, send_from_directory
from flask_cors import CORS

from models.database import init_db

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

from routes import api_bp

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"


def create_app() -> Flask:
    app = Flask(__name__)
    app.secret_key = os.environ.get("SECRET_KEY", "dev-healthcare-secret-change-me")
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    CORS(app, supports_credentials=True, resources={r"/*": {"origins": "*"}})

    init_db()
    app.register_blueprint(api_bp)

    @app.route("/")
    def index():
        return send_from_directory(FRONTEND, "index.html")

    @app.route("/login.html")
    def login_page():
        return send_from_directory(FRONTEND, "login.html")

    @app.route("/dashboard.html")
    def dashboard_page():
        return send_from_directory(FRONTEND, "dashboard.html")

    @app.route("/verify.html")
    def verify_page():
        return send_from_directory(FRONTEND, "verify.html")

    @app.route("/planner.html")
    def planner_page():
        return send_from_directory(FRONTEND, "planner.html")

    @app.route("/patient.html")
    def patient_page():
        return send_from_directory(FRONTEND, "patient.html")

    @app.route("/doctor.html")
    def doctor_page():
        return send_from_directory(FRONTEND, "doctor.html")

    @app.route("/css/<path:filename>")
    def css(filename: str):
        return send_from_directory(FRONTEND / "css", filename)

    @app.route("/js/<path:filename>")
    def js(filename: str):
        return send_from_directory(FRONTEND / "js", filename)

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "False") == "True"

    app.run(host="0.0.0.0", port=port, debug=debug)
