from __future__ import annotations

import importlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from flask import jsonify, request, session
from werkzeug.utils import secure_filename

from models.database import get_db
from ocr_service import ocr_result_to_dict, run_ocr

from . import api_bp
from utils.auth import hash_password, login_required, verify_password

UPLOAD_DIR = Path(__file__).resolve().parent.parent / "uploads"
VERIFY_THRESHOLD = float(os.environ.get("ID_VERIFY_CONFIDENCE_THRESHOLD", "65"))

MOCK_APPOINTMENTS = [
    {
        "id": 1,
        "patient": "Alex Morgan",
        "time": "2025-03-24T09:30:00Z",
        "reason": "Follow-up diabetes management",
        "status": "confirmed",
    },
    {
        "id": 2,
        "patient": "Jamie Lee",
        "time": "2025-03-25T14:00:00Z",
        "reason": "New patient intake",
        "status": "pending",
    },
    {
        "id": 3,
        "patient": "Sam Rivera",
        "time": "2025-03-26T11:15:00Z",
        "reason": "Hypertension review",
        "status": "confirmed",
    },
]

MOCK_PATIENT_QUERIES = [
    {"id": 101, "from_patient": "Pat_01", "summary": "Questions about metformin side effects", "received": "2025-03-20"},
    {"id": 102, "from_patient": "Pat_02", "summary": "Travel vaccines before trip", "received": "2025-03-21"},
]


@api_bp.route("/signup", methods=["POST"])
def signup():
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""
    role = (data.get("role") or "").strip().lower()
    if not username or not password:
        return jsonify({"error": "username and password required"}), 400
    if role not in ("doctor", "patient"):
        return jsonify({"error": "role must be doctor or patient"}), 400
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
            (username, hash_password(password), role),
        )
        conn.commit()
        uid = cur.lastrowid
    except sqlite3.IntegrityError:
        conn.rollback()
        return jsonify({"error": "Username already exists"}), 409
    except Exception:
        conn.rollback()
        return jsonify({"error": "Could not create user"}), 500
    finally:
        conn.close()
    session["user_id"] = uid
    session["username"] = username
    session["role"] = role
    return jsonify({"ok": True, "user_id": uid, "role": role})


@api_bp.route("/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""
    if not username or not password:
        return jsonify({"error": "username and password required"}), 400
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT id, password_hash, role FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    if not row or not verify_password(row["password_hash"], password):
        return jsonify({"error": "Invalid credentials"}), 401
    session["user_id"] = row["id"]
    session["username"] = username
    session["role"] = row["role"]
    return jsonify({"ok": True, "user_id": row["id"], "role": row["role"]})


@api_bp.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"ok": True})


@api_bp.route("/session", methods=["GET"])
def session_info():
    uid = session.get("user_id")
    if not uid:
        return jsonify({"logged_in": False}), 200
    return jsonify(
        {
            "logged_in": True,
            "user_id": uid,
            "username": session.get("username"),
            "role": session.get("role"),
        }
    )


@api_bp.route("/upload-id", methods=["POST"])
@login_required(role="doctor")
def upload_id():
    if "file" not in request.files:
        return jsonify({"error": "file field required"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    name = secure_filename(f.filename)
    uid = session["user_id"]
    path = UPLOAD_DIR / f"doc_{uid}_{name}"
    f.save(path)
    session["last_id_upload"] = str(path)
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT id FROM doctor_profiles WHERE user_id = ?", (uid,))
    row = cur.fetchone()
    if row:
        cur.execute(
            "UPDATE doctor_profiles SET id_image_path = ? WHERE user_id = ?",
            (str(path), uid),
        )
    else:
        cur.execute(
            "INSERT INTO doctor_profiles (user_id, id_image_path, verified) VALUES (?, ?, 0)",
            (uid, str(path)),
        )
    conn.commit()
    conn.close()
    return jsonify({"ok": True, "path": str(path)})


@api_bp.route("/verify-id", methods=["POST"])
@login_required(role="doctor")
def verify_id():
    data = request.get_json(silent=True) or {}
    path = data.get("path") or session.get("last_id_upload")
    if not path or not Path(path).is_file():
        return jsonify({"error": "Upload an ID image first via /upload-id"}), 400
    try:
        result = run_ocr(path)
    except Exception as e:
        return jsonify({"error": f"OCR failed: {e}"}), 500

    payload = ocr_result_to_dict(result)
    fields = payload["fields"]
    conf = payload["confidence"]
    field_score = min(100.0, len(fields) * 18.0)
    combined = (conf * 0.6 + field_score * 0.4) / 100.0
    combined_100 = combined * 100
    verified = combined_100 >= VERIFY_THRESHOLD and len(fields) >= 2

    uid = session["user_id"]
    now = datetime.now(timezone.utc).isoformat()
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE doctor_profiles SET
            verified = ?,
            name = COALESCE(?, name),
            specialty = COALESCE(?, specialty),
            phone = COALESCE(?, phone),
            email = COALESCE(?, email),
            id_number = COALESCE(?, id_number),
            ocr_confidence = ?,
            verified_at = CASE WHEN ? = 1 THEN ? ELSE verified_at END
        WHERE user_id = ?
        """,
        (
            1 if verified else 0,
            fields.get("name"),
            fields.get("specialty"),
            fields.get("phone"),
            fields.get("email"),
            fields.get("id_number"),
            conf,
            1 if verified else 0,
            now,
            uid,
        ),
    )
    if cur.rowcount == 0:
        cur.execute(
            """
            INSERT INTO doctor_profiles (
                user_id, verified, name, specialty, phone, email, id_number,
                ocr_confidence, id_image_path, verified_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                uid,
                1 if verified else 0,
                fields.get("name"),
                fields.get("specialty"),
                fields.get("phone"),
                fields.get("email"),
                fields.get("id_number"),
                conf,
                path,
                now if verified else None,
            ),
        )
    conn.commit()
    conn.close()

    status = "VERIFIED" if verified else "PENDING_REVIEW"
    return jsonify(
        {
            "extracted": payload,
            "combined_confidence": round(combined_100, 2),
            "threshold": VERIFY_THRESHOLD,
            "verification_status": status,
        }
    )


@api_bp.route("/planner-agent", methods=["POST"])
@login_required(role="doctor")
def planner_agent():
    data = request.get_json(silent=True) or {}
    goal = data.get("goal") or ""
    pa = importlib.import_module("planner_agent")
    out = pa.run_planner_agent(goal)
    uid = session["user_id"]
    conn = get_db()
    cur = conn.cursor()

    cur.execute(
        "INSERT INTO planner_runs (user_id, goal, result_json) VALUES (?, ?, ?)",
        (uid, goal, json.dumps(out)),
    )
    conn.commit()
    conn.close()
    return jsonify(out)


@api_bp.route("/symptom-check", methods=["POST"])
@login_required(role="patient")
def symptom_check():
    from services.symptom_service import analyze_symptoms

    data = request.get_json(silent=True) or {}
    symptoms = data.get("symptoms") or ""
    uid = session["user_id"]
    result = analyze_symptoms(symptoms, uid)
    if "error" in result:
        return jsonify(result), 400

    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO patient_queries (user_id, symptoms, response_json) VALUES (?, ?, ?)",
        (uid, symptoms, json.dumps(result)),
    )
    conn.commit()
    conn.close()
    return jsonify(result)


@api_bp.route("/doctor/appointments", methods=["GET"])
@login_required(role="doctor")
def doctor_appointments():
    return jsonify({"appointments": MOCK_APPOINTMENTS})


@api_bp.route("/doctor/patient-queries", methods=["GET"])
@login_required(role="doctor")
def doctor_patient_queries():
    return jsonify({"queries": MOCK_PATIENT_QUERIES})


@api_bp.route("/doctor/profile", methods=["GET"])
@login_required(role="doctor")
def doctor_profile():
    uid = session["user_id"]
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT verified, name, specialty, phone, email, id_number, ocr_confidence FROM doctor_profiles WHERE user_id = ?",
        (uid,),
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return jsonify({"profile": None})
    return jsonify({"profile": dict(row)})
