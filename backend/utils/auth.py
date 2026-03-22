from functools import wraps
from typing import Any, Callable

from flask import jsonify, session
from werkzeug.security import check_password_hash, generate_password_hash


def hash_password(password: str) -> str:
    return generate_password_hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    return check_password_hash(password_hash, password)


def login_required(role: str | None = None) -> Callable:
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def wrapped(*args: Any, **kwargs: Any):
            uid = session.get("user_id")
            if not uid:
                return jsonify({"error": "Authentication required"}), 401
            if role:
                r = session.get("role")
                if r != role:
                    return jsonify({"error": f"This action requires role: {role}"}), 403
            return f(*args, **kwargs)

        return wrapped

    return decorator
