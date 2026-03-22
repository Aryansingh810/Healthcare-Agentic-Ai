from flask import Blueprint

api_bp = Blueprint("api", __name__, url_prefix="")

from . import api_routes
