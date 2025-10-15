"""
Main application module for face swapping service.
"""
import logging
from flask import Flask, render_template
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash
from core.config import get_config, setup_environment
from routes.api import api_bp
from flask_swagger_ui import get_swaggerui_blueprint

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize HTTP Basic Auth
auth = HTTPBasicAuth()

# Hardcoded credentials
USERS = {
    "admin": generate_password_hash("Thien1lan@123")
}

@auth.verify_password
def verify_password(username, password):
    """Verify username and password."""
    if username in USERS and check_password_hash(USERS.get(username), password):
        return username
    return None


def create_app() -> Flask:
    """Application factory pattern."""
    app = Flask(__name__, template_folder="templates")

    # Load configuration
    config = get_config()

    # Setup environment (GPU, threads, directories)
    setup_environment(config)

    # Configure Flask for large file uploads
    app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
    app.config['UPLOAD_FOLDER'] = config.temp_directory

    logger.info("🚀 Initializing face swapping application...")

    # Swagger UI
    SWAGGER_URL = '/api/docs'
    API_URL = '/static/swagger.json'
    swaggerui_blueprint = get_swaggerui_blueprint(
        SWAGGER_URL,
        API_URL,
        config={'app_name': "Face Swap API"}
    )
    app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

    # Register blueprints
    app.register_blueprint(api_bp)

    # Web routes
    @app.route("/")
    @auth.login_required
    def index():
        """Render main page."""
        return render_template("index.html")

    return app


if __name__ == "__main__":
    app = create_app()
    config = get_config()
    logger.info("🚀 Starting face swapping service...")
    app.run(host=config.host, port=config.port, debug=config.debug)
