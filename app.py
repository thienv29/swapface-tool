"""
Main application module for face swapping service.
"""
import logging
from flask import Flask, render_template
from core.config import get_config, setup_environment
from routes.api import api_bp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_app() -> Flask:
    """Application factory pattern."""
    app = Flask(__name__, template_folder="templates")

    # Load configuration
    config = get_config()

    # Setup environment (GPU, threads, directories)
    setup_environment(config)

    logger.info("🚀 Initializing face swapping application...")

    # Register blueprints
    app.register_blueprint(api_bp)

    # Web routes
    @app.route("/")
    def index():
        """Render main page."""
        return render_template("index.html")

    return app


if __name__ == "__main__":
    app = create_app()
    config = get_config()
    logger.info("🚀 Starting face swapping service...")
    app.run(host=config.host, port=config.port, debug=config.debug)
