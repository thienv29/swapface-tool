"""
Configuration management for face swapping application.
"""
import os
from .models import Config


def setup_environment(config: Config):
    """Setup environment variables for GPU optimization."""
    os.environ["OMP_NUM_THREADS"] = str(config.omp_num_threads)
    os.environ["MKL_NUM_THREADS"] = str(config.mkl_num_threads)
    os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices

    # Create output directory
    os.makedirs(config.output_directory, exist_ok=True)


def get_config() -> Config:
    """Get application configuration from environment variables or defaults."""
    return Config(
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        debug=os.getenv("FLASK_DEBUG", "true").lower() == "true",
        output_directory=os.getenv("OUTPUT_DIR", "output"),
        temp_directory=os.getenv("TEMP_DIR", "/tmp"),
        max_batch_size=int(os.getenv("MAX_BATCH_SIZE", "8")),
        cuda_visible_devices=os.getenv("CUDA_VISIBLE_DEVICES", "0"),
        omp_num_threads=int(os.getenv("OMP_NUM_THREADS", "4")),
        mkl_num_threads=int(os.getenv("MKL_NUM_THREADS", "4"))
    )
