"""
API routes for face swapping operations.
"""
import logging
import sys
import os
from flask import Blueprint, request, jsonify
from flask_httpauth import HTTPBasicAuth
from typing import List, Tuple
import cv2

# Import auth from app.py
from werkzeug.security import generate_password_hash, check_password_hash

# Initialize HTTP Basic Auth (shared with app.py)
auth = HTTPBasicAuth()
USERS = {
    "admin": generate_password_hash("Thien1lan@123")
}

@auth.verify_password
def verify_password(username, password):
    """Verify username and password."""
    if username in USERS and check_password_hash(USERS.get(username), password):
        return username
    return None

# Add parent directory to path for absolute imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import get_config
from core.face_services import FaceServices
from core.swap_processor import SwapProcessor, BatchProcessor, FileProcessor
from core.models import SwapResult

logger = logging.getLogger(__name__)

# Initialize services
config = get_config()
face_services = FaceServices(config)
swap_processor = SwapProcessor(config, face_services)
batch_processor = BatchProcessor(config, swap_processor)
file_processor = FileProcessor(config)

# Initialize models
face_services.initialize()

# Create blueprint
api_bp = Blueprint('api', __name__)


def _extract_file_list(files, max_total_size_mb: float = 2000.0):
    """Extract uploaded files and their filenames with size validation."""
    target_files = []
    target_filenames = []
    total_size = 0

    for file in files:
        if file and file.filename:
            # Check file size before processing
            file.seek(0, 2)  # Seek to end to get size
            file_size = file.tell()
            file.seek(0)  # Seek back to beginning

            total_size += file_size

            # Check if adding this file would exceed total size limit
            if total_size > max_total_size_mb * 1024 * 1024:
                # Clean up already saved files
                for saved_path in target_files:
                    try:
                        if os.path.exists(saved_path):
                            os.remove(saved_path)
                    except Exception as e:
                        logger.warning(f"Failed to cleanup {saved_path}: {e}")
                raise ValueError(f"Total upload size exceeds maximum limit of {max_total_size_mb}MB")

            try:
                file_path = file_processor.save_uploaded_file(file, file.filename)
                target_files.append(file_path)
                target_filenames.append(file.filename)
                logger.info(f"Uploaded file: {file.filename} ({file_size / 1024 / 1024:.1f}MB)")
            except Exception as e:
                # Clean up on failure
                for saved_path in target_files:
                    try:
                        if os.path.exists(saved_path):
                            os.remove(saved_path)
                    except Exception as cleanup_e:
                        logger.warning(f"Failed to cleanup {saved_path}: {cleanup_e}")
                raise e

    logger.info(f"Total upload size: {total_size / 1024 / 1024:.1f}MB for {len(target_files)} files")
    return target_files, target_filenames


def _validate_request(src_file, tgt_files):
    """Validate incoming request."""
    if not src_file or not src_file.filename:
        return False, "Missing source file"

    if not tgt_files:
        return False, "Missing target files"

    # Validate source file
    if not file_processor.validate_file(src_file.filename):
        return False, f"Unsupported source file type: {src_file.filename}"

    # Validate target files
    for tgt_file in tgt_files:
        if not tgt_file or not tgt_file.filename:
            continue
        if not file_processor.validate_file(tgt_file.filename):
            return False, f"Unsupported target file type: {tgt_file.filename}"

    return True, ""


@api_bp.route("/swapface", methods=["POST"])
@auth.login_required
def swapface_api():
    """Handle face swap requests."""
    try:
        # Check if already processing
        if batch_processor.get_progress() and not batch_processor.get_progress().is_complete:
            return jsonify({
                "error": "Processing is already in progress. Please wait for the current batch to complete.",
                "retry_after": 30  # Suggest retry after 30 seconds
            }), 409

        src_file = request.files.get("source")
        tgt_files = request.files.getlist("targets")

        # Validate request
        is_valid, error_msg = _validate_request(src_file, tgt_files)
        if not is_valid:
            return jsonify({"error": error_msg}), 400

        # Save files
        src_path = file_processor.save_uploaded_file(src_file, src_file.filename)
        target_files, target_filenames = _extract_file_list(tgt_files)

        if not target_files:
            return jsonify({"error": "Cannot save any target files"}), 400

        # Validate source image
        source_img = cv2.imread(src_path)
        if source_img is None:
            file_processor.cleanup_temp_files([src_path])
            return jsonify({"error": "Cannot read source image"}), 400

        # Validate source face
        source_faces = face_services.detector.detect_faces(source_img)
        if not source_faces.success:
            file_processor.cleanup_temp_files([src_path])
            return jsonify({"error": source_faces.error_message or "No source face detected"}), 400

        # Start background processing
        success = batch_processor.start_background_processing(
            src_path, target_files, target_filenames
        )

        if not success:
            file_processor.cleanup_temp_files([src_path] + target_files)
            return jsonify({"error": "Cannot start processing"}), 500

        return jsonify({
            "message": "Processing started",
            "total_files": len(target_files)
        }), 202

    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@api_bp.route("/status", methods=["GET"])
@auth.login_required
def get_status():
    """Get current processing status."""
    try:
        progress = batch_processor.get_progress()
        if not progress:
            return jsonify({
                "is_processing": False,
                "completed": 0,
                "total": 0,
                "progress_percentage": 0.0,
                "file_progress": {},
                "current_file": None,
                "queue": []
            })

        current_time = __import__('time').time()
        elapsed = current_time - progress.start_time if progress.start_time else 0

        status = {
            "is_processing": not progress.is_complete,
            "completed": progress.completed,
            "total": progress.total_files,
            "progress_percentage": progress.progress_percentage,
            "current_file": progress.current_file,
            "queue": progress.queue,
            "start_time": progress.start_time,
            "elapsed": elapsed,
            "results": []
        }

        # Add speed and ETA
        if elapsed > 0 and progress.completed > 0:
            status["speed"] = progress.completed / elapsed
            if status["speed"] > 0 and progress.completed < progress.total_files:
                remaining = progress.total_files - progress.completed
                status["eta_seconds"] = remaining / status["speed"]
        else:
            status["speed"] = 0.0
            status["eta_seconds"] = 0.0

        # Add video progress
        if progress.video_progress:
            status.update(progress.video_progress)

        # Format file progress for API response
        file_progress_data = {}
        if progress.file_progress:
            for filename, file_prog in progress.file_progress.items():
                file_progress_data[filename] = {
                    "filename": file_prog.filename,
                    "progress_percentage": file_prog.progress_percentage,
                    "status": file_prog.status,
                    "current_frame": file_prog.current_frame,
                    "total_frames": file_prog.total_frames,
                    "file_type": file_prog.file_type,
                    "start_time": file_prog.start_time,
                    "estimated_time": file_prog.estimated_time,
                    "progress_text": file_prog.progress_text
                }
        status["file_progress"] = file_progress_data

        # Format results for API response
        if progress.results:
            status["results"] = [
                {
                    "success": r.success,
                    "error_message": r.error_message,
                    "original_name": r.original_name,
                    "file_type": r.file_type.value,
                    "result": r.output_path
                } for r in progress.results
            ]

        return jsonify(status)

    except Exception as e:
        logger.error(f"Status API error: {e}")
        return jsonify({"error": f"Status API error: {str(e)}"}), 500


@api_bp.route("/view/<path:filename>")
@auth.login_required
def view_file(filename):
    """Serve processed files."""
    from flask import send_file, Response
    import mimetypes

    # Get proper mimetype
    mimetype, _ = mimetypes.guess_type(filename)

    # Default fallback mimetypes
    if not mimetype:
        if filename.lower().endswith((".jpg", ".jpeg")):
            mimetype = "image/jpeg"
        elif filename.lower().endswith(".png"):
            mimetype = "image/png"
        elif filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            mimetype = "video/mp4"
        else:
            mimetype = "application/octet-stream"

    try:
        # Ensure file exists
        if not os.path.exists(filename):
            logger.error(f"File not found: {filename}")
            return jsonify({"error": "File not found"}), 404

        # Get file size for headers
        file_size = os.path.getsize(filename)

        response = send_file(
            filename,
            mimetype=mimetype,
            as_attachment=False,
            conditional=True
        )

        # Add headers for better browser support
        response.headers['Cache-Control'] = 'public, max-age=3600'
        response.headers['Content-Length'] = file_size

        return response

    except Exception as e:
        logger.error(f"Error serving file {filename}: {e}")
        return jsonify({"error": "Error serving file"}), 500


@api_bp.route("/download/<path:filename>")
@auth.login_required
def download_file(filename):
    """Download processed files."""
    from flask import send_file
    import mimetypes

    # Handle various URL formats: view/output/, output/, or just filename
    if filename.startswith('view/output/'):
        # Remove view/ prefix, keep output/
        filepath = filename[5:]  # Remove 'view/' prefix
    elif filename.startswith('output/'):
        # Use as is
        filepath = filename
    else:
        # Assume it's just the filename, prepend output/
        filepath = f"output/{filename}"

    # Get proper mimetype
    mimetype, _ = mimetypes.guess_type(filepath)

    # Default fallback mimetypes
    if not mimetype:
        if filepath.lower().endswith((".jpg", ".jpeg")):
            mimetype = "image/jpeg"
        elif filepath.lower().endswith(".png"):
            mimetype = "image/png"
        elif filepath.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            mimetype = "video/mp4"
        else:
            mimetype = "application/octet-stream"

    try:
        # Ensure file exists
        if not os.path.exists(filepath):
            logger.error(f"Download file not found: {filepath}")
            return jsonify({"error": "File not found"}), 404

        # Extract original filename for download
        original_filename = filepath.split('_', 1)[-1] if '_' in filepath else filepath
        original_filename = original_filename.replace('output/', '')

        response = send_file(
            filepath,
            mimetype=mimetype,
            as_attachment=True,
            download_name=original_filename,
            conditional=True
        )

        # Add headers for download
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'

        return response

    except Exception as e:
        logger.error(f"Error downloading file {filepath}: {e}")
        return jsonify({"error": "Error downloading file"}), 500
