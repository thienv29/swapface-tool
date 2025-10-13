"""
API routes for face swapping operations.
"""
import logging
import sys
import os
from flask import Blueprint, request, jsonify
from typing import List, Tuple
import cv2

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


def _extract_file_list(files) -> tuple[List, List[str]]:
    """Extract uploaded files and their filenames."""
    target_files = []
    target_filenames = []

    for file in files:
        if file and file.filename:
            file_path = file_processor.save_uploaded_file(file, file.filename)
            target_files.append(file_path)
            target_filenames.append(file.filename)

    return target_files, target_filenames


def _validate_request(src_file, tgt_files) -> tuple[bool, str]:
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
def swapface_api():
    """Handle face swap requests."""
    try:
        # Check if already processing
        if batch_processor.get_progress() and not batch_processor.get_progress().is_complete:
            return jsonify({"error": "Processing is already in progress"}), 409

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
def view_file(filename):
    """Serve processed files."""
    from flask import send_file
    mimetype = "image/jpeg" if filename.lower().endswith((".jpg", ".jpeg")) else "video/mp4"
    return send_file(filename, mimetype=mimetype)
