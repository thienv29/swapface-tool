"""
API routes for face swapping operations.
"""
import logging
import sys
import os
import aiofiles
import asyncio
import uuid
import requests
from flask import Blueprint, request, jsonify, Response, send_file
from flask_httpauth import HTTPBasicAuth
from typing import List, Tuple, Dict, Any, Optional
import cv2
import threading
import time
import io

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

# Global upload progress tracking
upload_progress: Dict[str, Dict[str, Any]] = {}
progress_lock = threading.Lock()

# Global chunk storage for resumable uploads
chunk_storage: Dict[str, Dict[str, Any]] = {}
chunk_lock = threading.Lock()

# Create blueprint
api_bp = Blueprint('api', __name__)


def _extract_file_list(files, max_total_size_mb: float = 20000.0, upload_id: Optional[str] = None, progress_lock=None):
    """Extract uploaded files and their filenames with size validation and progress tracking."""
    target_files = []
    target_filenames = []
    total_size = 0

    def update_progress(current_size: int, total_size: int):
        """Update upload progress for this session."""
        if upload_id and progress_lock:
            with progress_lock:
                if upload_id in upload_progress:
                    upload_progress[upload_id]["bytes_uploaded"] = current_size
                    upload_progress[upload_id]["total_bytes"] = total_size
                    upload_progress[upload_id]["progress_percentage"] = (
                        (current_size / total_size) * 100 if total_size > 0 else 0
                    )

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

                # Update progress after each file
                update_progress(total_size, max_total_size_mb * 1024 * 1024)

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


@api_bp.route("/upload-progress/<upload_id>", methods=["GET"])
@auth.login_required
def get_upload_progress(upload_id: str):
    """Get upload progress for a specific upload ID."""
    with progress_lock:
        if upload_id in upload_progress:
            return jsonify(upload_progress[upload_id])
        else:
            return jsonify({"error": "Upload ID not found"}), 404


def create_upload_session(total_files: int, total_size_mb: float) -> str:
    """Create a new upload session and return the upload ID."""
    upload_id = str(uuid.uuid4())
    with progress_lock:
        upload_progress[upload_id] = {
            "upload_id": upload_id,
            "start_time": time.time(),
            "total_files": total_files,
            "files_uploaded": 0,
            "bytes_uploaded": 0,
            "total_bytes": int(total_size_mb * 1024 * 1024),
            "progress_percentage": 0.0,
            "status": "uploading",
            "estimated_time_remaining": 0.0
        }
    logger.info(f"Created upload session: {upload_id}")
    return upload_id


def cleanup_expired_uploads():
    """Clean up expired upload sessions (older than 24 hours)."""
    current_time = time.time()
    expired_ids = []
    with progress_lock:
        for upload_id, session in upload_progress.items():
            if current_time - session["start_time"] > 24 * 3600:  # 24 hours
                expired_ids.append(upload_id)

        for upload_id in expired_ids:
            del upload_progress[upload_id]
            logger.info(f"Cleaned up expired upload session: {upload_id}")


def cleanup_expired_chunks():
    """Clean up expired chunk sessions (older than 24 hours)."""
    current_time = time.time()
    expired_ids = []
    with chunk_lock:
        for chunk_id, session in chunk_storage.items():
            if current_time - session["start_time"] > 24 * 3600:  # 24 hours
                expired_ids.append(chunk_id)
                # Clean up chunk files
                try:
                    import os
                    import shutil
                    if os.path.exists(session["chunk_dir"]):
                        shutil.rmtree(session["chunk_dir"])
                except Exception as e:
                    logger.warning(f"Failed to cleanup chunk directory {session['chunk_dir']}: {e}")

        for chunk_id in expired_ids:
            del chunk_storage[chunk_id]
            logger.info(f"Cleaned up expired chunk session: {chunk_id}")


@api_bp.route("/upload-chunk/<path:chunk_id>", methods=["PATCH", "HEAD"])
@auth.login_required
def upload_chunk_data(chunk_id: str):
    """Handle chunk data uploads for specific chunk sessions."""
    if request.method == 'PATCH':
        return _handle_chunk_upload_for_id(chunk_id)
    elif request.method == 'HEAD':
        return _handle_chunk_status_for_id(chunk_id)

def _handle_chunk_upload_for_id(chunk_id: str):
    """Handle chunk data upload for a specific session."""
    try:
        with chunk_lock:
            if chunk_id not in chunk_storage:
                return "Chunk session not found", 404

            chunk_info = chunk_storage[chunk_id]

            if chunk_info["completed"]:
                response = Response(status=200)
                response.headers['Tus-Resumable'] = '1.0.0'
                response.headers['Upload-Offset'] = str(chunk_info["uploaded_size"])
                return response

        # Get upload offset
        upload_offset = request.headers.get('Upload-Offset')
        if upload_offset is None:
            return "Missing Upload-Offset header", 400

        upload_offset = int(upload_offset)

        # Read chunk data
        chunk_data = request.get_data()
        if not chunk_data:
            return "No chunk data provided", 400

        chunk_size = len(chunk_data)

        # Validate offset
        with chunk_lock:
            chunk_info = chunk_storage[chunk_id]
            if upload_offset != chunk_info["uploaded_size"]:
                return f"Invalid offset: {upload_offset} != {chunk_info['uploaded_size']}", 409

        # Save chunk to file
        chunk_filename = f"chunk_{upload_offset:010d}"
        chunk_path = os.path.join(chunk_info["chunk_dir"], chunk_filename)

        with open(chunk_path, 'wb') as f:
            f.write(chunk_data)

        # Update chunk info
        with chunk_lock:
            chunk_storage[chunk_id]["uploaded_size"] += chunk_size
            chunk_storage[chunk_id]["chunks"].append({
                "offset": upload_offset,
                "size": chunk_size,
                "path": chunk_path
            })

            # Check if upload is complete
            if chunk_storage[chunk_id]["uploaded_size"] >= chunk_storage[chunk_id]["total_size"]:
                chunk_storage[chunk_id]["completed"] = True

            logger.info(f"Chunk {chunk_id}: {chunk_storage[chunk_id]['uploaded_size']}/{chunk_storage[chunk_id]['total_size']} bytes")

        response = Response(status=204)
        response.headers['Tus-Resumable'] = '1.0.0'
        response.headers['Upload-Offset'] = str(chunk_storage[chunk_id]["uploaded_size"])
        return response

    except Exception as e:
        logger.error(f"Chunk upload for ID error: {e}")
        return jsonify({"error": str(e)}), 500

def _handle_chunk_status_for_id(chunk_id: str):
    """Handle chunk status check for a specific session."""
    try:
        with chunk_lock:
            if chunk_id not in chunk_storage:
                return "Chunk session not found", 404

            chunk_info = chunk_storage[chunk_id]

        response = Response(status=200)
        response.headers['Tus-Resumable'] = '1.0.0'
        response.headers['Upload-Offset'] = str(chunk_info["uploaded_size"])
        response.headers['Upload-Length'] = str(chunk_info["total_size"])

        if chunk_info["completed"]:
            response.headers['Upload-Complete'] = '1'

        return response

    except Exception as e:
        logger.error(f"Chunk status for ID error: {e}")
        return jsonify({"error": str(e)}), 500

@api_bp.route("/upload-chunk", methods=["POST", "HEAD", "OPTIONS"])
@auth.login_required
def upload_chunk():
    """Tus-compatible resumable upload endpoint."""
    try:
        # Handle pre-flight request
        if request.method == 'OPTIONS':
            response = Response()
            response.headers['Tus-Resumable'] = '1.0.0'
            response.headers['Tus-Version'] = '1.0.0'
            response.headers['Tus-Max-Size'] = str(20 * 1024 * 1024 * 1024)  # 20GB
            response.headers['Tus-Extension'] = 'creation,expiration'
            return response

        # Handle creation request
        if request.method == 'POST':
            return _handle_chunk_creation()

        # Handle chunk upload
        if request.method == 'PATCH':
            return _handle_chunk_upload()

        # Handle HEAD request (status check)
        if request.method == 'HEAD':
            return _handle_chunk_status()

    except Exception as e:
        logger.error(f"Chunk upload error: {e}")
        return jsonify({"error": str(e)}), 500


def _handle_chunk_creation():
    """Handle chunk upload creation."""
    try:
        cleanup_expired_chunks()

        # Get metadata from headers
        upload_length = request.headers.get('Upload-Length')
        upload_metadata = request.headers.get('Upload-Metadata', '')

        if not upload_length:
            return "Missing Upload-Length header", 400

        upload_length = int(upload_length)
        max_size = 20 * 1024 * 1024 * 1024  # 20GB
        if upload_length > max_size:
            return f"Upload too large: {upload_length} > {max_size}", 413

        # Parse metadata - tus.js sends base64-encoded values, keys are plain text
        import base64
        logger.info(f"Raw upload metadata: '{upload_metadata}'")
        metadata = {}
        if upload_metadata:
            parts = upload_metadata.split(',')
            logger.info(f"Split metadata parts: {parts}")
            for part in parts:
                part = part.strip()

                # Handle various metadata formats that tus.js might send
                if ':' in part:
                    # Format: key:value
                    key, value = part.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                elif ' ' in part and '=' not in part:
                    # Format: key value (space separated)
                    key, value = part.split(' ', 1)
                    key = key.strip()
                    value = value.strip()
                elif '=' in part:
                    # Format: key=value
                    key, value = part.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                else:
                    # Try to decode the entire part as base64 for debugging
                    try:
                        decoded_part = base64.b64decode(part).decode('utf-8')
                        logger.warning(f"Cannot parse metadata part '{part}', decoded to: '{decoded_part}'")
                        continue
                    except:
                        logger.warning(f"Skipping unparseable metadata part: '{part}'")
                        continue

                # Decode base64 values
                if value:
                    try:
                        # Remove any trailing padding that might be causing issues
                        if not value.endswith('='):
                            # Try to decode as-is first
                            decoded_value = base64.b64decode(value).decode('utf-8')
                        else:
                            # Standard base64 with padding
                            decoded_value = base64.b64decode(value).decode('utf-8')
                        metadata[key] = decoded_value
                        logger.info(f"Decoded metadata key='{key}', value='{decoded_value}'")
                    except Exception as decode_error:
                        logger.warning(f"Failed to decode metadata value '{value}' for key '{key}': {decode_error}")
                        # Fallback: use value as-is if decoding fails
                        metadata[key] = value
                else:
                    metadata[key] = ''
                    logger.info(f"Empty metadata key='{key}'")

        logger.info(f"Final parsed metadata: {metadata}")

        # Create chunk session
        chunk_id = str(uuid.uuid4())
        chunk_dir = f"{config.temp_directory}/chunks_{chunk_id}"
        os.makedirs(chunk_dir, exist_ok=True)

        chunk_info = {
            "chunk_id": chunk_id,
            "start_time": time.time(),
            "chunk_dir": chunk_dir,
            "total_size": upload_length,
            "uploaded_size": 0,
            "metadata": metadata,
            "chunks": [],
            "completed": False
        }

        with chunk_lock:
            chunk_storage[chunk_id] = chunk_info

        logger.info(f"Created chunk session: {chunk_id}, size: {upload_length}")

        # Return creation response
        response = Response(status=201)
        response.headers['Tus-Resumable'] = '1.0.0'
        response.headers['Location'] = f"/upload-chunk/{chunk_id}"
        response.headers['Upload-Offset'] = '0'
        return response

    except Exception as e:
        logger.error(f"Chunk creation error: {e}")
        return jsonify({"error": str(e)}), 500


def _handle_chunk_upload():
    """Handle chunk data upload."""
    try:
        # Extract chunk ID from URL
        chunk_id = request.path.split('/')[-1]

        with chunk_lock:
            if chunk_id not in chunk_storage:
                return "Chunk session not found", 404

            chunk_info = chunk_storage[chunk_id]

            if chunk_info["completed"]:
                response = Response(status=200)
                response.headers['Tus-Resumable'] = '1.0.0'
                response.headers['Upload-Offset'] = str(chunk_info["uploaded_size"])
                return response

        # Get upload offset
        upload_offset = request.headers.get('Upload-Offset')
        if upload_offset is None:
            return "Missing Upload-Offset header", 400

        upload_offset = int(upload_offset)

        # Read chunk data
        chunk_data = request.get_data()
        if not chunk_data:
            return "No chunk data provided", 400

        chunk_size = len(chunk_data)

        # Validate offset
        with chunk_lock:
            chunk_info = chunk_storage[chunk_id]
            if upload_offset != chunk_info["uploaded_size"]:
                return f"Invalid offset: {upload_offset} != {chunk_info['uploaded_size']}", 409

        # Save chunk to file
        chunk_filename = f"chunk_{upload_offset:010d}"
        chunk_path = os.path.join(chunk_info["chunk_dir"], chunk_filename)

        with open(chunk_path, 'wb') as f:
            f.write(chunk_data)

        # Update chunk info
        with chunk_lock:
            chunk_storage[chunk_id]["uploaded_size"] += chunk_size
            chunk_storage[chunk_id]["chunks"].append({
                "offset": upload_offset,
                "size": chunk_size,
                "path": chunk_path
            })

            # Check if upload is complete
            if chunk_storage[chunk_id]["uploaded_size"] >= chunk_storage[chunk_id]["total_size"]:
                chunk_storage[chunk_id]["completed"] = True

            logger.info(f"Chunk {chunk_id}: {chunk_storage[chunk_id]['uploaded_size']}/{chunk_storage[chunk_id]['total_size']} bytes")

        response = Response(status=204)
        response.headers['Tus-Resumable'] = '1.0.0'
        response.headers['Upload-Offset'] = str(chunk_storage[chunk_id]["uploaded_size"])
        return response

    except Exception as e:
        logger.error(f"Chunk upload error: {e}")
        return jsonify({"error": str(e)}), 500


def _handle_chunk_status():
    """Handle chunk status check."""
    try:
        chunk_id = request.path.split('/')[-1]

        with chunk_lock:
            if chunk_id not in chunk_storage:
                return "Chunk session not found", 404

            chunk_info = chunk_storage[chunk_id]

        response = Response(status=200)
        response.headers['Tus-Resumable'] = '1.0.0'
        response.headers['Upload-Offset'] = str(chunk_info["uploaded_size"])
        response.headers['Upload-Length'] = str(chunk_info["total_size"])

        if chunk_info["completed"]:
            response.headers['Upload-Complete'] = '1'

        return response

    except Exception as e:
        logger.error(f"Chunk status error: {e}")
        return jsonify({"error": str(e)}), 500


@api_bp.route("/upload-chunk/<chunk_id>", methods=["GET"])
@auth.login_required
def get_uploaded_file(chunk_id: str):
    """Get the assembled uploaded file."""
    try:
        with chunk_lock:
            if chunk_id not in chunk_storage:
                return jsonify({"error": "Chunk session not found"}), 404

            chunk_info = chunk_storage[chunk_id]

            if not chunk_info["completed"]:
                return jsonify({"error": "Upload not completed"}), 400

        # Assemble file if not already assembled
        assembled_path = _assemble_chunks(chunk_id)
        if not assembled_path:
            return jsonify({"error": "Failed to assemble file"}), 500

        # Generate final URL
        filename = chunk_info["metadata"].get("filename", "unknown")
        logger.info(f"Chunk metadata: {chunk_info['metadata']}")
        logger.info(f"Resolved filename: {filename}")
        file_url = f"/chunk-assembly/{chunk_id}/{filename}"

        return jsonify({
            "url": file_url,
            "filename": filename,
            "size": chunk_info["total_size"],
            "type": chunk_info["metadata"].get("filetype", "application/octet-stream")
        })

    except Exception as e:
        logger.error(f"Get uploaded file error: {e}")
        return jsonify({"error": str(e)}), 500


def _assemble_chunks(chunk_id: str) -> Optional[str]:
    """Assemble all chunks into a complete file."""
    try:
        with chunk_lock:
            if chunk_id not in chunk_storage:
                return None

            chunk_info = chunk_storage[chunk_id]
            chunk_dir = chunk_info["chunk_dir"]
            metadata = chunk_info["metadata"]

            # Check if already assembled
            assembled_filename = f"assembled_{chunk_id}_{metadata.get('filename', 'unknown')}"
            assembled_path = os.path.join(config.temp_directory, assembled_filename)

            if os.path.exists(assembled_path):
                return assembled_path

            # Sort chunks by offset
            chunks = sorted(chunk_info["chunks"], key=lambda x: x["offset"])

            # Assemble file
            with open(assembled_path, 'wb') as assembled_file:
                for chunk in chunks:
                    chunk_path = chunk["path"]
                    if os.path.exists(chunk_path):
                        with open(chunk_path, 'rb') as chunk_file:
                            assembled_file.write(chunk_file.read())

            # Verify assembled file size
            if os.path.getsize(assembled_path) != chunk_info["total_size"]:
                logger.error(f"Assembled file size mismatch: {os.path.getsize(assembled_path)} != {chunk_info['total_size']}")
                os.remove(assembled_path)
                return None

            logger.info(f"Assembled file: {assembled_path} ({chunk_info['total_size']} bytes)")
            return assembled_path

    except Exception as e:
        logger.error(f"Chunk assembly error: {e}")
        return None


@api_bp.route("/chunk-assembly/<chunk_id>/<filename>")
@auth.login_required
def serve_assembled_file(chunk_id: str, filename: str):
    """Serve the assembled uploaded file."""
    try:
        with chunk_lock:
            if chunk_id not in chunk_storage:
                return jsonify({"error": "Chunk session not found"}), 404

            chunk_info = chunk_storage[chunk_id]

        assembled_path = _assemble_chunks(chunk_id)
        if not assembled_path:
            return jsonify({"error": "File assembly failed"}), 500

        # Serve the file
        from flask import send_file
        return send_file(
            assembled_path,
            as_attachment=False,
            download_name=filename,
            mimetype=chunk_info["metadata"].get("filetype", "application/octet-stream")
        )

    except Exception as e:
        logger.error(f"Serve assembled file error: {e}")
        return jsonify({"error": str(e)}), 500


@api_bp.route("/start-swap", methods=["POST"])
@auth.login_required
def start_swap():
    """Start face swap processing with uploaded file URLs."""
    try:
        if batch_processor.get_progress() and not batch_processor.get_progress().is_complete:
            return jsonify({
                "error": "Processing is already in progress. Please wait for the current batch to complete."
            }), 409

        cleanup_expired_chunks()

        data = request.get_json()
        if not data or 'files' not in data:
            return jsonify({"error": "Missing files data"}), 400

        files = data['files']
        if not files or len(files) == 0:
            return jsonify({"error": "No files provided"}), 400

        # Separate source and target files
        source_files = [f for f in files if f['type'] == 'source']
        target_files = [f for f in files if f['type'] == 'target']

        if len(source_files) != 1:
            return jsonify({"error": "Exactly one source file required"}), 400

        if len(target_files) == 0:
            return jsonify({"error": "At least one target file required"}), 400

        source_url = source_files[0]['url']
        target_urls = [f['url'] for f in target_files]
        target_filenames = [f['filename'] for f in target_files]

        logger.info(f"Starting swap processing for {len(target_files)} files")

        # Create upload session for progress tracking
        total_files = 1 + len(target_files)
        estimated_total_size_mb = 2000.0
        upload_id = create_upload_session(total_files, estimated_total_size_mb)

        # Ensure source file is assembled and get its path
        source_chunk_id = source_url.split('/')[-2]  # /upload-chunk/{chunk_id} -> chunk_id
        source_assembled_path = _assemble_chunks(source_chunk_id)
        if not source_assembled_path or not os.path.exists(source_assembled_path):
            logger.error(f"Source file not assembled: chunk_id={source_chunk_id}")
            return jsonify({"error": "Source file not assembled properly"}), 400

        # Ensure target files are assembled and get their paths
        target_assembled_paths = []
        for i, target_url in enumerate(target_urls):
            target_chunk_id = target_url.split('/')[-2]
            target_assembled_path = _assemble_chunks(target_chunk_id)
            if not target_assembled_path or not os.path.exists(target_assembled_path):
                logger.error(f"Target file {i} not assembled: chunk_id={target_chunk_id}")
                return jsonify({"error": f"Target file {i} not assembled properly"}), 400
            target_assembled_paths.append(target_assembled_path)

        # Update progress after validation
        with progress_lock:
            if upload_id in upload_progress:
                upload_progress[upload_id]["files_uploaded"] = total_files
                upload_progress[upload_id]["status"] = "completed"

        # Validate source face
        logger.info("Validating source image...")
        source_img = cv2.imread(source_assembled_path)
        if source_img is None:
            file_processor.cleanup_temp_files([source_assembled_path] + target_assembled_paths)
            with progress_lock:
                if upload_id in upload_progress:
                    upload_progress[upload_id]["status"] = "error"
            return jsonify({"error": "Cannot read source image"}), 400

        source_faces = face_services.detector.detect_faces(source_img)
        if not source_faces.success:
            file_processor.cleanup_temp_files([source_assembled_path] + target_assembled_paths)
            with progress_lock:
                if upload_id in upload_progress:
                    upload_progress[upload_id]["status"] = "error"
            return jsonify({"error": source_faces.error_message or "No source face detected"}), 400

        logger.info("Starting background processing...")
        success = batch_processor.start_background_processing(source_assembled_path, target_assembled_paths, target_filenames)

        if not success:
            file_processor.cleanup_temp_files([source_assembled_path] + target_assembled_paths)
            with progress_lock:
                if upload_id in upload_progress:
                    upload_progress[upload_id]["status"] = "error"
            return jsonify({"error": "Cannot start processing"}), 500

        logger.info(f"Processing started successfully for {len(target_files)} files")
        return jsonify({
            "message": "Processing started",
            "total_files": len(target_files),
            "upload_id": upload_id
        }), 202

    except Exception as e:
        logger.error(f"Start swap error: {e}")
        return jsonify({"error": str(e)}), 500


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
    """Handle face swap requests with optimized uploading."""
    try:
        # Check if already processing
        if batch_processor.get_progress() and not batch_processor.get_progress().is_complete:
            return jsonify({
                "error": "Processing is already in progress. Please wait for the current batch to complete.",
                "retry_after": 30  # Suggest retry after 30 seconds
            }), 409

        cleanup_expired_uploads()  # Clean up old upload sessions

        src_file = request.files.get("source")
        tgt_files = request.files.getlist("targets")

        # Validate request
        is_valid, error_msg = _validate_request(src_file, tgt_files)
        if not is_valid:
            return jsonify({"error": error_msg}), 400

        # Create upload session for progress tracking
        total_files = 1 + len(tgt_files)  # source + targets
        estimated_total_size_mb = 2000.0  # Initial estimate, will be updated during upload
        upload_id = create_upload_session(total_files, estimated_total_size_mb)

        # Save files with progress tracking
        logger.info("Starting optimized file upload processing...")

        # Save source file first
        src_path = file_processor.save_uploaded_file(src_file, src_file.filename)
        logger.info(f"Source file saved: {src_path}")

        # Update progress after source file
        with progress_lock:
            if upload_id in upload_progress:
                upload_progress[upload_id]["files_uploaded"] = 1

        # Save target files with progress tracking
        target_files, target_filenames = _extract_file_list(
            tgt_files,
            upload_id=upload_id,
            progress_lock=progress_lock
        )
        logger.info(f"Target files processed: {len(target_files)} files")

        # Update final progress
        with progress_lock:
            if upload_id in upload_progress:
                upload_progress[upload_id]["files_uploaded"] = total_files
                upload_progress[upload_id]["status"] = "completed"

        if not target_files:
            file_processor.cleanup_temp_files([src_path])
            with progress_lock:
                if upload_id in upload_progress:
                    upload_progress[upload_id]["status"] = "error"
            return jsonify({"error": "Cannot save any target files"}), 400

        # Validate source image
        logger.info("Validating source image...")
        source_img = cv2.imread(src_path)
        if source_img is None:
            file_processor.cleanup_temp_files([src_path] + target_files)
            with progress_lock:
                if upload_id in upload_progress:
                    upload_progress[upload_id]["status"] = "error"
            return jsonify({"error": "Cannot read source image"}), 400

        # Validate source face
        logger.info("Detecting faces in source image...")
        source_faces = face_services.detector.detect_faces(source_img)
        if not source_faces.success:
            file_processor.cleanup_temp_files([src_path] + target_files)
            with progress_lock:
                if upload_id in upload_progress:
                    upload_progress[upload_id]["status"] = "error"
            return jsonify({"error": source_faces.error_message or "No source face detected"}), 400

        logger.info("Starting background processing...")
        # Start background processing
        success = batch_processor.start_background_processing(
            src_path, target_files, target_filenames
        )

        if not success:
            file_processor.cleanup_temp_files([src_path] + target_files)
            with progress_lock:
                if upload_id in upload_progress:
                    upload_progress[upload_id]["status"] = "error"
            return jsonify({"error": "Cannot start processing"}), 500

        logger.info(f"Processing started successfully for {len(target_files)} files")
        return jsonify({
            "message": "Processing started",
            "total_files": len(target_files),
            "upload_id": upload_id
        }), 202

    except ValueError as e:
        # Handle file size limit exceeded
        if "exceeds maximum limit" in str(e):
            logger.warning(f"File size limit exceeded: {e}")
            return jsonify({"error": str(e)}), 413
        # Handle other validation errors
        logger.warning(f"Validation error: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({"error": "Internal server error occurred"}), 500


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


@api_bp.route("/swapface-url", methods=["GET"])
def swapface_url_api():
    """Handle face swap requests with URL inputs."""
    try:
        # Get URL parameters from request
        face_url = request.args.get("face")
        target_url = request.args.get("target")

        if not face_url or not target_url:
            return jsonify({"error": "Both 'face' and 'target' URL parameters are required"}), 400

        logger.info(f"Processing face swap from URLs: face={face_url[:50]}..., target={target_url[:50]}...")

        # Global caches
        url_cache = {}  # url -> {"file_path": str, "timestamp": float, "filename": str}
        swap_cache = {}  # pair_key -> {"output_path": str, "timestamp": float}
        cache_lock = threading.Lock()

        def get_cache_key(url: str) -> str:
            """Generate a cache key for URL."""
            import hashlib
            return hashlib.md5(url.encode()).hexdigest()

        def get_swap_cache_key(face_url: str, target_url: str) -> str:
            """Generate a cache key for face swap pair."""
            import hashlib
            pair_string = f"{face_url}|{target_url}"
            return hashlib.md5(pair_string.encode()).hexdigest()

        def cleanup_url_cache(max_age_hours: int = 24):
            """Clean up old cache entries."""
            current_time = time.time()
            expired_keys = []

            with cache_lock:
                # Clean up URL cache
                for key, cache_info in url_cache.items():
                    if current_time - cache_info["timestamp"] > max_age_hours * 3600:
                        expired_keys.append(key)
                        # Remove the cached file
                        try:
                            if os.path.exists(cache_info["file_path"]):
                                os.remove(cache_info["file_path"])
                                logger.info(f"Removed expired cached file: {cache_info['file_path']}")
                        except Exception as e:
                            logger.warning(f"Failed to remove expired cache file: {e}")

                # Remove expired entries from URL cache
                for key in expired_keys:
                    del url_cache[key]

                expired_keys = []

                # Clean up swap cache
                for key, cache_info in swap_cache.items():
                    if current_time - cache_info["timestamp"] > max_age_hours * 3600:
                        expired_keys.append(key)
                        # Remove the cached swap result
                        try:
                            if os.path.exists(cache_info["output_path"]):
                                os.remove(cache_info["output_path"])
                                logger.info(f"Removed expired cached swap result: {cache_info['output_path']}")
                        except Exception as e:
                            logger.warning(f"Failed to remove expired swap cache file: {e}")

                # Remove expired entries from swap cache
                for key in expired_keys:
                    del swap_cache[key]

                logger.info(f"Cleaned up {len(expired_keys)} expired swap cache entries")

        # Download images from URLs with caching
        def download_image(url: str, filename: str, max_retries: int = 3, base_delay: float = 1.0) -> str:
            """Download image from URL and save to temp directory with retry logic and caching."""
            import random
            import hashlib

            cache_key = get_cache_key(url)

            # Check cache first
            with cache_lock:
                if cache_key in url_cache:
                    cache_info = url_cache[cache_key]
                    # Check if cached file still exists and is not too old (24 hours)
                    if (os.path.exists(cache_info["file_path"]) and
                        time.time() - cache_info["timestamp"] < 24 * 3600):

                        # Copy cached file to new location for this request
                        new_uuid = str(uuid.uuid4())
                        new_path = f"{config.temp_directory}/{new_uuid}_{cache_info['filename']}"

                        try:
                            import shutil
                            shutil.copy2(cache_info["file_path"], new_path)
                            logger.info(f"Using cached image for {url}: {new_path}")
                            return new_path
                        except Exception as e:
                            logger.warning(f"Failed to copy from cache, downloading fresh: {e}")
                            # Fall through to download

            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    logger.info(f"Attempting to download {url} (attempt {attempt + 1}/{max_retries + 1})")

                    response = requests.get(url, timeout=30, stream=True, verify=False)
                    response.raise_for_status()

                    # Check content type
                    content_type = response.headers.get('content-type', '').lower()
                    if not content_type.startswith('image/'):
                        raise ValueError(f"URL does not point to an image: {content_type}")

                    # Get file size
                    content_length = response.headers.get('content-length')
                    if content_length:
                        size_mb = int(content_length) / (1024 * 1024)
                        if size_mb > 50:  # 50MB limit for URL downloads
                            raise ValueError(f"Image too large: {size_mb:.1f}MB")

                    file_uuid = str(uuid.uuid4())
                    file_path = f"{config.temp_directory}/{file_uuid}_{filename}"

                    with open(file_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)

                    # Verify the downloaded file is a valid image
                    img = cv2.imread(file_path)
                    if img is None:
                        os.remove(file_path)
                        raise ValueError("Downloaded file is not a valid image")

                    # Cache the successful download
                    cache_file_uuid = str(uuid.uuid4())
                    cache_file_path = f"{config.temp_directory}/cache_{cache_file_uuid}_{filename}"
                    try:
                        import shutil
                        shutil.copy2(file_path, cache_file_path)
                        with cache_lock:
                            url_cache[cache_key] = {
                                "file_path": cache_file_path,
                                "timestamp": time.time(),
                                "filename": filename
                            }
                        logger.info(f"Cached downloaded image: {cache_file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to cache file: {e}")

                    logger.info(f"Downloaded and saved image: {file_path}")
                    return file_path

                except requests.exceptions.HTTPError as e:
                    # Check if it's a 429 error (rate limit)
                    if response.status_code == 429:
                        if attempt < max_retries:
                            # Exponential backoff with jitter
                            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                            logger.warning(".2f")
                            time.sleep(delay)
                            continue
                        else:
                            logger.error(f"Failed to download after {max_retries + 1} attempts due to rate limiting: {url}")
                            raise ValueError(f"Failed to download image: {response.status_code} Client Error for url: {url}")
                    else:
                        # For other HTTP errors, don't retry
                        raise
                except (requests.exceptions.RequestException, ValueError) as e:
                    # For network errors (timeouts, connection errors) or validation errors, retry
                    last_exception = e
                    if attempt < max_retries:
                        # Exponential backoff with jitter for network errors
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                        logger.warning(".2f")
                        time.sleep(delay)
                    else:
                        logger.error(f"Failed to download after {max_retries + 1} attempts: {url}, error: {e}")
                        if isinstance(e, requests.exceptions.HTTPError):
                            raise
                        else:
                            raise ValueError(f"Failed to download image: {str(e)}")
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                        logger.warning(".2f")
                        time.sleep(delay)
                    else:
                        logger.error(f"Failed to download after {max_retries + 1} attempts: {url}, error: {e}")
                        raise ValueError(f"Failed to download image: {str(e)}")

            # This should never be reached, but just in case
            raise ValueError(f"Failed to download image after all retries: {str(last_exception)}")

        # Check if swap result is cached
        swap_cache_key = get_swap_cache_key(face_url, target_url)
        with cache_lock:
            if swap_cache_key in swap_cache:
                cache_info = swap_cache[swap_cache_key]
                # Check if cached result still exists and is not too old (24 hours)
                if (os.path.exists(cache_info["output_path"]) and
                    time.time() - cache_info["timestamp"] < 24 * 3600):

                    logger.info(f"Using cached swap result for pair {face_url[:30]}... + {target_url[:30]}...: {cache_info['output_path']}")
                    return send_file(
                        cache_info["output_path"],
                        mimetype='image/jpeg',
                        as_attachment=False
                    )

        # Download face image
        face_filename = "face_" + face_url.split('/')[-1].split('?')[0] or "face.jpg"
        face_path = download_image(face_url, face_filename)

        # Download target image
        target_filename = "target_" + target_url.split('/')[-1].split('?')[0] or "target.jpg"
        target_path = download_image(target_url, target_filename)

        try:
            # Load and validate images
            face_img = cv2.imread(face_path)
            target_img = cv2.imread(target_path)

            if face_img is None:
                return jsonify({"error": "Cannot read face image"}), 400

            if target_img is None:
                return jsonify({"error": "Cannot read target image"}), 400

            # Detect faces
            face_result = face_services.detector.detect_faces(face_img)
            target_result = face_services.detector.detect_faces(target_img)

            # If no face detected in either image, return target image
            if not face_result.success or not target_result.success:
                logger.warning(f"Face detection failed - face_result: {face_result.success}, target_result: {target_result.success}. Returning target image.")
                # Return target image directly
                return send_file(
                    target_path,
                    mimetype='image/jpeg',
                    as_attachment=False
                )

            # Perform face swap
            source_face = face_result.faces[0]
            target_face = target_result.faces[0]

            swapped_img = face_services.swapper.swap_faces(source_face, target_img, target_face)

            if swapped_img is None:
                logger.warning("Face swap failed, returning target image")
                # Return target image if swap fails
                return send_file(
                    target_path,
                    mimetype='image/jpeg',
                    as_attachment=False
                )

            # Save result
            output_uuid = str(uuid.uuid4())
            output_path = f"{config.output_directory}/{output_uuid}.jpg"
            success = cv2.imwrite(output_path, swapped_img)

            if not success:
                logger.warning("Failed to save result image, returning target image")
                # Return target image if save fails
                return send_file(
                    target_path,
                    mimetype='image/jpeg',
                    as_attachment=False
                )

            # Cache the swap result
            with cache_lock:
                swap_cache[swap_cache_key] = {
                    "output_path": output_path,
                    "timestamp": time.time()
                }
            logger.info(f"Cached swap result for pair {face_url[:30]}... + {target_url[:30]}...: {output_path}")

            logger.info("Face swap completed successfully")

            # Return image data directly for embedding in img src
            return send_file(
                output_path,
                mimetype='image/jpeg',
                as_attachment=False
            )

        finally:
            # Cleanup temp files
            try:
                if os.path.exists(face_path):
                    os.remove(face_path)
                if os.path.exists(target_path):
                    os.remove(target_path)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp files: {e}")

    except requests.exceptions.RequestException as e:
        logger.error(f"Network error during URL processing: {e}")
        return jsonify({"error": f"Failed to download image: {str(e)}"}), 400
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Unexpected error in swapface-url: {e}")
        return jsonify({"error": "Internal server error occurred"}), 500
