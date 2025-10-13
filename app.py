from flask import Flask, request, jsonify, render_template, send_file
import cv2
import numpy as np
import os
import uuid
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
import tempfile
import threading
import time

# Windows GTX 1650 GPU optimization
os.environ["OMP_NUM_THREADS"] = "4"  # GTX 1650 có 4GB VRAM, limit threads
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Enable CUDA for GTX 1650

app = Flask(__name__, template_folder="templates")
os.makedirs("output", exist_ok=True)

# Global status tracking
processing_status = {
    "is_processing": False,
    "current_file": None,
    "queue": [],
    "completed": 0,
    "total": 0,
    "start_time": None,
    "results": [],
    "current_video_progress": 0,
    "current_video_total_frames": 0,
    "current_video_processed_frames": 0
}

print("🚀 Loading face detection and swap models with GTX 1650 GPU optimizations...")
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0, det_size=(640, 640))  # GPU for GTX 1650
swapper = get_model("models/inswapper_128.onnx", download=False)
print("✅ Models loaded successfully!")

def detect_faces(img):
    faces = face_app.get(img)
    if not faces:
        print("⚠️ No face detected in image.")
    return faces

def swap_face(source_img, target_img):
    source_faces = detect_faces(source_img)
    target_faces = detect_faces(target_img)
    if not source_faces or not target_faces:
        print("❌ No faces found for swapping.")
        return target_img

    source_face = source_faces[0]
    target_face = target_faces[0]

    try:
        # Optimized: direct swap without scaling for speed
        swapped = swapper.get(target_img, target_face, source_face, paste_back=True)

        return swapped

    except Exception as e:
        print(f"⚠️ Swap error: {e}")
        return target_img

def swap_face_cached(source_face_cached, target_img):
    """Swap face with cached source_face (CPU optimization)"""
    target_faces = detect_faces(target_img)
    if not target_faces:
        print("❌ No target face found for swapping.")
        return target_img

    target_face = target_faces[0]

    try:
        # Optimized: direct swap without scaling without re-detecting source
        swapped = swapper.get(target_img, target_face, source_face_cached, paste_back=True)

        return swapped

    except Exception as e:
        print(f"⚠️ Swap error: {e}")
        return target_img

def swap_video_cached(source_face_cached, video_path, output_path, source_img_fake):
    """Swap face in video with parallel frame processing and better memory management"""
    global processing_status

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Cannot open video")
        return False

    # Get video info
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Update video progress tracking
    processing_status["current_video_total_frames"] = total_frames
    processing_status["current_video_processed_frames"] = 0

    # Use H.264 codec for better compression and reasonable file size
    try:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
    except:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frames = []
    frame_count = 0
    batch_size = 8  # Process 8 frames at once on M1

    print(f"🎬 Processing video with parallel frames: total frames ~{total_frames}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame)
        frame_count += 1

        # Process batch frames when enough or at end of video
        if len(frames) >= batch_size or frame_count % batch_size == 0:
            print(f"🎬 Processing batch: {len(frames)} frames")

            # Parallel processing frames in batch
            swapped_batch = []
            for frame in frames:
                try:
                    swapped_frame = swap_face_cached(source_face_cached, frame)
                    swapped_batch.append(swapped_frame)
                except Exception as e:
                    print(f"⚠️ Batch frame error: {e}")
                    swapped_batch.append(frame)

            # Write batch to video
            for swapped_frame in swapped_batch:
                out.write(swapped_frame)
                processing_status["current_video_processed_frames"] += 1

            frames = []  # Reset batch

    # Process remaining frames
    if frames:
        print(f"🎬 Processing remaining {len(frames)} frames")
        for frame in frames:
            try:
                swapped_frame = swap_face_cached(source_face_cached, frame)
                out.write(swapped_frame)
                processing_status["current_video_processed_frames"] += 1
            except Exception as e:
                print(f"⚠️ Final frame error: {e}")
                out.write(frame)

    cap.release()
    out.release()

    # Reset video progress
    processing_status["current_video_progress"] = 0
    processing_status["current_video_total_frames"] = 0
    processing_status["current_video_processed_frames"] = 0

    print(f"✅ Video completed: {output_path}")
    return True

@app.route("/")
def index():
    return render_template("index.html")


def process_single_target_cached(source_face_cached, target_file_path, target_filename, source_shape):
    """Process a single target file (image or video) with cached source_face"""

    # Check if image
    target_img = cv2.imread(target_file_path)
    if target_img is not None:
        # Process image with cached source_face
        print(f"📷 Processing image: {target_filename}")
        result_img = swap_face_cached(source_face_cached, target_img)
        target_uuid = str(uuid.uuid4())
        out_path = f"output/{target_uuid}.jpg"
        cv2.imwrite(out_path, result_img)
        return {"result": f"/view/{out_path}", "type": "image", "original_name": target_filename}
    else:
        # Check if video
        if target_filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            print(f"🎬 Processing video: {target_filename}")
            target_uuid = str(uuid.uuid4())
            out_path = f"output/{target_uuid}.mp4"
            # Create fake source_img for video processing
            source_img_fake = np.zeros(source_shape, dtype=np.uint8)
            if swap_video_cached(source_face_cached, target_file_path, out_path, source_img_fake):
                return {"result": f"/view/{out_path}", "type": "video", "original_name": target_filename}
            else:
                return {"error": f"Cannot process video: {target_filename}"}
        else:
            return {"error": f"Unsupported file: {target_filename}"}

def process_files_background(source_face_cached, temp_file_paths, original_names, source_img_shape, src_path):
    """Background processing function that processes files sequentially"""
    global processing_status

    processing_status["queue"] = original_names.copy()
    processing_status["is_processing"] = True
    processing_status["start_time"] = time.time()
    processing_status["results"] = []
    processing_status["completed"] = 0
    processing_status["total"] = len(temp_file_paths)

    print(f"🚀 Starting sequential processing for {len(temp_file_paths)} targets")

    for i, (temp_path, original_name) in enumerate(zip(temp_file_paths, original_names)):
        processing_status["current_file"] = original_name
        print(f"📂 Processing {i+1}/{len(temp_file_paths)}: {original_name}")

        result = process_single_target_cached(source_face_cached, temp_path, original_name, source_img_shape)
        processing_status["results"].append(result)
        processing_status["completed"] += 1

        print(f"✅ Completed: {result.get('original_name', 'Unknown')}")

    # Processing finished
    processing_status["current_file"] = None
    processing_status["is_processing"] = False
    processing_status["queue"] = []

    # Clean up temp files
    try:
        os.remove(src_path)
        for temp_path in temp_file_paths:
            try:
                os.remove(temp_path)
            except:
                pass
    except Exception as e:
        print(f"⚠️ Cannot remove temp files: {e}")

    print("🏁 All processing completed!")

@app.route("/swapface", methods=["POST"])
def swapface_api():
    global processing_status

    if processing_status["is_processing"]:
        return jsonify({"error": "Processing is already in progress"}), 409

    src_file = request.files.get("source")
    tgt_files = request.files.getlist("targets")  # Get multiple targets

    if not src_file or not tgt_files:
        return jsonify({"error": "Missing source file or targets"}), 400

    # Save source file
    src_uuid = str(uuid.uuid4())
    src_path = f"/tmp/{src_uuid}_{src_file.filename}"
    src_file.save(src_path)

    source_img = cv2.imread(src_path)
    if source_img is None:
        return jsonify({"error": "Cannot read source image"}), 400

    # Cache source faces to reuse (CPU optimization)
    source_faces_cached = detect_faces(source_img)
    if not source_faces_cached:
        return jsonify({"error": "No source face detected"}), 400

    # Save target files to temp paths
    temp_file_paths = []
    original_names = []

    for tgt_file in tgt_files:
        tgt_uuid = str(uuid.uuid4())
        tgt_path = f"/tmp/{tgt_uuid}_{tgt_file.filename}"
        try:
            tgt_file.save(tgt_path)
            temp_file_paths.append(tgt_path)
            original_names.append(tgt_file.filename)
        except Exception as e:
            print(f"❌ Cannot save temp file {tgt_file.filename}: {e}")

    if not temp_file_paths:
        return jsonify({"error": "Cannot save any target files"}), 400

    # Start background processing
    thread = threading.Thread(
        target=process_files_background,
        args=(source_faces_cached[0], temp_file_paths, original_names, source_img.shape, src_path)
    )
    thread.daemon = True
    thread.start()

    return jsonify({"message": "Processing started", "total_files": len(temp_file_paths)}), 202

@app.route("/status", methods=["GET"])
def get_status():
    """Get current processing status"""
    current_time = time.time()
    start_time = processing_status.get("start_time", None)
    elapsed = current_time - start_time if start_time else 0

    status_info = processing_status.copy()

    # Calculate percentage and speed
    status_info["progress_percentage"] = (
        (status_info["completed"] / status_info["total"] * 100)
        if status_info["total"] > 0 else 0
    )

    # Calculate processing speed (files per second, but avoid division by zero)
    if elapsed > 0:
        status_info["speed"] = status_info["completed"] / elapsed
    else:
        status_info["speed"] = 0

    # Estimate time remaining
    if status_info["speed"] > 0 and status_info["completed"] < status_info["total"]:
        remaining_files = status_info["total"] - status_info["completed"]
        status_info["eta_seconds"] = remaining_files / status_info["speed"]
    else:
        status_info["eta_seconds"] = 0

    # Add video progress if current file is video
    status_info["video_progress_percentage"] = (
        (status_info["current_video_processed_frames"] / status_info["current_video_total_frames"] * 100)
        if status_info["current_video_total_frames"] > 0 else 0
    )

    return jsonify(status_info)


@app.route("/view/<path:filename>")
def view_file(filename):
    mimetype = "image/jpeg" if filename.lower().endswith((".jpg", ".jpeg", ".png")) else "video/mp4"
    return send_file(filename, mimetype=mimetype)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
