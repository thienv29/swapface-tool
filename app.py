from flask import Flask, request, jsonify, render_template, send_file
import cv2
import numpy as np
import os
import uuid
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
import tempfile

# Windows GTX 1650 GPU optimization
os.environ["OMP_NUM_THREADS"] = "4"  # GTX 1650 có 4GB VRAM, limit threads
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Enable CUDA for GTX 1650

app = Flask(__name__, template_folder="templates")
os.makedirs("output", exist_ok=True)

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
    """Swap face in video with parallel frame processing"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Cannot open video")
        return False

    # Get video info
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frames = []
    frame_count = 0
    batch_size = 8  # Process 8 frames at once on M1

    print(f"🎬 Processing video with parallel frames: total frames ~{int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")

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

            frames = []  # Reset batch

    # Process remaining frames
    if frames:
        print(f"🎬 Processing remaining {len(frames)} frames")
        for frame in frames:
            try:
                swapped_frame = swap_face_cached(source_face_cached, frame)
                out.write(swapped_frame)
            except Exception as e:
                print(f"⚠️ Final frame error: {e}")
                out.write(frame)

    cap.release()
    out.release()
    print(f"✅ Video completed: {output_path}")
    return True

@app.route("/")
def index():
    return render_template("index.html")


import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import cpu_count

# Số workers tối đa cho processing parallel GTX 1650 (4GB VRAM optimization)
MAX_WORKERS = min(8, cpu_count())  # Limit workers cho GTX 1650

def process_single_target_cached(source_face_cached, target_file, target_filename, source_shape):
    """Process a single target file (image or video) with cached source_face"""
    target_uuid = str(uuid.uuid4())
    tgt_path = f"/tmp/{target_uuid}_{target_filename}"
    try:
        target_file.save(tgt_path)
    except Exception as e:
        print(f"❌ Temp file save error: {e}")
        return {"error": f"Cannot save file: {target_filename}"}

    # Check if image
    target_img = cv2.imread(tgt_path)
    if target_img is not None:
        # Process image with cached source_face
        print(f"📷 Processing image: {target_filename}")
        result_img = swap_face_cached(source_face_cached, target_img)
        out_path = f"output/{target_uuid}.jpg"
        cv2.imwrite(out_path, result_img)
        return {"result": f"/view/{out_path}", "type": "image", "original_name": target_filename}
    else:
        # Check if video
        if target_filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            print(f"🎬 Processing video: {target_filename}")
            out_path = f"output/{target_uuid}.mp4"
            # Create fake source_img for video processing
            source_img_fake = np.zeros(source_shape, dtype=np.uint8)
            if swap_video_cached(source_face_cached, tgt_path, out_path, source_img_fake):
                return {"result": f"/view/{out_path}", "type": "video", "original_name": target_filename}
            else:
                return {"error": f"Cannot process video: {target_filename}"}
        else:
            return {"error": f"Unsupported file: {target_filename}"}

@app.route("/swapface", methods=["POST"])
def swapface_api():
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

    print(f"🚀 Starting batch processing for {len(tgt_files)} targets")

    results = []

    # Use ThreadPoolExecutor với số workers tối ưu cho M1
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks với source_faces_cached
        future_to_file = {
            executor.submit(process_single_target_cached, source_faces_cached[0], tgt_file, tgt_file.filename, source_img.shape): tgt_file.filename
            for tgt_file in tgt_files
        }

        # Collect results as they complete
        for future in future_to_file:
            try:
                result = future.result()
                results.append(result)
                print(f"✅ Completed: {result.get('original_name', 'Unknown')}")
            except Exception as e:
                print(f"❌ Error processing file: {e}")
                results.append({"error": f"Processing error: {e}"})

    # Clean up temp source file
    try:
        os.remove(src_path)
    except Exception as e:
        print(f"⚠️ Không thể xóa temp source file: {e}")

    return jsonify({"results": results})


@app.route("/view/<path:filename>")
def view_file(filename):
    mimetype = "image/jpeg" if filename.lower().endswith((".jpg", ".jpeg", ".png")) else "video/mp4"
    return send_file(filename, mimetype=mimetype)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
