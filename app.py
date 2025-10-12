from flask import Flask, request, jsonify, render_template, send_file
import cv2
import numpy as np
import os
import uuid
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
import tempfile

app = Flask(__name__, template_folder="templates")
os.makedirs("output", exist_ok=True)

print("🚀 Loading face detection and swap models...")
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=-1, det_size=(640, 640))  # CPU cho M1
swapper = get_model("models/inswapper_128.onnx", download=False)
print("✅ Models loaded successfully!")

def detect_faces(img):
    faces = face_app.get(img)
    if not faces:
        print("⚠️ Không phát hiện khuôn mặt nào trong ảnh.")
    return faces

def swap_face(source_img, target_img):
    source_faces = detect_faces(source_img)
    target_faces = detect_faces(target_img)
    if not source_faces or not target_faces:
        print("❌ Không tìm thấy khuôn mặt để swap.")
        return target_img

    source_face = source_faces[0]
    target_face = target_faces[0]

    try:
        (x1, y1, x2, y2) = target_face.bbox.astype(int)
        face_w, face_h = x2 - x1, y2 - y1
        face_size = max(face_w, face_h)

        # 🚀 Nếu khuôn mặt nhỏ hơn 200px, scale toàn bộ ảnh lên trước
        scale_up = 2.0 if face_size < 150 else 1.0
        if scale_up > 1.0:
            target_img = cv2.resize(target_img, None, fx=scale_up, fy=scale_up, interpolation=cv2.INTER_CUBIC)
            source_img = cv2.resize(source_img, None, fx=scale_up, fy=scale_up, interpolation=cv2.INTER_CUBIC)
            # Cập nhật lại khuôn mặt sau khi resize
            target_face = face_app.get(target_img)[0]
            source_face = face_app.get(source_img)[0]

        swapped = swapper.get(target_img, target_face, source_face, paste_back=True)

        # 🔙 Resize ngược về kích thước gốc nếu có phóng to
        if scale_up > 1.0:
            swapped = cv2.resize(swapped, (int(target_img.shape[1]/scale_up), int(target_img.shape[0]/scale_up)), interpolation=cv2.INTER_AREA)

        return swapped

    except Exception as e:
        print(f"⚠️ Swap lỗi: {e}")
        return target_img

def swap_video(source_img, video_path, output_path):
    """Swap face trong video"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Không thể mở video")
        return False

    # Lấy thông tin video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Khởi tạo video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"🎬 Đang xử lý video: {total_frames} frames")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        print(f"🎬 Frame {frame_count}/{total_frames}")

        # Swap face cho mỗi frame
        try:
            swapped_frame = swap_face(source_img, frame)
            out.write(swapped_frame)
        except Exception as e:
            print(f"⚠️ Lỗi frame {frame_count}: {e}")
            out.write(frame)  # Viết frame gốc nếu swap thất bại

    cap.release()
    out.release()
    print(f"✅ Video hoàn thành: {output_path}")
    return True

@app.route("/")
def index():
    return render_template("index.html")


import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import cpu_count

# Số threads tối đa cho processing parallel
MAX_WORKERS = min(4, cpu_count())

def process_single_target(source_img, target_file, target_filename):
    """Xử lý một target file (image hoặc video)"""
    target_uuid = str(uuid.uuid4())
    tgt_path = f"/tmp/{target_uuid}_{target_filename}"
    try:
        target_file.save(tgt_path)
    except Exception as e:
        print(f"❌ Lỗi lưu file tạm: {e}")
        return {"error": f"Không thể lưu file: {target_filename}"}

    # Check if image
    target_img = cv2.imread(tgt_path)
    if target_img is not None:
        # Process image
        print(f"📷 Processing image: {target_filename}")
        result_img = swap_face(source_img, target_img)
        out_path = f"output/{target_uuid}.jpg"
        cv2.imwrite(out_path, result_img)
        return {"result": f"/view/{out_path}", "type": "image", "original_name": target_filename}
    else:
        # Check if video
        if target_filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            print(f"🎬 Processing video: {target_filename}")
            out_path = f"output/{target_uuid}.mp4"
            if swap_video(source_img, tgt_path, out_path):
                return {"result": f"/view/{out_path}", "type": "video", "original_name": target_filename}
            else:
                return {"error": f"Không thể xử lý video: {target_filename}"}
        else:
            return {"error": f"File không hỗ trợ: {target_filename}"}

@app.route("/swapface", methods=["POST"])
def swapface_api():
    src_file = request.files.get("source")
    tgt_files = request.files.getlist("targets")  # Get multiple targets

    if not src_file or not tgt_files:
        return jsonify({"error": "Thiếu file source hoặc targets"}), 400

    # Save source file
    src_uuid = str(uuid.uuid4())
    src_path = f"/tmp/{src_uuid}_{src_file.filename}"
    src_file.save(src_path)

    source_img = cv2.imread(src_path)
    if source_img is None:
        return jsonify({"error": "Không đọc được ảnh source"}), 400

    print(f"🚀 Starting batch processing for {len(tgt_files)} targets")

    results = []

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(process_single_target, source_img, tgt_file, tgt_file.filename): tgt_file.filename
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
                results.append({"error": f"Lỗi xử lý: {e}"})

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
