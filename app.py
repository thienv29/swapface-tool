from flask import Flask, request, jsonify, render_template, send_file
import cv2
import numpy as np
import os
import uuid
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

app = Flask(__name__, template_folder="templates")
os.makedirs("output", exist_ok=True)

print("🚀 Loading face detection and swap models...")
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=-1, det_size=(640, 640))  # CPU cho M1
swapper = get_model("/Users/thienvuquy/.insightface/models/inswapper_128.onnx", download=False)
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
        if isinstance(source_face, np.ndarray) or not hasattr(source_face, "normed_embedding"):
            source_face = face_app.get(source_img)[0]
        if isinstance(target_face, np.ndarray) or not hasattr(target_face, "normed_embedding"):
            target_face = face_app.get(target_img)[0]

        swapped = swapper.get(target_img, target_face, source_face, paste_back=True)

        (x1, y1, x2, y2) = target_face.bbox.astype(int)
        face_crop = swapped[y1:y2, x1:x2]
        target_crop = target_img[y1:y2, x1:x2]

        # 🩹 Tạo mask mềm quanh khuôn mặt
        h, w = face_crop.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(mask, (w // 2, h // 2), int(min(h, w) * 0.45), 1, -1)
        mask = cv2.GaussianBlur(mask, (31, 31), 10)
        mask = np.expand_dims(mask, axis=2)

        # 🧩 Blend thủ công (không dùng seamlessClone)
        blended = face_crop.astype(np.float32) * mask + target_crop.astype(np.float32) * (1 - mask)
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        # ✨ Sharpen nhẹ vùng blend (trả lại độ nét)
        sharpen = cv2.addWeighted(blended, 1.25, cv2.GaussianBlur(blended, (0, 0), 2), -0.25, 0)

        result = target_img.copy()
        result[y1:y2, x1:x2] = sharpen

        return result

    except Exception as e:
        print(f"⚠️ Swap lỗi: {e}")
        return target_img

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/swapface", methods=["POST"])
def swapface_api():
    src_file = request.files.get("source")
    tgt_file = request.files.get("target")

    if not src_file or not tgt_file:
        return jsonify({"error": "Thiếu file source hoặc target"}), 400

    src_path = f"/tmp/{uuid.uuid4()}_{src_file.filename}"
    tgt_path = f"/tmp/{uuid.uuid4()}_{tgt_file.filename}"
    src_file.save(src_path)
    tgt_file.save(tgt_path)

    source_img = cv2.imread(src_path)
    target_img = cv2.imread(tgt_path)
    if source_img is None or target_img is None:
        return jsonify({"error": "Không đọc được ảnh"}), 400

    result_img = swap_face(source_img, target_img)
    out_path = f"output/{uuid.uuid4()}.jpg"
    cv2.imwrite(out_path, result_img)
    print(f"✅ Kết quả lưu tại: {out_path}")

    return jsonify({"result": f"/view/{out_path}", "type": "image"})


@app.route("/view/<path:filename>")
def view_file(filename):
    mimetype = "image/jpeg" if filename.lower().endswith((".jpg", ".jpeg", ".png")) else "video/mp4"
    return send_file(filename, mimetype=mimetype)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)