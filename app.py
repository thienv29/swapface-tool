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
