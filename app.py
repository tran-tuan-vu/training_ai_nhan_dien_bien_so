"""
Ứng dụng web đánh giá mô hình YOLO nhận diện biển số xe.
Upload ảnh -> chạy inference -> trả về ảnh có bounding box + danh sách phát hiện.
Với mỗi biển số phát hiện được, cắt vùng ảnh tương ứng và dùng PaddleOCR để đọc chữ.
Kết quả (ảnh, biển số, thời gian) được lưu vào MongoDB để tra cứu lại sau;
số thứ tự (stt) hiển thị được tính động theo vị trí, không lưu cố định trong DB.
"""

import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import cv2
from flask import Flask, Response, render_template, request, jsonify, url_for
from pymongo import MongoClient, DESCENDING
from pymongo.errors import PyMongoError
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
WEIGHTS_PATH = ROOT / "runs" / "license_plate_train" / "weights" / "best.pt"
UPLOAD_DIR = ROOT / "static" / "uploads"
RESULT_DIR = ROOT / "static" / "results"
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

MONGO_URI = "mongodb://localhost:27017/"
MONGO_DB_NAME = "license_plate_db"
MONGO_COLLECTION_NAME = "detections"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB

_model = None
_ocr = None
_mongo_client = None
_model_lock = threading.Lock()
_ocr_lock = threading.Lock()

# Biên độ mở rộng vùng cắt biển số trước khi đưa vào OCR, giúp OCR không bị
# cắt sát mất ký tự ở rìa khi bounding box của YOLO hơi hẹp.
CROP_PADDING_RATIO = 0.08

# Khi 1 biển số vẫn đứng yên trước camera nhiều giây, chỉ lưu lại vào MongoDB
# tối đa 1 lần mỗi khoảng thời gian này (giây), tránh lưu trùng lặp liên tục.
CAMERA_SAVE_COOLDOWN_SEC = 8


def get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                if not WEIGHTS_PATH.exists():
                    raise FileNotFoundError(f"Không tìm thấy file trọng số mô hình: {WEIGHTS_PATH}")
                _model = YOLO(str(WEIGHTS_PATH))
    return _model


def get_ocr():
    global _ocr
    if _ocr is None:
        with _ocr_lock:
            if _ocr is None:
                from paddleocr import PaddleOCR

                _ocr = PaddleOCR(
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    lang="en",
                )
    return _ocr


def get_mongo_db():
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
    return _mongo_client[MONGO_DB_NAME]


def get_mongo_collection():
    return get_mongo_db()[MONGO_COLLECTION_NAME]


def save_detections_to_db(detections, result_url):
    """Lưu từng biển số phát hiện được thành 1 document trong MongoDB.

    Không lưu "stt" cố định trong DB: nếu lưu 1 số tăng dần rồi có bản ghi bị
    xóa (thủ công hoặc qua ứng dụng), các số còn lại sẽ bị hụt/lệch. Thay vào
    đó "stt" luôn được tính lại theo vị trí hiển thị mỗi khi đọc danh sách
    (xem route /history), nên luôn liền mạch 1,2,3... dù đã xóa bớt bản ghi.

    Lỗi kết nối DB không được làm hỏng luồng nhận diện chính, nên chỉ log cảnh báo.
    """
    if not detections:
        return False
    try:
        collection = get_mongo_collection()
        now = datetime.now()  # giờ hệ thống (địa phương), khớp với giờ hiển thị trên giao diện
        for d in detections:
            collection.insert_one({
                "image": result_url,
                "plate_text": d["plate_text"],
                "created_at": now,
            })
        return True
    except PyMongoError as e:
        print(f"[MongoDB] Không thể lưu kết quả: {e}")
        return False


def crop_with_padding(image, x1, y1, x2, y2, ratio=CROP_PADDING_RATIO):
    h, w = image.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    pad_x, pad_y = bw * ratio, bh * ratio
    cx1 = max(0, int(x1 - pad_x))
    cy1 = max(0, int(y1 - pad_y))
    cx2 = min(w, int(x2 + pad_x))
    cy2 = min(h, int(y2 + pad_y))
    return image[cy1:cy2, cx1:cx2]


def read_plate_text(plate_crop):
    """Chạy PaddleOCR trên vùng ảnh biển số, trả về (text, confidence trung bình)."""
    if plate_crop.size == 0:
        return "", 0.0

    ocr = get_ocr()
    outputs = ocr.predict(plate_crop)
    if not outputs:
        return "", 0.0

    res = outputs[0]
    texts = res.get("rec_texts", []) if hasattr(res, "get") else res["rec_texts"]
    scores = res.get("rec_scores", []) if hasattr(res, "get") else res["rec_scores"]
    polys = res.get("rec_polys", res.get("dt_polys", [])) if hasattr(res, "get") else []

    if not texts:
        return "", 0.0

    # Sắp xếp các dòng chữ theo thứ tự đọc (trên->dưới, trái->phải) để ghép
    # đúng thứ tự cho biển số 2 dòng (phổ biến ở biển số xe máy).
    items = list(zip(texts, scores, polys)) if polys else list(zip(texts, scores, [None] * len(texts)))

    def sort_key(item):
        _, _, poly = item
        if poly is None:
            return (0, 0)
        ys = [p[1] for p in poly]
        xs = [p[0] for p in poly]
        return (sum(ys) / len(ys), sum(xs) / len(xs))

    items.sort(key=sort_key)

    combined_text = " ".join(t.strip() for t, _, _ in items if t.strip())
    avg_conf = sum(s for _, s, _ in items) / len(items) if items else 0.0
    return combined_text.upper(), round(float(avg_conf), 4)


class CameraWorker:
    """Quản lý 1 camera: 1 luồng đọc frame + chạy YOLO liên tục (để video mượt),
    và 1 luồng OCR riêng luôn xử lý biển số mới nhất đang chờ (vì OCR chậm hơn
    nhiều so với tốc độ camera nên không thể xử lý tuyệt đối từng frame)."""

    def __init__(self):
        self.cap = None
        self.conf = 0.25
        self.running = False
        self.control_lock = threading.Lock()

        self.frame_lock = threading.Lock()
        self.latest_jpeg = None

        self.pending_lock = threading.Lock()
        self.pending_crop = None

        self.overlay_text = ""
        self.overlay_until = 0

        self.last_saved_text = None
        self.last_saved_time = 0

    def start(self, conf=0.25):
        with self.control_lock:
            if self.running:
                return
            self.conf = conf
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise RuntimeError("Không mở được camera. Kiểm tra xem camera có đang bị ứng dụng khác chiếm dụng không.")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap = cap
            self.running = True
            threading.Thread(target=self._capture_loop, daemon=True).start()
            threading.Thread(target=self._ocr_loop, daemon=True).start()

    def stop(self):
        with self.control_lock:
            self.running = False
            if self.cap is not None:
                self.cap.release()
                self.cap = None
        with self.frame_lock:
            self.latest_jpeg = None

    def _capture_loop(self):
        model = get_model()
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            with _model_lock:
                results = model.predict(frame, conf=self.conf, verbose=False)
            result = results[0]
            annotated = result.plot()

            if result.boxes is not None and len(result.boxes) > 0:
                best_box = max(result.boxes, key=lambda b: float(b.conf[0]))
                x1, y1, x2, y2 = best_box.xyxy[0].tolist()
                crop = crop_with_padding(frame, x1, y1, x2, y2)
                with self.pending_lock:
                    self.pending_crop = crop

            if self.overlay_text and time.time() < self.overlay_until:
                cv2.putText(
                    annotated, self.overlay_text, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA,
                )

            ok2, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok2:
                with self.frame_lock:
                    self.latest_jpeg = buf.tobytes()

    def _ocr_loop(self):
        while self.running:
            crop = None
            with self.pending_lock:
                if self.pending_crop is not None:
                    crop = self.pending_crop
                    self.pending_crop = None
            if crop is None:
                time.sleep(0.05)
                continue

            plate_text, ocr_conf = read_plate_text(crop)
            if not plate_text:
                continue

            self.overlay_text = f"{plate_text} ({ocr_conf * 100:.0f}%)"
            self.overlay_until = time.time() + 4

            now = time.time()
            is_new_plate = plate_text != self.last_saved_text
            cooldown_passed = (now - self.last_saved_time) > CAMERA_SAVE_COOLDOWN_SEC
            if is_new_plate or cooldown_passed:
                snapshot_name = f"cam_{uuid.uuid4().hex[:8]}.jpg"
                snapshot_path = RESULT_DIR / snapshot_name
                with self.frame_lock:
                    jpeg = self.latest_jpeg
                if jpeg:
                    snapshot_path.write_bytes(jpeg)
                    # Không dùng url_for() ở đây vì đang chạy trong background thread,
                    # không có Flask app/request context (sẽ ném RuntimeError).
                    result_url = f"/static/results/{snapshot_name}"
                    save_detections_to_db([{"plate_text": plate_text}], result_url)
                    self.last_saved_text = plate_text
                    self.last_saved_time = now

    def get_jpeg(self):
        with self.frame_lock:
            return self.latest_jpeg


camera_worker = CameraWorker()


@app.route("/")
def index():
    return render_template("index.html", weights_name=WEIGHTS_PATH.name)


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Không tìm thấy ảnh trong yêu cầu."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Chưa chọn ảnh."}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"error": f"Định dạng không hỗ trợ: {ext}"}), 400

    try:
        conf = float(request.form.get("conf", 0.25))
    except ValueError:
        conf = 0.25
    conf = max(0.01, min(conf, 1.0))

    uid = uuid.uuid4().hex[:8]
    upload_name = f"{uid}{ext}"
    upload_path = UPLOAD_DIR / upload_name
    file.save(upload_path)

    try:
        model = get_model()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500

    t0 = time.time()
    with _model_lock:
        results = model.predict(source=str(upload_path), conf=conf, verbose=False)
    detect_ms = (time.time() - t0) * 1000

    result = results[0]
    result_name = f"{uid}_result.jpg"
    result_path = RESULT_DIR / result_name
    annotated = result.plot()  # numpy array (BGR)
    cv2.imwrite(str(result_path), annotated)

    original_image = result.orig_img  # numpy array (BGR), ảnh gốc đã đọc sẵn bởi ultralytics

    detections = []
    names = result.names
    t_ocr0 = time.time()
    if result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            plate_crop = crop_with_padding(original_image, x1, y1, x2, y2)
            plate_text, ocr_conf = read_plate_text(plate_crop)
            detections.append({
                "class": names.get(int(box.cls[0]), str(int(box.cls[0]))),
                "confidence": round(float(box.conf[0]), 4),
                "bbox": [round(v, 1) for v in [x1, y1, x2, y2]],
                "plate_text": plate_text,
                "ocr_confidence": ocr_conf,
            })
    ocr_ms = (time.time() - t_ocr0) * 1000

    detections.sort(key=lambda d: d["confidence"], reverse=True)

    original_url = url_for("static", filename=f"uploads/{upload_name}")
    result_url = url_for("static", filename=f"results/{result_name}")
    saved_to_db = save_detections_to_db(detections, result_url)

    return jsonify({
        "original_url": original_url,
        "result_url": result_url,
        "detections": detections,
        "num_detections": len(detections),
        "inference_ms": round(detect_ms, 1),
        "ocr_ms": round(ocr_ms, 1),
        "conf_threshold": conf,
        "saved_to_db": saved_to_db,
    })


@app.route("/history")
def history():
    limit = request.args.get("limit", 20, type=int)
    limit = max(1, min(limit, 200))

    try:
        collection = get_mongo_collection()
        total = collection.count_documents({})
        records = list(
            collection.find().sort([("created_at", DESCENDING), ("_id", DESCENDING)]).limit(limit)
        )
    except PyMongoError as e:
        return jsonify({"error": f"Không thể kết nối MongoDB: {e}"}), 500

    # "stt" luôn được tính lại theo vị trí hiện tại (không đọc từ DB), nên vẫn
    # liền mạch dù có bản ghi đã bị xóa trước đó (bản ghi mới nhất = số lớn nhất).
    for idx, r in enumerate(records):
        r["stt"] = total - idx
        r["_id"] = str(r["_id"])
        r["created_at"] = r["created_at"].isoformat()

    return jsonify({"records": records})


@app.route("/video_feed")
def video_feed():
    conf = request.args.get("conf", 0.25, type=float)
    conf = max(0.01, min(conf, 1.0))
    try:
        camera_worker.start(conf=conf)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500

    def generate():
        last_jpeg = None
        while camera_worker.running:
            jpeg = camera_worker.get_jpeg()
            # Chỉ gửi khi có frame mới (capture loop chậm hơn nhiều so với tốc độ
            # polling này); tránh gửi lặp cùng 1 frame ở tốc độ tối đa gây nghẽn.
            if jpeg is None or jpeg is last_jpeg:
                time.sleep(0.03)
                continue
            last_jpeg = jpeg
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/camera/stop", methods=["POST"])
def camera_stop():
    camera_worker.stop()
    return jsonify({"stopped": True})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
