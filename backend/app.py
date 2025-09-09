import os
import cv2
import time
import uuid
import torch
import librosa
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from deepface import DeepFace

# ---------------- INITIAL SETUP ----------------
app = Flask(__name__)
CORS(app)

# Serve images from uploads directory
from flask import send_from_directory

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- LOAD YOLOv5 MODEL ----------------
print("Loading default YOLOv5s model...")
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
yolo_model.conf = 0.2  # Lower confidence threshold for more sensitive detection
print("YOLOv5s model loaded ✅")

# Allowed belongings
WANTED_CLASSES = {"backpack", "handbag", "suitcase", "cell phone", "laptop", "book", "umbrella"}

# Save uploaded file
def save_file(file):
    filename = f"{int(time.time())}_{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)
    return path

# ---------------- FACE RECOGNITION ----------------
@app.route("/analyze/face", methods=["POST"])
def analyze_face():
    try:
        input_img = request.files.get("input")
        frame_file = request.files.get("frame")
        if not input_img or not frame_file:
            return jsonify({"error": "Provide 'input' and 'frame'"}), 400

        input_path = save_file(input_img)
        frame_path = save_file(frame_file)

        video_exts = [".mp4", ".avi", ".mov", ".mkv"]
        _, ext = os.path.splitext(frame_path)
        ext = ext.lower()
        if ext in video_exts:
            vidcap = cv2.VideoCapture(frame_path)
            fps = vidcap.get(cv2.CAP_PROP_FPS)
            total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = int(total_frames / fps) if fps else total_frames
            sec = 0
            first_match = None
            while sec < duration:
                vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
                success, image = vidcap.read()
                if not success:
                    break
                temp_frame_path = f"{frame_path}_frame_{sec}.jpg"
                cv2.imwrite(temp_frame_path, image)
                frame_number = int(vidcap.get(cv2.CAP_PROP_POS_FRAMES))
                result = DeepFace.verify(
                    img1_path=input_path,
                    img2_path=temp_frame_path,
                    model_name="Facenet512",
                    enforce_detection=False,
                    detector_backend="opencv"
                )
                if bool(result.get("verified")):
                    # Mark the face in the frame
                    try:
                        face_det = DeepFace.detectFace(temp_frame_path, detector_backend="opencv", enforce_detection=False, align=False)
                        # Get face coordinates using OpenCV
                        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                        image = cv2.imread(temp_frame_path)
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                        for (x, y, w, h) in faces:
                            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
                        marked_path = temp_frame_path.replace('.jpg', '_marked.jpg')
                        cv2.imwrite(marked_path, image)
                    except Exception:
                        marked_path = temp_frame_path
                    first_match = {
                        "verified": True,
                        "distance": float(result.get("distance", 0.0)),
                        "threshold": float(result.get("threshold", 0.0)),
                        "model": result.get("model", "Facenet512"),
                        "frame_number": frame_number,
                        "second": sec,
                        "frame": marked_path
                    }
                    break
                sec += 1
            vidcap.release()
            if first_match:
                return jsonify(first_match)
            else:
                return jsonify({"verified": False, "model": "Facenet512", "frame": None})
        else:
            result = DeepFace.verify(
                img1_path=input_path,
                img2_path=frame_path,
                model_name="Facenet512",  # More stable than default
                enforce_detection=False,
                detector_backend="opencv"
            )
            marked_path = None
            if bool(result.get("verified")):
                try:
                    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                    image = cv2.imread(frame_path)
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                    for (x, y, w, h) in faces:
                        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    marked_path = frame_path.replace('.jpg', '_marked.jpg').replace('.jpeg', '_marked.jpeg').replace('.png', '_marked.png')
                    cv2.imwrite(marked_path, image)
                except Exception:
                    marked_path = frame_path
            return jsonify({
                "verified": bool(result.get("verified")),
                "distance": float(result.get("distance", 0.0)),
                "threshold": float(result.get("threshold", 0.0)),
                "model": result.get("model", "Facenet512"),
                "frame": marked_path if marked_path else None
            })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ---------------- OBJECT DETECTION ----------------
@app.route("/analyze/object", methods=["POST"])
def analyze_object():
    input_file = request.files.get("input")  # reference image
    frame_file = request.files.get("frame")  # sample image/video
    if not input_file or not frame_file:
        return jsonify({"error": "Provide 'input' and 'frame'"}), 400

    try:
        input_path = save_file(input_file)
        frame_path = save_file(frame_file)

        # Detect most confident class in reference image (person or object)
        ref_results = yolo_model(input_path)
        ref_df = ref_results.pandas().xyxy[0]
        if ref_df.empty:
            return jsonify({
                "detections": [],
                "frame": None,
                "message": "No object detected in reference image. Please upload a clear image of the object you want to find."
            })
        # Find the detection with highest confidence
        best_row = ref_df.loc[ref_df['confidence'].idxmax()]
        target_class = best_row['name']

        video_exts = [".mp4", ".avi", ".mov", ".mkv"]
        _, ext = os.path.splitext(frame_path)
        ext = ext.lower()
        sample_objects = set()
        detections = []
        marked_path = None
        if ext in video_exts:
            vidcap = cv2.VideoCapture(frame_path)
            fps = vidcap.get(cv2.CAP_PROP_FPS)
            total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = int(total_frames / fps) if fps else total_frames
            sec = 0
            best_detection = None
            best_conf = -1
            best_frame = None
            best_frame_number = None
            best_marked_path = None
            while sec < duration:
                vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
                success, image = vidcap.read()
                if not success:
                    break
                temp_frame_path = f"{frame_path}_frame_{sec}.jpg"
                cv2.imwrite(temp_frame_path, image)
                frame_number = int(vidcap.get(cv2.CAP_PROP_POS_FRAMES))
                results = yolo_model(temp_frame_path)
                df = results.pandas().xyxy[0]
                for _, row in df.iterrows():
                    if row["confidence"] >= yolo_model.conf and row["name"] == target_class:
                        if row["confidence"] > best_conf:
                            detection = {
                                "label": row["name"],
                                "confidence": float(row["confidence"]),
                                "bbox": [float(row["xmin"]), float(row["ymin"]), float(row["xmax"]), float(row["ymax"])]
                            }
                            # Draw bounding box for detected object or person
                            x1, y1, x2, y2 = int(row["xmin"]), int(row["ymin"]), int(row["xmax"]), int(row["ymax"])
                            image = cv2.imread(temp_frame_path)
                            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                            marked_path = temp_frame_path.replace('.jpg', '_marked.jpg').replace('.jpeg', '_marked.jpeg').replace('.png', '_marked.png')
                            cv2.imwrite(marked_path, image)
                            best_detection = detection
                            best_conf = row["confidence"]
                            best_frame = marked_path
                            best_frame_number = frame_number
                            best_sec = sec
                sec += 1
            vidcap.release()
            if best_detection:
                return jsonify({
                    "detections": [best_detection],
                    "second": best_sec,
                    "frame_number": best_frame_number,
                    "frame": best_frame
                })
            else:
                return jsonify({"detections": [], "second": None, "frame": None, "message": "Object not found in video."})
        else:
            results = yolo_model(frame_path)
            df = results.pandas().xyxy[0]
            image = cv2.imread(frame_path)
            best_detection = None
            best_conf = -1
            best_marked_path = None
            for _, row in df.iterrows():
                if row["confidence"] >= yolo_model.conf and row["name"] == target_class:
                    if row["confidence"] > best_conf:
                        detection = {
                            "label": row["name"],
                            "confidence": float(row["confidence"]),
                            "bbox": [float(row["xmin"]), float(row["ymin"]), float(row["xmax"]), float(row["ymax"])]
                        }
                        # Draw bounding box for detected object or person
                        x1, y1, x2, y2 = int(row["xmin"]), int(row["ymin"]), int(row["xmax"]), int(row["ymax"])
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        marked_path = frame_path.replace('.jpg', '_marked.jpg').replace('.jpeg', '_marked.jpeg').replace('.png', '_marked.png')
                        cv2.imwrite(marked_path, image)
                        best_detection = detection
                        best_conf = row["confidence"]
                        best_marked_path = marked_path
            if best_detection:
                return jsonify({
                    "detections": [best_detection],
                    "frame": best_marked_path
                })
            else:
                return jsonify({"detections": [], "frame": None, "message": "Object not found in image."})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ---------------- EMOTION DETECTION ----------------
@app.route("/analyze/emotion", methods=["POST"])
def analyze_emotion():
    try:
        frame_file = request.files.get("frame")
        if not frame_file:
            return jsonify({"error": "Provide 'frame'"}), 400

        frame_path = save_file(frame_file)

        video_exts = [".mp4", ".avi", ".mov", ".mkv"]
        image_exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
        _, ext = os.path.splitext(frame_path)
        ext = ext.lower()
        if ext in video_exts:
            vidcap = cv2.VideoCapture(frame_path)
            fps = vidcap.get(cv2.CAP_PROP_FPS)
            total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = int(total_frames / fps) if fps else total_frames
            sec = 0
            first_emotion = None
            while sec < duration:
                vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
                success, image = vidcap.read()
                if not success:
                    break
                temp_frame_path = f"{frame_path}_frame_{sec}.jpg"
                cv2.imwrite(temp_frame_path, image)
                try:
                    emo = DeepFace.analyze(
                        img_path=temp_frame_path,
                        actions=['emotion'],
                        enforce_detection=False
                    )
                    if isinstance(emo, list):
                        emo = emo[0]
                    if emo.get("dominant_emotion", None):
                        emotions = emo.get("emotion", {})
                        emotions_clean = {k: float(v) for k, v in emotions.items()}
                        first_emotion = {
                            "dominant_emotion": emo.get("dominant_emotion", "unknown"),
                            "emotions": emotions_clean,
                            "second": sec
                        }
                        break
                except Exception:
                    pass
                sec += 1
            vidcap.release()
            if first_emotion:
                return jsonify(first_emotion)
            else:
                return jsonify({"dominant_emotion": None, "emotions": {}, "second": None})
        elif ext in image_exts:
            emo = DeepFace.analyze(
                img_path=frame_path,
                actions=['emotion'],
                enforce_detection=False
            )
            if isinstance(emo, list):
                emo = emo[0]
            emotions = emo.get("emotion", {})
            emotions_clean = {k: float(v) for k, v in emotions.items()}
            return jsonify({
                "dominant_emotion": emo.get("dominant_emotion", "unknown"),
                "emotions": emotions_clean
            })
        else:
            return jsonify({"error": "Unsupported file type for emotion analysis. Please upload an image or video file."}), 400
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ---------------- AUDIO PANIC DETECTION ----------------
@app.route("/analyze/audio", methods=["POST"])
def analyze_audio():
    try:
        audio = request.files.get("audio")
        if not audio:
            return jsonify({"error": "Provide 'audio'"}), 400

        allowed_types = [".mp3", ".wav", ".ogg", ".mpeg", ".mpg", ".aac", ".m4a", ".webm"]
        _, ext = os.path.splitext(audio.filename)
        ext = ext.lower()
        if ext not in allowed_types:
            return jsonify({"error": "Unsupported file type for audio analysis. Please upload a valid audio file (mp3, wav, ogg, etc)."}), 400

        path = save_file(audio)
        y, sr = librosa.load(path, sr=None, mono=True)
        energy = float(np.mean(np.abs(y))) if len(y) else 0.0
        status = "PANIC" if energy > 0.06 else "NORMAL"

        return jsonify({"energy": round(energy, 4), "status": status})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ---------------- HEALTH CHECK ----------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "yolov5": True, "deepface": True})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8001, debug=True, use_reloader=False)
