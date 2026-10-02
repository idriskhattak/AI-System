import cv2
import os
from pathlib import Path
from flask import Flask, render_template, Response, request
from werkzeug.utils import secure_filename

from detectors.gender import GenderDetector
from detectors.fire import FireDetector
from detectors.human import HumanDetector
from detectors.ethnicity import EthnicityDetector
from detectors.fight import FightDetector

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = BASE_DIR / "weights"


# -------------------------
# Model factory
# -------------------------

def load_model(model_name):
    if model_name == "human":
        return HumanDetector()

    elif model_name == "gender":
        return GenderDetector()

    elif model_name == "ethnicity":
        return EthnicityDetector()

    elif model_name == "fire":
        return FireDetector()

    elif model_name == "fight":
        fight_weights = WEIGHTS_DIR / "fight_model.pth"
        return FightDetector(fight_weights)

    elif model_name == "full":
        return HumanDetector()  # can improve later

    else:
        raise ValueError("Invalid model")


# -------------------------
# Streaming generator
# -------------------------

def generate_frames(video_path, model_name):

    model = load_model(model_name)

    # -------------------------
    # SPECIAL CASE: FIGHT MODEL
    # -------------------------
    if model_name == "fight":

        output_path = BASE_DIR / "temp_fight_output.mp4"

        # This runs full video processing internally
        model.predict_video(video_path, output_path)

        cap = cv2.VideoCapture(str(output_path))

        try:
            while True:
                success, frame = cap.read()
                if not success:
                    break

                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' +
                       frame_bytes +
                       b'\r\n')
        finally:
            cap.release()

        return

    # -------------------------
    # NORMAL FRAME-BY-FRAME MODELS
    # -------------------------

    cap = cv2.VideoCapture(video_path)

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            detections = model.predict(frame)

            for x, y, w, h, label, conf in detections:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{label} {conf:.2f}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   frame_bytes +
                   b'\r\n')
    finally:
        cap.release()


# -------------------------
# Routes
# -------------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["video"]
    filename = secure_filename(file.filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)
    return {"path": path}


@app.route("/stream")
def stream():
    video_path = request.args.get("video")
    model_name = request.args.get("model")

    return Response(
        generate_frames(video_path, model_name),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


if __name__ == "__main__":
    app.run(debug=True)
