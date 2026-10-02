from pathlib import Path
import sys
import time

# Make local modules importable
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

# Import repo utilities
from UtilsFiles.Fight_utils import loadModel, predict_on_video  # noqapyht


# ==== FIXED PATHS (edit these if you want) ====
MODEL_PATH = REPO_ROOT / "Models" / "model_16_m3_0.8888.pth"
VIDEO_PATH = REPO_ROOT / "WhatsApp Video 2025-08-18 at 12.02.33_50549d28.mp4"
OUTPUT_PATH = REPO_ROOT / "output.mp4"

# ==== SETTINGS ====
SEQUENCE_LENGTH = 16
SKIP = 2
SHOW_INFO = True

# ==== NEW SETTINGS ====
PREVIEW = True
WINDOW_NAME = "Fight Detection (Live)"
EXIT_KEY = "q"
PREVIEW_WAIT = 1  # increase to 10 if window not updating smoothly


def main():
    if not MODEL_PATH.exists():
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        return
    if not VIDEO_PATH.exists():
        print(f"[ERROR] Input video not found: {VIDEO_PATH}")
        return

    print(f"[INFO] Loading model: {MODEL_PATH}")
    model = loadModel(str(MODEL_PATH))  # loadModel takes ONLY the path

    print(f"[INFO] Running on: {VIDEO_PATH}")
    t0 = time.time()

    ran = False

    try:
        # Your real signature:
        # predict_on_video(video_file_path, output_file_path, model, SEQUENCE_LENGTH, skip=2, showInfo=False, ...)
        # We pass preview options positionally to avoid keyword mismatch issues.
        predict_on_video(
            str(VIDEO_PATH),
            str(OUTPUT_PATH),
            model,
            SEQUENCE_LENGTH,
            SKIP,
            SHOW_INFO,
            PREVIEW,
            WINDOW_NAME,
            EXIT_KEY,
            PREVIEW_WAIT
        )
        ran = True

    except TypeError as e:
        print("[ERROR] Couldn't call predict_on_video with this signature.")
        print("Details:", e)
        return

    dt = time.time() - t0
    if ran:
        print(f"[OK] Live preview shown, and saved annotated video -> {OUTPUT_PATH}  (elapsed {dt:.1f}s)")


if __name__ == "__main__":
    main()
