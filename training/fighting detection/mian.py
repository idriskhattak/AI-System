from pathlib import Path
import sys
import time

# Make local modules importable
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

# Import repo utilities
from UtilsFiles.Fight_utils import loadModel, predict_on_video  # noqa


# ==== FIXED PATHS (edit these if you want) ====
MODEL_PATH = REPO_ROOT / "Models" / "model_16_m3_0.8888.pth"
VIDEO_PATH = REPO_ROOT / "WhatsApp Video 2025-08-18 at 12.02.33_50549d28.mp4"           # <- put your test video here or change the path
OUTPUT_PATH = REPO_ROOT / "output.mp4"

# ==== SETTINGS ====
SEQUENCE_LENGTH = 16
SKIP = 2
SHOW_INFO = True


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

    # The repo’s predict_on_video signature varies between versions.
    # We try the known variants in order (positional args only).
    ran = False
    try:
        # Variant A (most common in this repo):
        # predict_on_video(inputPath, outputPath, model, sequenceLength, skip, showInfo)
        predict_on_video(str(VIDEO_PATH), str(OUTPUT_PATH), model,
                         SEQUENCE_LENGTH, SKIP, SHOW_INFO)
        ran = True
    except TypeError:
        try:
            # Variant B (some forks invert first two args):
            # predict_on_video(model, inputPath, outputPath, sequenceLength, skip, showInfo)
            predict_on_video(model, str(VIDEO_PATH), str(OUTPUT_PATH),
                             SEQUENCE_LENGTH, SKIP, SHOW_INFO)
            ran = True
        except TypeError as e:
            print("[ERROR] Couldn't call predict_on_video with known signatures.")
            print("Details:", e)
            print("If you share the first 20 lines of UtilsFiles/Fight_utils.py (the function definition),")
            print("I’ll set the exact call for your copy.")
            return

    dt = time.time() - t0
    if ran:
        print(f"[OK] Saved annotated video -> {OUTPUT_PATH}  (elapsed {dt:.1f}s)")


if __name__ == "__main__":
    main()
