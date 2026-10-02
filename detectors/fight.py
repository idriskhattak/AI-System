from pathlib import Path
import sys
import cv2
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = BASE_DIR / "weights"
# Add project root so UtilsFiles works
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from UtilsFiles.Fight_utils import loadModel, predict_on_video


class FightDetector:
    def __init__(self, weights_path):
        self.weights_path = Path(weights_path)

        if not self.weights_path.exists():
            raise FileNotFoundError(f"Fight model not found: {self.weights_path}")

        print("[INFO] Loading Fight Model...")
        self.model = loadModel(str(self.weights_path))

    def predict_video(
        self,
        input_video,
        output_video,
        sequence_length=16,
        skip=2,
        show_info=True,
        preview=True
    ):
        predict_on_video(
            str(input_video),
            str(output_video),
            self.model,
            sequence_length,
            skip,
            show_info,
            preview,
            "Fight Detection",
            "q",
            1
        )
