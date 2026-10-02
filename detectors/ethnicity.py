import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import models, transforms
from project_utils.paths import WEIGHTS_DIR


class EthnicityDetector:
    def __init__(self, weights_name="fairface_resnet18.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_path = WEIGHTS_DIR / weights_name
        if not self.model_path.exists():
            raise FileNotFoundError(f"Ethnicity model not found: {self.model_path}")

        ckpt = torch.load(str(self.model_path), map_location=self.device)

        self.classes = ckpt["classes"]
        img_size = ckpt.get("img_size", 224)

        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, len(self.classes))

        state_dict = ckpt["state_dict"]
        if any(k.startswith("module.") for k in state_dict):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],
                                 [0.229,0.224,0.225])
        ])

        # Simple Haar face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def predict(self, frame_bgr):
        """
        Returns list of:
        (x, y, w, h, label, confidence)
        """

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        boxes = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(40, 40)
        )

        detections = []
        crops = []
        rects = []

        for (x, y, w, h) in boxes:
            face = frame_bgr[y:y+h, x:x+w]
            if face.size == 0:
                continue

            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            tensor = self.transform(Image.fromarray(face_rgb))

            crops.append(tensor)
            rects.append((x, y, w, h))

        if not crops:
            return []

        batch = torch.stack(crops).to(self.device)

        with torch.no_grad():
            probs = F.softmax(self.model(batch), dim=1).cpu().numpy()

        for (x, y, w, h), p in zip(rects, probs):
            idx = int(np.argmax(p))
            label = self.classes[idx]

            # Remap logic (optional — keep your original rule)
            if label not in {"White", "Black"}:
                label = "Asian"

            detections.append((x, y, w, h, label, float(p[idx])))

        return detections
