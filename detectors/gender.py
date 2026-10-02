import cv2
import numpy as np
from PIL import Image
from torchvision import models, transforms
import torch
import torch.nn as nn
from project_utils.paths import DATA_DIR
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = BASE_DIR / "weights"

class GenderDetector:
    def __init__(self, weights_name="gender_resnet18_best.pth", img_size=224, class_names=("female", "male")):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = list(class_names)
        self.img_size = img_size

        self.model_path = WEIGHTS_DIR / weights_name
        if not self.model_path.exists():
            raise FileNotFoundError(f"Gender model not found: {self.model_path}")

        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, len(self.class_names))
        state = torch.load(str(self.model_path), map_location="cpu")
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

        self.preprocess = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        # simple face detector (haar)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def predict(self, frame_bgr: np.ndarray):
        """Returns detections: list of (x,y,w,h,label,conf)"""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        boxes = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(40, 40))
        detections = []

        crops = []
        rects = []
        H, W = frame_bgr.shape[:2]

        for (x, y, w, h) in boxes:
            x = max(0, x); y = max(0, y)
            w = min(w, W - x); h = min(h, H - y)
            face = frame_bgr[y:y+h, x:x+w]
            if face.size == 0:
                continue
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            tensor = self.preprocess(Image.fromarray(face_rgb))
            crops.append(tensor)
            rects.append((x, y, w, h))

        if not crops:
            return []

        batch = torch.stack(crops).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(batch), dim=1).cpu().numpy()

        for (x, y, w, h), p in zip(rects, probs):
            idx = int(np.argmax(p))
            detections.append((x, y, w, h, self.class_names[idx], float(p[idx])))

        return detections
