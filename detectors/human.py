import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

class HumanDetector:
    def __init__(self, weights_path=None, conf=0.45, imgsz=640):
        self.conf = conf
        self.imgsz = imgsz

        # If no custom weights, use official yolov8
        self.model = YOLO(weights_path or "yolov8s.pt")

        self.unique_ids = set()

    def predict(self, frame):
        """
        Returns detections:
        list of (x, y, w, h, label, conf)
        """
        results = self.model.track(
            frame,
            imgsz=self.imgsz,
            conf=self.conf,
            classes=[0],  # person class only
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        detections = []

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()

            for box, track_id, conf in zip(boxes, ids, confs):
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1

                self.unique_ids.add(track_id)

                detections.append(
                    (x1, y1, w, h, f"Person {track_id}", float(conf))
                )

        return detections

    def get_total_count(self):
        return len(self.unique_ids)
