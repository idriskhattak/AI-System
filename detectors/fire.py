from pathlib import Path
import torch
import sys
import cv2

YOLOV5_DIR = Path("D:/Projects/Comsat-Project/yolov5-fire-detection/yolov5")

class FireDetector:
    def __init__(self, weights_name="yolov5s_best.pt", conf=0.25):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        weights_path = YOLOV5_DIR / weights_name
        if not weights_path.exists():
            raise FileNotFoundError(f"Fire model not found: {weights_path}")

        # 🔥 CRITICAL FIX: insert YOLO path at highest priority
        sys.path.insert(0, str(YOLOV5_DIR))

        # Now import safely
        from models.common import DetectMultiBackend
        from utils.general import check_img_size, non_max_suppression, scale_boxes
        from utils.torch_utils import select_device
        from utils.augmentations import letterbox

        self.device_obj = select_device('')
        self.model = DetectMultiBackend(str(weights_path), device=self.device_obj)

        self.stride = self.model.stride
        self.names = self.model.names
        self.imgsz = check_img_size((640, 640), s=self.stride)

        self.letterbox = letterbox
        self.nms = non_max_suppression
        self.scale_boxes = scale_boxes

        self.conf = conf

    def predict(self, frame):
        import numpy as np

        img0 = frame.copy()
        img = self.letterbox(img0, self.imgsz, stride=self.stride, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)

        img_tensor = torch.from_numpy(img).to(self.device_obj)
        img_tensor = img_tensor.float() / 255.0
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        with torch.no_grad():
            pred = self.model(img_tensor)
            pred = self.nms(pred, self.conf, 0.45)

        detections = []
        for det in pred:
            if len(det):
                det[:, :4] = self.scale_boxes(
                    img_tensor.shape[2:], det[:, :4], img0.shape
                ).round()

                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = map(int, xyxy)
                    w = x2 - x1
                    h = y2 - y1
                    label = self.names[int(cls)]
                    detections.append((x1, y1, w, h, label, float(conf)))

        return detections
