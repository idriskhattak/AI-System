# infer_auto_blur.py
# - One script for images / videos / webcam
# - Detect faces (DeepFace 'mediapipe' by default; fallback to OpenCV Haar)
# - Classify each detected face using YOUR checkpoint (any labels)
# - Robust bbox filtering, readable labels scaled by box size
# - Batch classification for speed, save + show outputs
# - Consistent preview window size (resizes display only, not saved output)

import sys, time, csv, cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import models, transforms as T

BASE_DIR = Path(__file__).resolve().parent
# ---------------- USER CONFIG ----------------
INPUT = BASE_DIR / "data" / "test.jpeg" # image path, video path, or 0 for webcam
BASE_DIR = Path(__file__).resolve().parent
CKPT_PATH = BASE_DIR / "fairface_resnet18.pt"  # your model checkpoint (any labels)
BACKEND          = "mediapipe"          # 'mediapipe', 'retinaface' (if cached), or 'opencv'
SAVE_OVERLAY     = True                 # save <image>_pred.png or <video>_pred.mp4
SHOW_WINDOW      = True                 # show annotated window (set False on headless servers)
SAVE_CROPS       = False                # also save detected face crops
WRITE_CSV_LOG    = False                # write detections/preds to CSV

# Video-specific:
FRAME_STRIDE     = 1                    # process every Nth frame
MAX_LONG_EDGE    = 1280                 # processing resize long edge for speed; None to disable
OUT_FPS          = None                 # None -> source fps

# Display-only (does not affect saved files / processing):
DISPLAY_LONG_EDGE = 900                 # preview window long-edge (same size for all)
WINDOW_NAME       = "Prediction"
# ---------------------------------------------

def log(*a): print(*a, flush=True)

# ---------- Model ----------
def strip_module_prefix(state):
    if any(k.startswith("module.") for k in state):
        return {k.replace("module.", "", 1): v for k, v in state.items()}
    return state

def build_model(ckpt, device):
    """
    Expected checkpoint dict:
      - 'classes': list[str]  (e.g., ['Blurry','Sharp'] or your labels)
      - 'img_size': int       (default 224)
      - 'arch': str           ('resnet18' default, supports 'convnext_tiny')
      - 'state_dict': weights
    """
    classes  = ckpt["classes"]
    img_size = ckpt.get("img_size", 224)
    arch     = ckpt.get("arch", "resnet18")

    if arch == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(classes))
    elif arch == "convnext_tiny":
        model = models.convnext_tiny(weights=None)
        in_feats = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_feats, len(classes))
    else:
        log(f"[warn] Unknown arch '{arch}', defaulting to resnet18.")
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(classes))

    sd = strip_module_prefix(ckpt["state_dict"])
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    return model, classes, img_size

def make_transform(sz):
    return T.Compose([
        T.Resize((sz, sz)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

@torch.no_grad()
def classify_batch(model, classes, tfm, device, pil_list):
    """Classify a list of PIL face crops in one batch (faster for multiple faces)."""
    if not pil_list:
        return []
    xs = torch.stack([tfm(p) for p in pil_list], dim=0).to(device)
    logits = model(xs)
    probs  = F.softmax(logits, dim=1)
    confs, ids = probs.max(dim=1)
    labels = [classes[i.item()] for i in ids]

    # 🔥 remap logic: only keep White & Black, all others → Asian
    remapped = []
    for lbl, conf in zip(labels, confs):
        if lbl not in {"White", "Black"}:
            lbl = "Asian"
        remapped.append((lbl, float(conf.item())))
    return remapped

# ---------- Drawing (scale by box size) ----------
def draw_box(img_bgr, xywh, text):
    x, y, w, h = map(int, xywh)
    x1, y1, x2, y2 = x, y, x + w, y + h
    H, W = img_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W - 1, x2), min(H - 1, y2)

    # Scale using box height so labels stay readable for small faces in big images
    font_scale = max(0.5, min(2.5, h / 180.0))
    thickness  = max(2, min(6, int(h / 120.0)))
    pad_y      = max(6, int(8 * font_scale))

    cv2.rectangle(img_bgr, (x1,y1), (x2,y2), (0,255,0), thickness)

    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    bg_x1, bg_y1 = x1, max(0, y1 - th - pad_y)
    bg_x2, bg_y2 = x1 + tw + 10, y1
    cv2.rectangle(img_bgr, (bg_x1, bg_y1), (bg_x2, bg_y2), (0,255,0), -1)
    cv2.putText(img_bgr, text, (x1 + 5, y1 - baseline - 4),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0),
                max(1, thickness-1), cv2.LINE_AA)

# ---------- Detection ----------
def to_pil(face_bgr):
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def clip_xywh(x, y, w, h, W, H):
    x = max(0, x); y = max(0, y)
    w = max(0, min(w, W - x)); h = max(0, min(h, H - y))
    return x, y, w, h

def valid_box(x, y, w, h, W, H):
    """Filter out boxes that are too small/large or have odd aspect ratios."""
    if w <= 0 or h <= 0: return False
    area = w * h
    frac = area / float(W * H)
    if frac < 0.001 or frac > 0.6:  # too tiny or almost the whole frame
        return False
    ar = w / float(h)
    if ar < 0.4 or ar > 2.5:        # extremely tall/wide unlikely for faces
        return False
    return True

def detect_faces(frame_bgr, backend):
    """Return list of dicts: {'xywh':(x,y,w,h), 'face_pil':PIL.Image}."""
    H, W = frame_bgr.shape[:2]

    # 1) Try DeepFace backend if requested (works with np arrays)
    if backend in {"mediapipe", "retinaface", "mtcnn", "opencv"}:
        try:
            from deepface import DeepFace
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            faces = DeepFace.extract_faces(
                img_path=frame_rgb,
                detector_backend=backend,
                align=True,
                enforce_detection=False
            )
            out = []
            for f in faces:
                area = f.get("facial_area", {})
                x = int(area.get("x", 0)); y = int(area.get("y", 0))
                w = int(area.get("w", 0)); h = int(area.get("h", 0))
                x, y, w, h = clip_xywh(x, y, w, h, W, H)
                if not valid_box(x, y, w, h, W, H):
                    continue
                crop = frame_bgr[y:y+h, x:x+w]
                out.append({"xywh": (x, y, w, h), "face_pil": to_pil(crop)})
            if out:
                return out
        except Exception as e:
            log(f"[info] DeepFace failed/absent ({e}). Falling back to OpenCV Haar.")

    # 2) Fallback: OpenCV Haar
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    dets = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(40,40))
    out = []
    for (x, y, w, h) in dets:
        x, y, w, h = clip_xywh(int(x), int(y), int(w), int(h), W, H)
        if not valid_box(x, y, w, h, W, H):
            continue
        crop = frame_bgr[y:y+h, x:x+w]
        out.append({"xywh": (x, y, w, h), "face_pil": to_pil(crop)})
    return out

# ---------- Utils ----------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VID_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

def is_webcam_token(s):
    if isinstance(s, int): return True
    if isinstance(s, str) and s.isdigit(): return True
    return False

def detect_input_type(inp):
    if is_webcam_token(inp):
        return "webcam"
    p = Path(inp)
    if not p.exists():
        ext = p.suffix.lower()
        if ext in IMG_EXTS: return "image"
        if ext in VID_EXTS: return "video"
        return "unknown"
    if p.is_file():
        ext = p.suffix.lower()
        if ext in IMG_EXTS: return "image"
        if ext in VID_EXTS: return "video"
        cap = cv2.VideoCapture(str(p))
        ok, _ = cap.read()
        cap.release()
        return "video" if ok else "image"
    return "unknown"

def resize_for_speed(frame_bgr, max_long_edge):
    if not max_long_edge:
        return frame_bgr, 1.0
    h, w = frame_bgr.shape[:2]
    long_edge = max(h, w)
    if long_edge <= max_long_edge:
        return frame_bgr, 1.0
    scale = max_long_edge / float(long_edge)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized, scale

# ------ Display-only resize (consistent preview size) ------
def resize_for_display(frame_bgr, target_long_edge=DISPLAY_LONG_EDGE):
    h, w = frame_bgr.shape[:2]
    long_edge = max(h, w)
    if long_edge == 0:
        return frame_bgr
    scale = target_long_edge / float(long_edge)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    if new_w <= 0 or new_h <= 0:
        return frame_bgr
    return cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

def show_preview(win_name, frame_bgr):
    disp = resize_for_display(frame_bgr, DISPLAY_LONG_EDGE)
    cv2.imshow(win_name, disp)

# ---------- Image pipeline ----------
def run_on_image(img_path, model, classes, tfm, device):
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        sys.exit(f"Failed to read image: {img_path}")

    faces = detect_faces(img_bgr, BACKEND)

    if not faces:
        log("No faces detected.")
        if SHOW_WINDOW:
            show_preview(WINDOW_NAME, img_bgr); cv2.waitKey(0); cv2.destroyAllWindows()
        return

    log(f"Detected {len(faces)} face(s). Classifying…")
    crops_dir = Path("faces") / Path(img_path).stem
    if SAVE_CROPS: crops_dir.mkdir(parents=True, exist_ok=True)

    # CSV log
    csv_path = Path(img_path).with_suffix("").as_posix() + "_detections.csv"
    csv_file = open(csv_path, "w", newline="") if WRITE_CSV_LOG else None
    writer = csv.writer(csv_file) if csv_file else None
    if writer: writer.writerow(["frame", "face_idx", "x", "y", "w", "h", "label", "confidence"])

    # Batch classify
    pil_list = [f["face_pil"] for f in faces]
    preds = classify_batch(model, classes, tfm, device, pil_list)

    for i, (item, (label, conf)) in enumerate(zip(faces, preds), 1):
        x, y, w, h = item["xywh"]
        log(f"[Face {i}] {label:12s} conf={conf*100:.1f}% xywh=({x},{y},{w},{h})")
        if SAVE_CROPS:
            item["face_pil"].save(crops_dir / f"face_{i:02d}.png")
        draw_box(img_bgr, (x, y, w, h), f"{label} ({conf*100:.1f}%)")
        if writer: writer.writerow([0, i, x, y, w, h, label, f"{conf:.4f}"])

    if csv_file:
        csv_file.close(); log(f"Wrote CSV -> {csv_path}")

    if SAVE_OVERLAY:
        out_path = Path(img_path).with_suffix("").as_posix() + "_pred.png"
        cv2.imwrite(out_path, img_bgr)
        log(f"Saved overlay -> {out_path}")

    if SHOW_WINDOW:
        show_preview(WINDOW_NAME, img_bgr); cv2.waitKey(0); cv2.destroyAllWindows()

# ---------- Video/Webcam pipeline ----------
def run_on_video(video_source, model, classes, tfm, device):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        sys.exit(f"Could not open video source: {video_source}")

    src_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fps_out = OUT_FPS if OUT_FPS is not None else src_fps

    if is_webcam_token(video_source):
        out_path = Path("webcam_pred.mp4")
        csv_path = Path("webcam_detections.csv")
        crops_dir = Path("faces") / "webcam"
    else:
        vp = Path(str(video_source))
        out_path = Path(vp.with_suffix("").as_posix() + "_pred.mp4")
        csv_path = Path(vp.with_suffix("").as_posix() + "_detections.csv")
        crops_dir = Path("faces") / vp.stem

    dummy = np.zeros((max(1,src_h), max(1,src_w), 3), np.uint8)
    dummy_resized, _ = resize_for_speed(dummy, MAX_LONG_EDGE)
    out_h, out_w = dummy_resized.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps_out, (out_w, out_h))
    if not writer.isOpened():
        sys.exit("Failed to open VideoWriter (codec/permissions?).")

    if SAVE_CROPS: crops_dir.mkdir(parents=True, exist_ok=True)

    csv_file = open(csv_path, "w", newline="") if WRITE_CSV_LOG else None
    writer_csv = csv.writer(csv_file) if csv_file else None
    if writer_csv: writer_csv.writerow(["frame", "face_idx", "x", "y", "w", "h", "label", "confidence"])

    log(f"Source: {video_source} | {src_w}x{src_h}@{src_fps:.2f}fps")
    log(f"Output: {out_path} | {out_w}x{out_h}@{fps_out:.2f}fps")
    log(f"Backend: {BACKEND} | Frame stride: {FRAME_STRIDE}")

    frame_idx = 0
    t0 = time.time()

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_idx += 1

        if FRAME_STRIDE > 1 and (frame_idx % FRAME_STRIDE) != 0:
            resized, _ = resize_for_speed(frame_bgr, MAX_LONG_EDGE)
            writer.write(resized)
            if SHOW_WINDOW:
                show_preview(WINDOW_NAME, resized)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
            continue

        frame_proc, _ = resize_for_speed(frame_bgr, MAX_LONG_EDGE)

        faces = detect_faces(frame_proc, BACKEND)

        # Batch classify
        pil_list = [f["face_pil"] for f in faces]
        preds = classify_batch(model, classes, tfm, device, pil_list)

        for i, (item, (label, conf)) in enumerate(zip(faces, preds), 1):
            x, y, w, h = item["xywh"]
            log(f"[Frame {frame_idx} | Face {i}] {label:12s} conf={conf*100:.1f}% xywh=({x},{y},{w},{h})")

            if SAVE_CROPS:
                item["face_pil"].save(crops_dir / f"frame_{frame_idx:06d}_face_{i:02d}.png")

            draw_box(frame_proc, (x, y, w, h), f"{label} ({conf*100:.1f}%)")

            if writer_csv:
                writer_csv.writerow([frame_idx, i, x, y, w, h, label, f"{conf:.4f}"])

        writer.write(frame_proc)

        if SHOW_WINDOW:
            show_preview(WINDOW_NAME, frame_proc)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    writer.release()
    if SHOW_WINDOW: cv2.destroyAllWindows()
    if csv_file:
        csv_file.close(); log(f"Wrote CSV -> {csv_path}")
    dt = time.time() - t0
    log(f"Done. Wrote -> {out_path} | Frames: {frame_idx} | Time: {dt:.1f}s")

# ---------- Entry ----------
def main():
    # Load your model
    ckpt_path = Path(CKPT_PATH)
    if not ckpt_path.exists():
        sys.exit(f"Checkpoint not found: {ckpt_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model, classes, img_size = build_model(ckpt, device)
    tfm = make_transform(img_size)
    log(f"Classes: {classes}")

    # Decide input type
    inp = INPUT
    kind = detect_input_type(inp)
    if kind == "unknown":
        sys.exit(f"Could not determine input type for '{inp}'. Use an image path, video path, or 0 for webcam.")

    if kind == "image":
        log(f"[mode=image] INPUT={inp}")
        run_on_image(inp, model, classes, tfm, device)
    elif kind == "video":
        log(f"[mode=video] INPUT={inp}")
        run_on_video(str(inp), model, classes, tfm, device)
    elif kind == "webcam":
        log(f"[mode=webcam] INPUT={inp}")
        src = int(inp) if isinstance(inp, str) and inp.isdigit() else (inp if isinstance(inp, int) else 0)
        run_on_video(src, model, classes, tfm, device)

if __name__ == "__main__":
    main()
