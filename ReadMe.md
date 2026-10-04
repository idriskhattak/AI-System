# Human Insight AI

A Flask application for running several computer-vision models over an uploaded
video behind one interface. Upload a file, pick a model, watch annotated frames come
back as a live MJPEG stream.

## Why it exists

I had written a handful of one-off detector scripts, each with its own loop, its own
output format and its own way of being run. This project is the attempt to put them
behind a single interface so that switching models does not mean switching programs.

The design decision worth pointing at is the **model factory**: `load_model()` returns
a detector object, and every detector exposes the same prediction shape to the
streaming pipeline. The pipeline does not know or care which model it is driving.
Adding a sixth detector means writing one class, not touching the video path.

## Detectors

| Model | Weights | Notes |
| --- | --- | --- |
| Human detection | `yolov8s.pt` | Person detection per frame |
| Fire detection | `yolov5s_best.pt` | Custom-trained YOLOv5 |
| Gender classification | `gender_resnet18_best.pth` | ResNet-18 |
| Ethnicity classification | `fairface_resnet18.pt` | ResNet-18 on FairFace — **see the note below** |
| Fight detection | `fight_model.pth` | Needs full-video context, so it takes a separate path |

Four of the five run frame by frame. Fight detection needs temporal context across
the whole clip, so it bypasses the per-frame loop entirely — that split is the one
place the uniform interface breaks down, and it is deliberate.

## Running it

```bash
pip install flask opencv-python torch torchvision ultralytics
python main-app.py
```

Then open `http://127.0.0.1:5000`.

| Route | Method | Purpose |
| --- | --- | --- |
| `/` | GET | Upload form and model selector |
| `/upload` | POST | Accepts the video file |
| `/stream` | GET | MJPEG stream of annotated frames |

Model weights live in `weights/` and are not downloaded automatically.

## Layout

```
main-app.py       Flask app, routes, frame loop, model factory
detectors/        One class per model, all exposing the same interface
weights/          Model weights
templates/        Upload page and stream view
training/         Training notebooks for the custom models
project_utils/    Shared helpers
UtilsFiles/       Supporting assets
uploads/          Runtime upload target
```

## Known limitations

This is a systems prototype, not a deployment.

- **No benchmark suite.** No accuracy, precision or recall figures for any of the five
  detectors, and no measured throughput. I do not currently know how fast it runs on
  hardware other than my own.
- **No `requirements.txt`.** The install line above is reconstructed from imports;
  versions are unpinned.
- **The fire detector depends on a machine-specific YOLOv5 path** and will not load
  cleanly on a fresh checkout.
- **Single-video, single-model at a time.** No queue, no concurrency, no auth.

## A note on the gender and ethnicity detectors

Two of these classify people by gender and by ethnicity. I built them as part of
exploring what the shared interface could carry, and I would remove both before this
went anywhere real.

The reasoning is the same as in
[the write-up for my final-year project](https://idriskhattak.github.io/Idris_Portfolio/projects/people-counting/):
a model that assigns a categorical demographic label from appearance returns a
confident answer for every face it is shown, its error is not evenly distributed
across the groups it claims to distinguish, and its categories are whichever ones its
training data happened to use. FairFace is a more carefully constructed dataset than
most, which changes the size of the problem and not its nature.

They stay in the repository because this is a record of what I built. They are not a
feature I would ship.
