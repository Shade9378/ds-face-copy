# Face-to-Sliders

Local face analysis and 3D reconstruction. Slider generation is still in development.

## Setup

- Install Python 3.12 and [uv](https://docs.astral.sh/uv/).
- Ubuntu/WSL: run `sudo apt install libgles2 libegl1`.
- Download the [MediaPipe Face Landmarker model](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python) to `models/face_landmarker.task`.

```bash
uv sync --extra dev
```

## Run

```bash
uv run face-to-slider
```

Open the terminal's local URL, upload a photo, and click **Analyze face**.

## Reconstruction

- Follow the [MICA setup guide](docs/mica_setup.md).
- Optional: follow the [3DDFA-V2 setup guide](docs/tddfa_setup.md) and enable **Allow 3DDFA-V2 fallback**.
- With the configured local installation, start the app using:

```bash
source models/local.env
uv run face-to-slider
```

Click **Reconstruct face** for a preview and geometry downloads.
