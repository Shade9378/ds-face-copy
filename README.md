# Face-to-Sliders

Local face analysis for a FromSoftware character-creator project. Slider generation is still in development.

## Setup

Install Python 3.12 and [uv](https://docs.astral.sh/uv/), then run:

```bash
uv sync --extra dev
```

On Ubuntu or WSL:

```bash
sudo apt install libgles2 libegl1
```

Download the Face Landmarker model from the
[MediaPipe setup guide](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python)
and save it as `models/face_landmarker.task`.

## Run

```bash
uv run face-to-slider
```

Open the local URL printed in the terminal, upload a photo, and click **Analyze face**.

For a model stored elsewhere:

```bash
FACE_LANDMARKER_MODEL=/path/to/face_landmarker.task uv run face-to-slider
```

## MICA reconstruction

Set up MICA in a separate CUDA environment using [these instructions](docs/mica_setup.md),
then set MICA_REPO and MICA_PYTHON before starting the app.
Use **Reconstruct with MICA** to generate a mesh preview and download the geometry.


For the configured local installation:

```bash
source models/local.env
uv run face-to-slider
```
