# MICA setup

MICA runs in its own environment. The Gradio app sends the native face crop to its
official demo and reads the mesh, identity parameters and 68 FLAME landmarks.

1. Review the [MICA license](https://github.com/Zielon/MICA/blob/master/LICENSE).
   MICA restricts commercial use and redistribution. FLAME and InsightFace weights
   have separate terms. This repository includes none of those assets.
2. Get [MICA](https://github.com/Zielon/MICA) in a folder outside this repository.
   Use its documented environment and pretrained inference setup, including its
   submodules. The upstream environment uses Python 3.9 and PyTorch 1.13 with CUDA;
   keep it separate from this project's Python 3.12 environment.
3. Obtain the pretrained MICA checkpoint and the FLAME2020 assets through the
   [upstream instructions](https://github.com/Zielon/MICA/blob/master/README.md).
   Use the paths in the checkout's `configs/config.py`; current source expects
   `data/FLAME2020/`, even though some README text mentions `data/pretrained/FLAME2020/`.
4. Install the antelopev2 InsightFace models at `~/.insightface/models/antelopev2/`
   for the same user running the app. The adapter requires local ONNX files before
   starting MICA. Its detector uses these files; missing assets aren't downloaded by the app.
5. Use trusted upstream checkpoint files. Before inference, the adapter checks CUDA
   availability and requires both ArcFace and FLAME model weights in the checkpoint.
6. Test the official `demo.py` in that environment first. CUDA support and a working
   ONNX Runtime installation are required by the upstream demo.

Required checkout assets:

```text
data/pretrained/mica.tar
data/FLAME2020/generic_model.pkl
data/FLAME2020/landmark_embedding.npy
data/FLAME2020/head_template.obj
data/FLAME2020/FLAME_masks/FLAME_masks.pkl
```

Point the app at the checkout and its environment:

```bash
export MICA_REPO=/absolute/path/to/MICA
export MICA_PYTHON=/absolute/path/to/mica/environment/bin/python
uv run face-to-slider
```

Set `MICA_CHECKPOINT` to override `data/pretrained/mica.tar`.
Click **Reconstruct with MICA** after uploading a photograph.
Reconstruction downloads contain OBJ geometry in millimeters, landmarks and identity arrays,
an untextured frontal preview, and a JSON report. UI exports remain in `outputs/`;
set `FACE_OUTPUT_DIR` to change that folder. Temporary source crops are deleted after inference.

Command-line usage:

```bash
uv run python scripts/reconstruct_photo.py portrait.jpg \
  --face-model models/face_landmarker.task \
  --mica-repo /path/to/MICA \
  --mica-python /path/to/mica/environment/bin/python \
  --output outputs/portrait
```

This integration is experimental. Software tests cover the subprocess contract, geometry,
rendering, error handling and export. Real MICA inference has also passed smoke tests on
a frontal portrait and an upstream example with roughly 23 degrees of estimated yaw.
These checks establish that reconstruction runs; they do not measure likeness accuracy.
The preview uses canonical geometry without skin texture. Feature distances are normalized
by eye-center spacing; they are not game slider values. Forehead measurements, automatic
occlusion scoring, 3DDFA-V2 fallback and game calibration are still pending.


## Check setup

The app's **MICA setup** panel lists missing assets and can inspect Python dependencies
and CUDA without reconstructing a photo. The command-line equivalent is:

```bash
uv run python scripts/check_mica_setup.py
uv run python scripts/check_mica_setup.py --runtime
```

Use --mica-repo and --mica-python to check a setup without environment variables.
The command exits with status 1 when setup is incomplete or runtime checks fail.
Asset checks alone do not load models. Even a successful runtime check does not establish
that real inference works; test a consented photograph after installing the assets.

Reconstruction reports include input quality flags. Low resolution, possible blur,
exposure problems, failed or large pose estimates, cropped faces and multiple faces trigger
review_required. An input without those flags is still unassessed for reconstruction accuracy.
Occlusion detection is not implemented and is reported separately as unassessed.


## Local installation

This machine's installed environment is configured in the ignored models/local.env file.
Start the app from the project folder:

```bash
source models/local.env
uv run face-to-slider
```

The local MICA environment uses Python 3.9.25, PyTorch 1.13.1 with CUDA 11.7,
and the RTX 4060. The RTX 5070 is not supported by that legacy PyTorch build.
Dependency versions are recorded in models/mica_requirements.txt.

The official checkpoint includes optimizer and training metadata that the legacy
weights-only loader cannot read. The local mica_weights.tar contains just the original
ArcFace and FLAME state dictionaries. Every tensor was checked against the trusted
official download before using it. No model training was performed.

The user-supplied FLAME2020 generic_model.pkl is installed in MICA's data/FLAME2020 folder.
The original licensed download folder, model checkout, environments and generated outputs
are excluded from Git.

Run real-model tests explicitly:

```bash
source models/local.env
RUN_MICA_INTEGRATION=1 \
FACE_TEST_PORTRAIT=/absolute/path/to/frontal.jpg \
FACE_TEST_NONFRONTAL=/absolute/path/to/angled.jpg uv run pytest
```

Use a non-frontal test image with at least 15 degrees of estimated yaw.
The default suite skips real MICA inference unless enabled.
