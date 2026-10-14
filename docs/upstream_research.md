# Upstream investigation — 2026-10-14

This review guides integration; no upstream implementation or restricted weights are vendored.

- [MediaPipe Face Landmarker Python](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python)
  documents the Tasks API, local model paths, image mode and multiple-face output.
  Use Tasks rather than legacy Face Mesh. Its relative landmarks are not metrical identity geometry.
- [MICA README](https://github.com/Zielon/MICA/blob/master/README.md) documents pretrained MICA,
  FLAME2020, InsightFace assets and mesh/render output. Its original environment must be checked
  independently for modern Python/PyTorch compatibility.
  [MICA license](https://github.com/Zielon/MICA/blob/master/LICENSE) restricts use to specified
  non-commercial purposes and restricts distribution. Review FLAME and each InsightFace model's
  separate terms. Do not redistribute assets or assume commercial permission.
- [3DDFA-V2](https://github.com/cleardusk/3DDFA_V2) is an upstream dense-alignment and pose fallback
  candidate. The [code license](https://github.com/cleardusk/3DDFA_V2/blob/master/LICENSE) is MIT;
  that alone does not establish permissions for every pretrained/BFM asset.
- [Elden Sliders Tool source](https://github.com/krisuuhh/Elden-Sliders-Tool/blob/main/sliders.js)
  illustrates category names and numeric ranges but maps measurements with heuristic functions.
  Those constants are not a validated forward model or paired in-game calibration dataset.
  Do not adopt its accuracy claims or copy its mappings.
- [Character Preset Manager](https://github.com/Hapfel1/character-presets-tool)
  documents JSON and save-based appearance presets. Treat its schema as a research candidate:
  independently verify slider identities, versions, categorical bases and export interoperability.
  No save operations are used here.
- [Elden Bling Auto Sliders](https://github.com/fazedankinbank/EldenBlingAutoSliders)
  describes screenshot/OCR extraction and keyboard import; it explicitly says its code is not
  open sourced. It may inform a manual collection workflow but is not a reusable implementation.

Neither save-format parameters nor a list of slider names establishes how controls affect geometry.
Verified screenshots, template choices and held-out tests are required before generation is enabled.
