import os
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr

from preprocessing.face_detection import MediaPipeFaceDetector
from preprocessing.models import AnalysisConfig
from preprocessing.pipeline import prepare_face

GAME_NAMES = ["Elden Ring", "Dark Souls 1", "Dark Souls 2", "Dark Souls 3", "Bloodborne"]


def analyze_photograph(image_path: str | None, game_name: str, upscale: bool) -> tuple:
    if image_path is None:
        raise gr.Error("Upload a photograph first.")
    config = AnalysisConfig()
    model_path = Path(os.environ.get("FACE_LANDMARKER_MODEL", "models/face_landmarker.task"))
    detector = None
    try:
        detector = MediaPipeFaceDetector(model_path, config.max_faces)
        prepared = prepare_face(image_path, detector, config, upscale)
    except (ValueError, RuntimeError) as error:
        raise gr.Error(str(error)) from error
    finally:
        if detector is not None:
            detector.close()
    status = (
        f"{game_name}: slider generation unsupported — no verified calibration dataset. "
        "3D reconstruction and frontal preview are not implemented. "
        "Displayed images are photograph crops."
    )
    return prepared.crop, prepared.preview, prepared.report.model_dump(mode="json"), status


def create_app() -> gr.Blocks:
    with gr.Blocks(title="Face-to-Sliders") as app:
        gr.Markdown("# Face-to-Sliders\nUpload a face photo. Sliders are still in development.")
        photograph = gr.Image(type="filepath", label="Photograph")
        game = gr.Dropdown(GAME_NAMES, value="Elden Ring", label="Target game")
        upscale = gr.Checkbox(label="Enlarge small crops")
        analyze = gr.Button("Analyze face", variant="primary")
        with gr.Row():
            crop = gr.Image(label="Native-resolution crop")
            preview = gr.Image(label="Resized crop")
        report = gr.JSON(label="Face analysis")
        status = gr.Textbox(label="Game support", interactive=False)
        analyze.click(
            analyze_photograph, [photograph, game, upscale], [crop, preview, report, status]
        )
    return app


def main() -> None:
    create_app().launch(server_name="127.0.0.1", share=False)
