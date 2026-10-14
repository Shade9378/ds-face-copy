import os
from pathlib import Path
from uuid import uuid4

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr

from preprocessing.face_detection import MediaPipeFaceDetector
from preprocessing.models import AnalysisConfig
from preprocessing.pipeline import PreparedFace, prepare_face
from reconstruction.export import create_reconstruction_archive, export_reconstruction
from reconstruction.mica_model import MicaModel, mica_config_from_environment

GAME_NAMES = ["Elden Ring", "Dark Souls 1", "Dark Souls 2", "Dark Souls 3", "Bloodborne"]


def prepare_photograph(image_path: str | None, upscale: bool = False) -> PreparedFace:
    if image_path is None:
        raise ValueError("Upload a photograph first.")
    config = AnalysisConfig()
    model_path = Path(os.environ.get("FACE_LANDMARKER_MODEL", "models/face_landmarker.task"))
    detector = MediaPipeFaceDetector(model_path, config.max_faces)
    try:
        return prepare_face(image_path, detector, config, upscale)
    finally:
        detector.close()


def analyze_photograph(image_path: str | None, game_name: str, upscale: bool) -> tuple:
    try:
        prepared = prepare_photograph(image_path, upscale)
    except (ValueError, RuntimeError) as error:
        raise gr.Error(str(error)) from error
    status = f"{game_name}: slider generation unsupported — calibration data is needed."
    return prepared.crop, prepared.preview, prepared.report.model_dump(mode="json"), status


def reconstruct_photograph(image_path: str | None, game_name: str) -> tuple:
    try:
        config = mica_config_from_environment()
        prepared = prepare_photograph(image_path)
        reconstructed = MicaModel(config).reconstruct(prepared)
        prepared.report.reconstruction_status = "experimental: MICA canonical mesh"
        output_root = Path(os.environ.get("FACE_OUTPUT_DIR", "outputs"))
        output_directory = output_root / uuid4().hex
        files = export_reconstruction(reconstructed, output_directory, prepared.report)
        archive_path = create_reconstruction_archive(files, output_directory / "reconstruction.zip")
    except (ValueError, RuntimeError, OSError) as error:
        raise gr.Error(str(error)) from error
    status = (
        "MICA reconstruction complete. Check the preview and quality warnings. "
        f"{game_name}: slider generation unsupported — calibration data is needed."
    )
    return (
        reconstructed.preview,
        reconstructed.report.model_dump(mode="json"),
        str(archive_path),
        status,
    )


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
        reconstruct = gr.Button("Reconstruct with MICA")
        mesh_preview = gr.Image(label="Frontal mesh preview")
        reconstruction_report = gr.JSON(label="Reconstruction")
        download = gr.File(label="Download mesh, landmarks, identity parameters and report")
        reconstruction_status = gr.Textbox(label="Reconstruction status", interactive=False)
        analyze.click(
            analyze_photograph, [photograph, game, upscale], [crop, preview, report, status]
        )
        reconstruct.click(
            reconstruct_photograph,
            [photograph, game],
            [mesh_preview, reconstruction_report, download, reconstruction_status],
            concurrency_limit=1,
        )
        photograph.change(
            lambda: (None, None, None, "", None, None, None, ""),
            outputs=[
                crop,
                preview,
                report,
                status,
                mesh_preview,
                reconstruction_report,
                download,
                reconstruction_status,
            ],
        )
    return app


def main() -> None:
    create_app().launch(server_name="127.0.0.1", share=False)
