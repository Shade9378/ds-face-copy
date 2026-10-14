from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ui import app


def test_app_builds():
    assert app.create_app() is not None


def test_ui_requires_upload():
    with pytest.raises(app.gr.Error, match="Upload"):
        app.analyze_photograph(None, "Elden Ring", False)


def test_ui_closes_detector_and_labels_outputs(monkeypatch, tmp_path):
    closed = []

    class StubDetector:
        def __init__(self, model_path, max_faces):
            pass

        def detect_faces(self, image):
            points = np.zeros((478, 3))
            points[:, 0] = np.linspace(10, 90, 478)
            points[:, 1] = np.linspace(10, 90, 478)
            return [points]

        def close(self):
            closed.append(True)

    monkeypatch.setattr(app, "MediaPipeFaceDetector", StubDetector)
    image_path = tmp_path / "input.png"
    Image.new("RGB", (100, 100), "gray").save(image_path)
    crop, preview, report, status = app.analyze_photograph(str(image_path), "Bloodborne", True)
    assert crop.shape[0] < preview.shape[0]
    assert report["detected_faces"] == 1
    assert "Bloodborne: slider generation unsupported" in status
    assert closed == [True]


def test_reconstruction_requires_mica_configuration(monkeypatch):
    monkeypatch.delenv("MICA_REPO", raising=False)
    monkeypatch.delenv("MICA_PYTHON", raising=False)
    with pytest.raises(app.gr.Error, match="MICA_REPO and MICA_PYTHON"):
        app.reconstruct_photograph("portrait.png", "Elden Ring")


def test_reconstruction_ui_exports_result(monkeypatch, tmp_path):
    from preprocessing.models import AnalysisConfig
    from preprocessing.pipeline import prepare_face
    from reconstruction.face_geometry import extract_geometric_features
    from reconstruction.models import FaceGeometry, ReconstructedFace, ReconstructionReport
    from reconstruction.preview import render_frontal_preview

    class StubDetector:
        def detect_faces(self, image):
            points = np.zeros((478, 3))
            points[:, 0] = np.linspace(10, 90, 478)
            points[:, 1] = np.linspace(10, 90, 478)
            return [points]

    prepared = prepare_face(Image.new("RGB", (100, 100), "gray"), StubDetector(), AnalysisConfig())
    landmarks = np.zeros((68, 3))
    landmarks[36:42, 0] = -20
    landmarks[42:48, 0] = 20
    geometry = FaceGeometry(
        vertices=np.array([[-30.0, -30, 0], [30.0, -30, 0], [0.0, 30, 10]]),
        triangles=np.array([[0, 1, 2]]),
        landmarks=landmarks,
        identity_parameters=np.zeros(300),
    )
    reconstructed = ReconstructedFace(
        geometry=geometry,
        preview=render_frontal_preview(geometry, 64),
        report=ReconstructionReport(
            vertex_count=3,
            triangle_count=1,
            landmark_count=68,
            features=extract_geometric_features(geometry),
            warnings=["synthetic fixture"],
            checkpoint_sha256="synthetic fixture",
        ),
    )

    class StubMica:
        def __init__(self, config):
            pass

        def reconstruct(self, face):
            assert face is prepared
            return reconstructed

    monkeypatch.setattr(app, "mica_config_from_environment", lambda: object())
    monkeypatch.setattr(app, "prepare_photograph", lambda image_path: prepared)
    monkeypatch.setattr(app, "MicaModel", StubMica)
    monkeypatch.setenv("FACE_OUTPUT_DIR", str(tmp_path))
    preview, report, archive, status = app.reconstruct_photograph("input.png", "Elden Ring")
    assert preview.shape == (64, 64, 3)
    assert report["backend"] == "MICA"
    assert "Elden Ring: slider generation unsupported" in status
    assert prepared.report.reconstruction_status == "experimental: MICA canonical mesh"
    assert (tmp_path / Path(archive).parent.name / "reconstruction.zip").is_file()
