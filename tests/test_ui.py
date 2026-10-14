"""UI construction and explicit unsupported-status checks."""

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
