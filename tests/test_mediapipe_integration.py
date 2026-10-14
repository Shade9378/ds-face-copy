import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from preprocessing.face_detection import MediaPipeFaceDetector
from preprocessing.pipeline import prepare_face


@pytest.fixture
def detector():
    model_path = os.environ.get("FACE_LANDMARKER_MODEL")
    if not model_path:
        pytest.skip("Set FACE_LANDMARKER_MODEL to run real model inference tests.")
    backend = MediaPipeFaceDetector(Path(model_path))
    try:
        yield backend
    finally:
        backend.close()


def test_blank_image_has_no_face(detector):
    assert detector.detect_faces(np.zeros((256, 256, 3), dtype=np.uint8)) == []


def test_local_portrait_pipeline(detector):
    portrait_path = os.environ.get("FACE_TEST_PORTRAIT")
    if not portrait_path:
        pytest.skip("Set FACE_TEST_PORTRAIT to a consented local portrait.")
    result = prepare_face(portrait_path, detector)
    assert result.report.detected_faces >= 1
    assert len(result.report.landmarks) >= 468
    assert result.crop.size > 0
    assert result.report.pose is not None
    with Image.open(portrait_path) as image:
        assert result.report.image_width > 0
        assert image.width > 0
