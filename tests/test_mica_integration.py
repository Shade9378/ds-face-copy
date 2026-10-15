import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from preprocessing.face_detection import MediaPipeFaceDetector
from preprocessing.pipeline import prepare_face
from reconstruction.export import export_reconstruction
from reconstruction.mica_model import MicaModel, mica_config_from_environment


@pytest.mark.parametrize("image_variable", ["FACE_TEST_PORTRAIT", "FACE_TEST_NONFRONTAL"])
def test_real_mica_reconstruction(image_variable, tmp_path):
    if os.environ.get("RUN_MICA_INTEGRATION") != "1":
        pytest.skip("Set RUN_MICA_INTEGRATION=1 to enable local MICA inference tests.")
    photograph = os.environ.get(image_variable)
    face_model = os.environ.get("FACE_LANDMARKER_MODEL")
    if not photograph or not face_model:
        pytest.skip(f"Set {image_variable} and FACE_LANDMARKER_MODEL to local test assets.")
    source = Path(photograph)
    digest_before = hashlib.sha256(source.read_bytes()).hexdigest()
    detector = MediaPipeFaceDetector(Path(face_model))
    try:
        prepared = prepare_face(source, detector)
    finally:
        detector.close()
    if image_variable == "FACE_TEST_NONFRONTAL":
        assert prepared.report.pose is not None
        assert abs(prepared.report.pose.yaw_degrees) >= 15
    reconstructed = MicaModel(mica_config_from_environment()).reconstruct(prepared)
    assert reconstructed.geometry.vertices.shape == (5023, 3)
    assert reconstructed.geometry.landmarks.shape == (68, 3)
    assert reconstructed.geometry.identity_parameters.shape == (300,)
    assert np.isfinite(reconstructed.geometry.vertices).all()
    assert reconstructed.report.units == "millimeters"
    assert reconstructed.report.status == "experimental"
    assert reconstructed.preview.shape == (512, 512, 3)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest_before
    prepared.report.reconstruction_status = "experimental: MICA canonical mesh"
    files = export_reconstruction(reconstructed, tmp_path, prepared.report)
    assert all(path.is_file() and path.stat().st_size > 0 for path in files)
