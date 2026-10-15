import cv2
import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from games.base_adapter import UnsupportedGenerationError
from games.elden_ring import EldenRingAdapter
from preprocessing.face_detection import MediaPipeFaceDetector, select_largest_face
from preprocessing.image_processing import crop_face, load_image, upscale_crop
from preprocessing.models import AnalysisConfig, AnalysisReport, FaceBounds
from preprocessing.pipeline import prepare_face
from preprocessing.pose_estimation import (
    GENERIC_FACE_POINTS,
    POSE_LANDMARK_INDICES,
    estimate_head_pose,
)


def face_points(left=20, top=20, right=80, bottom=80):
    points = np.zeros((478, 3), dtype=np.float64)
    points[:, 0] = np.linspace(left, right, len(points))
    points[:, 1] = np.linspace(top, bottom, len(points))
    return points


class StubDetector:
    def __init__(self, faces):
        self.faces = faces

    def detect_faces(self, image):
        return self.faces


def test_largest_face_selection_and_invalid_candidates():
    small, large = face_points(), face_points(10, 5, 95, 90)
    invalid = large.copy()
    invalid[0, 0] = np.nan
    points, bounds = select_largest_face([small, invalid, large], (100, 100, 3))
    np.testing.assert_array_equal(points, large)
    assert bounds == FaceBounds(left=10, top=5, right=95, bottom=90)


@pytest.mark.parametrize("faces", [[], [np.zeros((4, 3))], [face_points(120, 120, 130, 130)]])
def test_no_usable_face(faces):
    with pytest.raises(ValueError, match="No usable face"):
        select_largest_face(faces, (100, 100, 3))


def test_crop_clamps_padding_and_does_not_mutate_source():
    original = np.full((100, 120, 3), 128, np.uint8)
    crop, bounds = crop_face(original, FaceBounds(left=0, top=0, right=50, bottom=50), 0.2)
    assert bounds == FaceBounds(left=0, top=0, right=60, bottom=60)
    crop[:] = 0
    assert original.min() == 128


def test_load_exif_orientation_and_rgb_preservation():
    image = Image.new("RGB", (10, 20), (20, 40, 60))
    image.getexif()[274] = 6
    loaded = load_image(image, 1000)
    assert loaded.shape == (10, 20, 3)
    np.testing.assert_array_equal(loaded[0, 0], [20, 40, 60])
    assert image.size == (10, 20)


def test_transparent_pixels_composited_on_white():
    loaded = load_image(Image.new("RGBA", (2, 2), (0, 0, 0, 0)), 100)
    assert loaded.min() == 255


def test_pixel_limit_and_invalid_file(tmp_path):
    with pytest.raises(ValueError, match="pixel limit"):
        load_image(Image.new("RGB", (10, 10)), 99)
    path = tmp_path / "bad.jpg"
    path.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="decode"):
        load_image(path, 1000)


def test_upscale_bounded_and_never_downsamples():
    crop = np.zeros((10, 20, 3), np.uint8)
    preview, factor = upscale_crop(crop, 512, 4)
    assert factor == 4
    assert preview.shape == (40, 80, 3)
    large = np.zeros((600, 800, 3), np.uint8)
    preview, factor = upscale_crop(large, 512, 4)
    assert preview.shape == large.shape
    assert factor == 1


@pytest.mark.parametrize("angles", [(0, 0, 0), (15, 30, -10), (-10, -40, 20)])
def test_pose_recovers_synthetic_camera_rotation(angles):
    pitch, yaw, roll = np.deg2rad(angles)
    rotation_x = np.array(
        [[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]]
    )
    rotation_y = np.array(
        [[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]]
    )
    rotation_z = np.array(
        [[np.cos(roll), -np.sin(roll), 0], [np.sin(roll), np.cos(roll), 0], [0, 0, 1]]
    )
    vector, _ = cv2.Rodrigues(rotation_z @ rotation_y @ rotation_x)
    camera = np.array([[640, 0, 320], [0, 640, 240], [0, 0, 1]], dtype=np.float64)
    projected, _ = cv2.projectPoints(
        GENERIC_FACE_POINTS, vector, np.array([0.0, 0.0, 1800.0]), camera, np.zeros(4)
    )
    points = face_points()
    points[POSE_LANDMARK_INDICES, :2] = projected.reshape(-1, 2)
    pose = estimate_head_pose(points, (480, 640, 3))
    assert pose is not None
    np.testing.assert_allclose(
        [pose.pitch_degrees, pose.yaw_degrees, pose.roll_degrees], angles, atol=0.01
    )
    assert pose.reprojection_error_pixels < 0.01


def test_pipeline_serialization_and_quality_are_independent_of_upscale():
    image = Image.new("RGB", (100, 100), (20, 20, 20))
    detector = StubDetector([face_points()])
    native = prepare_face(image, detector)
    resized = prepare_face(image, detector, upscale=True)
    assert native.report.quality == resized.report.quality
    assert native.report.quality.mean_brightness == 20
    assert native.report.quality.laplacian_variance == 0
    assert any("underexposure" in message for message in native.report.quality.warnings)
    assert resized.preview.shape[0] > resized.crop.shape[0]
    restored = AnalysisReport.model_validate_json(resized.report.model_dump_json())
    assert restored == resized.report
    resized.crop[:] = 0
    assert resized.original.min() == 20


def test_missing_model_is_actionable(tmp_path):
    with pytest.raises(ValueError, match="FACE_LANDMARKER_MODEL"):
        MediaPipeFaceDetector(tmp_path / "missing.task")


def test_config_rejects_invalid_thresholds():
    with pytest.raises(ValidationError):
        AnalysisConfig(crop_padding=-1)


def test_generation_refuses_uncalibrated_settings():
    with pytest.raises(UnsupportedGenerationError, match="calibration"):
        EldenRingAdapter().generate_sliders({"jaw_width": 1.0})


def test_quality_flags_follow_configured_thresholds():
    from preprocessing.models import HeadPose
    from preprocessing.pipeline import assess_image_quality

    crop = np.full((200, 200, 3), 128, dtype=np.uint8)
    bounds = FaceBounds(left=0, top=0, right=200, bottom=200)
    pose = HeadPose(pitch_degrees=0, yaw_degrees=40, roll_degrees=0, reprojection_error_pixels=25)
    quality = assess_image_quality(crop, bounds, pose, AnalysisConfig())
    assert set(quality.flags) == {"possible_blur", "large_rotation", "pose_fit_error"}
    relaxed = assess_image_quality(
        crop,
        bounds,
        pose,
        AnalysisConfig(blur_threshold=0, max_pose_degrees=50, max_pose_error_ratio=0.2),
    )
    assert relaxed.flags == []
    assert any("Occlusion confidence unavailable" in message for message in relaxed.warnings)
