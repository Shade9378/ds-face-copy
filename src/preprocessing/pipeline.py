from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from preprocessing.face_detection import FaceDetector, select_largest_face
from preprocessing.image_processing import crop_face, load_image, upscale_crop
from preprocessing.models import AnalysisConfig, AnalysisReport, FaceBounds, HeadPose, QualityReport
from preprocessing.pose_estimation import estimate_head_pose


@dataclass
class PreparedFace:
    original: np.ndarray
    crop: np.ndarray
    preview: np.ndarray
    report: AnalysisReport


def assess_image_quality(
    crop: np.ndarray, bounds: FaceBounds, pose: HeadPose | None, config: AnalysisConfig
) -> QualityReport:
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    width, height = bounds.right - bounds.left, bounds.bottom - bounds.top
    warnings = []
    flags = []
    if min(width, height) < config.min_face_pixels:
        warnings.append("Low-resolution face; resizing cannot restore missing detail.")
        flags.append("low_resolution")
    if blur < config.blur_threshold:
        warnings.append("Possible blur or low texture.")
        flags.append("possible_blur")
    if brightness < config.dark_threshold:
        warnings.append("Possible underexposure.")
        flags.append("underexposure")
    if brightness > config.bright_threshold:
        warnings.append("Possible overexposure.")
        flags.append("overexposure")
    if pose is None:
        warnings.append("Head pose estimation failed.")
        flags.append("pose_unavailable")
    else:
        if max(abs(pose.yaw_degrees), abs(pose.pitch_degrees)) > config.max_pose_degrees:
            warnings.append("Large head rotation; hidden regions cannot be measured reliably.")
            flags.append("large_rotation")
        if pose.reprojection_error_pixels > config.max_pose_error_ratio * min(width, height):
            warnings.append("High pose-fit residual; pose estimate may be unreliable.")
            flags.append("pose_fit_error")
    warnings.append("Occlusion confidence unavailable; inspect covered facial features manually.")
    return QualityReport(
        face_width_pixels=width,
        face_height_pixels=height,
        laplacian_variance=blur,
        mean_brightness=brightness,
        warnings=warnings,
        flags=flags,
    )


def prepare_face(
    source: str | Path | Image.Image,
    detector: FaceDetector,
    config: AnalysisConfig | None = None,
    upscale: bool = False,
) -> PreparedFace:
    config = config or AnalysisConfig()
    original = load_image(source, config.max_pixels)
    faces = detector.detect_faces(original)
    points, bounds = select_largest_face(faces, original.shape)
    crop, crop_bounds = crop_face(original, bounds, config.crop_padding)
    pose = estimate_head_pose(points, original.shape)
    tight_crop, _ = crop_face(original, bounds, 0)
    quality = assess_image_quality(tight_crop, bounds, pose, config)
    if len(faces) > 1:
        quality.warnings.append("Multiple faces detected; verify the selected face crop.")
        quality.flags.append("multiple_faces")
    if len(faces) >= config.max_faces:
        quality.warnings.append("Detection limit reached; a larger face may have been omitted.")
        quality.flags.append("detection_limit")
    if (
        bounds.left == 0
        or bounds.top == 0
        or bounds.right == original.shape[1]
        or bounds.bottom == original.shape[0]
    ):
        quality.warnings.append("Face touches the image boundary and may be truncated.")
        quality.flags.append("truncated_face")
    preview, factor = (
        upscale_crop(crop, config.upscale_target, config.max_upscale_factor)
        if upscale
        else (crop.copy(), 1.0)
    )
    report = AnalysisReport(
        image_width=original.shape[1],
        image_height=original.shape[0],
        detected_faces=len(faces),
        face_bounds=bounds,
        crop_bounds=crop_bounds,
        landmarks=[tuple(point) for point in points],
        pose=pose,
        quality=quality,
        upscale_factor=factor,
    )
    return PreparedFace(original=original, crop=crop, preview=preview, report=report)
