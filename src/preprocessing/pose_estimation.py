"""Experimental generic-face perspective pose estimation."""

import cv2
import numpy as np

from preprocessing.models import HeadPose

POSE_LANDMARK_INDICES = [1, 152, 33, 263, 61, 291]
# Nose, chin, outer eyes and mouth corners: x right, y down, z away.
# Generic proportions in arbitrary units, not the subject's reconstructed geometry.
GENERIC_FACE_POINTS = np.array(
    [
        [0, 0, 0],
        [0, 330, 65],
        [-225, -170, 135],
        [225, -170, 135],
        [-150, 150, 125],
        [150, 150, 125],
    ],
    dtype=np.float64,
)


def estimate_head_pose(points: np.ndarray, image_shape: tuple[int, ...]) -> HeadPose | None:
    """Fit six landmarks with estimated intrinsics; return None for failed fits."""
    height, width = image_shape[:2]
    camera = np.array([[width, 0, width / 2], [0, width, height / 2], [0, 0, 1]], dtype=np.float64)
    image_points = np.ascontiguousarray(points[POSE_LANDMARK_INDICES, :2], dtype=np.float64)
    try:
        success, rotation, translation = cv2.solvePnP(
            GENERIC_FACE_POINTS,
            image_points,
            camera,
            np.zeros(4),
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success or translation[2, 0] <= 0:
            return None
        projected, _ = cv2.projectPoints(
            GENERIC_FACE_POINTS, rotation, translation, camera, np.zeros(4)
        )
        residual = float(
            np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - image_points) ** 2, axis=1)))
        )
        matrix, _ = cv2.Rodrigues(rotation)
        pitch, yaw, roll = cv2.RQDecomp3x3(matrix)[0]
        if not np.isfinite([pitch, yaw, roll, residual]).all():
            return None
        return HeadPose(
            pitch_degrees=pitch,
            yaw_degrees=yaw,
            roll_degrees=roll,
            reprojection_error_pixels=residual,
        )
    except cv2.error:
        return None
