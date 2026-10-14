"""MediaPipe Tasks backend with explicit local model provisioning."""

from pathlib import Path
from typing import Protocol

import numpy as np

from preprocessing.models import FaceBounds


class FaceDetector(Protocol):
    """Injectable detector returning source-pixel landmarks, one array per face."""

    def detect_faces(self, image: np.ndarray) -> list[np.ndarray]:
        """Return arrays of shape (landmark count, 3)."""
        ...


class MediaPipeFaceDetector:
    """Run Face Landmarker locally; never implicitly download model assets."""

    def __init__(self, model_path: Path, max_faces: int = 10) -> None:
        if not model_path.is_file():
            raise ValueError(
                "Face Landmarker model missing. Set FACE_LANDMARKER_MODEL to a local .task file."
            )
        import mediapipe as mp

        self._mediapipe = mp
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=max_faces,
        )
        try:
            self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        except OSError as error:
            raise RuntimeError(
                "MediaPipe native runtime could not load. On Ubuntu/WSL, install "
                "libgles2 and libegl1; see README installation instructions."
            ) from error

    def detect_faces(self, image: np.ndarray) -> list[np.ndarray]:
        """Convert normalized landmarks to source pixels; z is relative, not metric depth."""
        mp = self._mediapipe
        result = self._landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=image))
        height, width = image.shape[:2]
        return [
            np.array([(p.x * width, p.y * height, p.z * width) for p in face], dtype=np.float64)
            for face in result.face_landmarks
        ]

    def close(self) -> None:
        """Release native resources."""
        self._landmarker.close()


def select_largest_face(
    faces: list[np.ndarray], image_shape: tuple[int, ...]
) -> tuple[np.ndarray, FaceBounds]:
    """Choose the largest valid visible landmark bounding box by pixel area."""
    height, width = image_shape[:2]
    candidates = []
    for points in faces:
        if points.ndim != 2 or points.shape[0] < 468 or points.shape[1] != 3:
            continue
        if not np.isfinite(points).all():
            continue
        bounds = FaceBounds(
            left=max(0, int(np.floor(points[:, 0].min()))),
            top=max(0, int(np.floor(points[:, 1].min()))),
            right=min(width, int(np.ceil(points[:, 0].max()))),
            bottom=min(height, int(np.ceil(points[:, 1].max()))),
        )
        if bounds.right > bounds.left and bounds.bottom > bounds.top:
            area = (bounds.right - bounds.left) * (bounds.bottom - bounds.top)
            candidates.append((area, points, bounds))
    if not candidates:
        raise ValueError("No usable face detected. Try a clearer photo with visible features.")
    _, points, bounds = max(candidates, key=lambda candidate: candidate[0])
    return points.copy(), bounds
