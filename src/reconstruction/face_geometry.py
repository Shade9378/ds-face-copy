from pathlib import Path

import numpy as np

from reconstruction.models import FaceGeometry, ReconstructionError

EYE_LANDMARK_RANGES = (slice(36, 42), slice(42, 48))
MIN_GEOMETRY_SCALE = 1e-8


def validate_face_geometry(geometry: FaceGeometry) -> None:
    vertices = geometry.vertices
    triangles = geometry.triangles
    landmarks = geometry.landmarks
    identity = geometry.identity_parameters
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3:
        raise ReconstructionError("MICA mesh must contain at least three 3D vertices.")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or len(triangles) == 0:
        raise ReconstructionError("MICA mesh must contain triangles.")
    if not np.issubdtype(triangles.dtype, np.integer):
        raise ReconstructionError("Mesh triangle indices must be integers.")
    if triangles.min() < 0 or triangles.max() >= len(vertices):
        raise ReconstructionError("Mesh triangle indices are out of bounds.")
    if landmarks.shape != (68, 3):
        raise ReconstructionError("Expected MICA's 68 canonical 3D landmarks.")
    if identity.ndim != 1 or identity.size == 0:
        raise ReconstructionError("MICA identity parameters are missing or invalid.")
    if not all(np.isfinite(values).all() for values in (vertices, landmarks, identity)):
        raise ReconstructionError("MICA output contains non-finite geometry.")
    if np.linalg.norm(np.ptp(vertices, axis=0)) < MIN_GEOMETRY_SCALE:
        raise ReconstructionError("MICA produced a collapsed mesh.")


def load_mica_geometry(output_directory: Path) -> FaceGeometry:
    mesh_path = output_directory / "mesh.obj"
    vertices = []
    triangles = []
    try:
        with mesh_path.open(encoding="utf-8") as mesh_file:
            for line in mesh_file:
                fields = line.split()
                if not fields:
                    continue
                if fields[0] == "v":
                    vertices.append([float(value) for value in fields[1:4]])
                elif fields[0] == "f":
                    if len(fields) != 4:
                        raise ReconstructionError("MICA mesh contains a non-triangular face.")
                    indices = [int(value.split("/")[0]) for value in fields[1:]]
                    triangles.append(
                        [index - 1 if index > 0 else len(vertices) + index for index in indices]
                    )
        landmarks = np.load(output_directory / "kpt68.npy", allow_pickle=False)
        if landmarks.shape == (1, 68, 3):
            landmarks = landmarks[0]
        identity = np.load(output_directory / "identity.npy", allow_pickle=False)
        geometry = FaceGeometry(
            vertices=np.asarray(vertices, dtype=np.float64),
            triangles=np.asarray(triangles, dtype=np.int64),
            landmarks=np.asarray(landmarks, dtype=np.float64),
            identity_parameters=np.asarray(identity, dtype=np.float64),
        )
        validate_face_geometry(geometry)
        return geometry
    except (OSError, ValueError, IndexError) as error:
        raise ReconstructionError("MICA output is missing or malformed.") from error


def extract_geometric_features(geometry: FaceGeometry) -> dict[str, float]:
    validate_face_geometry(geometry)
    points = geometry.landmarks
    left_eye, right_eye = [points[indices].mean(axis=0) for indices in EYE_LANDMARK_RANGES]
    scale = float(np.linalg.norm(left_eye - right_eye))
    if scale < MIN_GEOMETRY_SCALE:
        raise ReconstructionError("Eye centers overlap; geometry cannot be normalized.")

    def normalized_distance(first: int, second: int) -> float:
        return float(np.linalg.norm(points[first] - points[second]) / scale)

    eye_midpoint = (left_eye + right_eye) / 2
    mouth_midpoint = (points[48] + points[54]) / 2
    return {
        "jaw_width": normalized_distance(4, 12),
        "jaw_span": normalized_distance(0, 16),
        "chin_to_mouth": float(np.linalg.norm(points[8] - mouth_midpoint) / scale),
        "chin_width": normalized_distance(7, 9),
        "nose_length": normalized_distance(27, 33),
        "nose_width": normalized_distance(31, 35),
        "nose_tip_depth": float((points[30, 2] - eye_midpoint[2]) / scale),
        "left_eye_width": normalized_distance(36, 39),
        "right_eye_width": normalized_distance(42, 45),
        "left_eye_height": (normalized_distance(37, 41) + normalized_distance(38, 40)) / 2,
        "right_eye_height": (normalized_distance(43, 47) + normalized_distance(44, 46)) / 2,
        "mouth_width": normalized_distance(48, 54),
        "lip_height": normalized_distance(51, 57),
        "eye_to_chin": float(np.linalg.norm(eye_midpoint - points[8]) / scale),
        "brow_span": normalized_distance(17, 26),
    }


def export_mesh_obj(geometry: FaceGeometry, output_path: Path) -> None:
    validate_face_geometry(geometry)
    with output_path.open("w", encoding="utf-8") as output:
        for vertex in geometry.vertices:
            output.write("v " + " ".join(f"{value:.9g}" for value in vertex) + "\n")
        for triangle in geometry.triangles:
            output.write("f " + " ".join(str(int(index) + 1) for index in triangle) + "\n")
