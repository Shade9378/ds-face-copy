import numpy as np

from reconstruction.face_geometry import MIN_GEOMETRY_SCALE, validate_face_geometry
from reconstruction.models import FaceGeometry, ReconstructionError

BACKGROUND_COLOR = np.array([24, 26, 31], dtype=np.uint8)
SURFACE_COLOR = np.array([190, 195, 205], dtype=np.float64)
LIGHT_DIRECTION = np.array([-0.3, 0.5, 1.0], dtype=np.float64)
PREVIEW_MARGIN = 0.08


def render_frontal_preview(geometry: FaceGeometry, size: int = 512) -> np.ndarray:
    validate_face_geometry(geometry)
    if not 64 <= size <= 1024:
        raise ValueError("Preview size must be between 64 and 1024 pixels.")
    vertices = geometry.vertices
    lower = vertices[:, :2].min(axis=0)
    upper = vertices[:, :2].max(axis=0)
    extent = float((upper - lower).max())
    if extent < MIN_GEOMETRY_SCALE:
        raise ReconstructionError("Mesh cannot be projected into a frontal preview.")
    scale = (size - 1) * (1 - 2 * PREVIEW_MARGIN) / extent
    projected = (vertices[:, :2] - (lower + upper) / 2) * scale
    projected[:, 1] *= -1
    projected += (size - 1) / 2
    image = np.broadcast_to(BACKGROUND_COLOR, (size, size, 3)).copy()
    depth_buffer = np.full((size, size), -np.inf)
    light = LIGHT_DIRECTION / np.linalg.norm(LIGHT_DIRECTION)
    for triangle in geometry.triangles:
        screen_points = projected[triangle]
        minimum = np.maximum(np.floor(screen_points.min(axis=0)).astype(int), 0)
        maximum = np.minimum(np.ceil(screen_points.max(axis=0)).astype(int), size - 1)
        if np.any(minimum > maximum):
            continue
        first, second, third = screen_points
        denominator = (second[1] - third[1]) * (first[0] - third[0]) + (third[0] - second[0]) * (
            first[1] - third[1]
        )
        if abs(denominator) < MIN_GEOMETRY_SCALE:
            continue
        rows, columns = np.mgrid[minimum[1] : maximum[1] + 1, minimum[0] : maximum[0] + 1]
        weight_first = (
            (second[1] - third[1]) * (columns - third[0])
            + (third[0] - second[0]) * (rows - third[1])
        ) / denominator
        weight_second = (
            (third[1] - first[1]) * (columns - third[0]) + (first[0] - third[0]) * (rows - third[1])
        ) / denominator
        weight_third = 1 - weight_first - weight_second
        triangle_depth = (
            weight_first * vertices[triangle[0], 2]
            + weight_second * vertices[triangle[1], 2]
            + weight_third * vertices[triangle[2], 2]
        )
        depth_region = depth_buffer[minimum[1] : maximum[1] + 1, minimum[0] : maximum[0] + 1]
        visible = (
            (weight_first >= 0)
            & (weight_second >= 0)
            & (weight_third >= 0)
            & (triangle_depth > depth_region)
        )
        normal = np.cross(
            vertices[triangle[1]] - vertices[triangle[0]],
            vertices[triangle[2]] - vertices[triangle[0]],
        )
        length = float(np.linalg.norm(normal))
        if length < MIN_GEOMETRY_SCALE:
            continue
        normal /= length
        if normal[2] < 0:
            normal *= -1
        brightness = 0.25 + 0.75 * max(0.0, float(normal @ light))
        image_region = image[minimum[1] : maximum[1] + 1, minimum[0] : maximum[0] + 1]
        image_region[visible] = np.clip(SURFACE_COLOR * brightness, 0, 255).astype(np.uint8)
        depth_region[visible] = triangle_depth[visible]
    if not np.isfinite(depth_buffer).any():
        raise ReconstructionError("Mesh has no visible frontal triangles.")
    return image
