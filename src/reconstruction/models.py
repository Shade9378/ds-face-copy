from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class ReconstructionError(RuntimeError):
    pass


class MicaConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    repository_path: Path
    python_executable: Path
    checkpoint_path: Path | None = None
    timeout_seconds: float = Field(default=300, gt=0, le=3600)
    preview_size: int = Field(default=512, ge=64, le=1024)


class ReconstructionReport(BaseModel):
    schema_version: str = "1.0"
    backend: str = "MICA"
    status: str = "experimental"
    units: str = "millimeters"
    coordinate_system: str = "MICA canonical FLAME coordinates; preview looks along -z"
    vertex_count: int
    triangle_count: int
    landmark_count: int
    features: dict[str, float]
    feature_normalization: str = "3D distances divided by distance between eye landmark centers"
    unavailable_features: list[str] = ["forehead: no forehead landmarks in the 68-point output"]
    warnings: list[str]
    checkpoint_sha256: str
    upstream_revision: str | None = None


@dataclass
class FaceGeometry:
    vertices: np.ndarray
    triangles: np.ndarray
    landmarks: np.ndarray
    identity_parameters: np.ndarray


@dataclass
class ReconstructedFace:
    geometry: FaceGeometry
    preview: np.ndarray
    report: ReconstructionReport
