from pydantic import BaseModel, ConfigDict, Field


class AnalysisConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_pixels: int = Field(default=24_000_000, gt=0)
    max_faces: int = Field(default=10, ge=1, le=50)
    crop_padding: float = Field(default=0.2, ge=0, le=1)
    min_face_pixels: int = Field(default=160, gt=0)
    blur_threshold: float = Field(default=80, ge=0)
    dark_threshold: float = Field(default=50, ge=0, le=255)
    bright_threshold: float = Field(default=205, ge=0, le=255)
    max_pose_degrees: float = Field(default=35, gt=0, le=90)
    upscale_target: int = Field(default=512, gt=0, le=2048)
    max_upscale_factor: float = Field(default=4, ge=1, le=4)


class FaceBounds(BaseModel):
    left: int
    top: int
    right: int
    bottom: int


class HeadPose(BaseModel):
    pitch_degrees: float
    yaw_degrees: float
    roll_degrees: float
    reprojection_error_pixels: float


class QualityReport(BaseModel):
    face_width_pixels: int
    face_height_pixels: int
    laplacian_variance: float
    mean_brightness: float
    warnings: list[str]


class AnalysisReport(BaseModel):
    schema_version: str = "1.0"
    image_width: int
    image_height: int
    detected_faces: int
    face_bounds: FaceBounds
    crop_bounds: FaceBounds
    landmarks: list[tuple[float, float, float]]
    landmark_coordinates: str = "source pixels x/y; relative MediaPipe z scaled by image width"
    pose: HeadPose | None
    quality: QualityReport
    upscale_factor: float = 1
    reconstruction_status: str = "unsupported: MICA integration and licensed assets unavailable"
