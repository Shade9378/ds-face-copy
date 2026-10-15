import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

from preprocessing.pipeline import PreparedFace
from reconstruction.face_geometry import extract_geometric_features, load_mica_geometry
from reconstruction.models import (
    MicaConfig,
    ReconstructedFace,
    ReconstructionError,
    ReconstructionReport,
)
from reconstruction.preview import render_frontal_preview

MICA_RUNTIME_CHECK = """
import sys
import torch
if not torch.cuda.is_available():
    raise RuntimeError("MICA requires a working CUDA-enabled PyTorch environment.")
checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
if not isinstance(checkpoint, dict):
    raise RuntimeError("MICA checkpoint must contain model state dictionaries.")
for name in ("arcface", "flameModel"):
    if not isinstance(checkpoint.get(name), dict) or not checkpoint[name]:
        raise RuntimeError("MICA checkpoint is missing weights for " + name)
"""


REQUIRED_REPOSITORY_FILES = (
    "demo.py",
    "configs/config.py",
    "micalib/models/mica.py",
    "data/FLAME2020/generic_model.pkl",
    "data/FLAME2020/landmark_embedding.npy",
    "data/FLAME2020/head_template.obj",
    "data/FLAME2020/FLAME_masks/FLAME_masks.pkl",
)


def mica_config_from_environment() -> MicaConfig:
    repository = os.environ.get("MICA_REPO")
    python_executable = os.environ.get("MICA_PYTHON")
    if not repository or not python_executable:
        raise ReconstructionError("Set MICA_REPO and MICA_PYTHON to your local MICA setup.")
    checkpoint = os.environ.get("MICA_CHECKPOINT")
    return MicaConfig(
        repository_path=Path(repository).expanduser(),
        python_executable=Path(python_executable).expanduser(),
        checkpoint_path=Path(checkpoint).expanduser() if checkpoint else None,
    )


def resolve_mica_paths(config: MicaConfig) -> tuple[Path, Path, Path]:
    repository = config.repository_path.resolve()
    executable = config.python_executable.expanduser().absolute()
    checkpoint = (
        config.checkpoint_path.resolve()
        if config.checkpoint_path
        else repository / "data/pretrained/mica.tar"
    )
    return repository, executable, checkpoint


def validate_mica_setup(config: MicaConfig) -> tuple[Path, Path, Path]:
    repository, executable, checkpoint = resolve_mica_paths(config)
    missing = [
        str(repository / name)
        for name in REQUIRED_REPOSITORY_FILES
        if not (repository / name).is_file()
    ]
    missing.extend(str(path) for path in (executable, checkpoint) if not path.is_file())
    detector_models = Path.home() / ".insightface/models/antelopev2"
    if not detector_models.is_dir() or not any(detector_models.glob("*.onnx")):
        missing.append(str(detector_models / "*.onnx"))
    if missing:
        raise ReconstructionError("MICA setup is incomplete. Missing: " + ", ".join(missing))
    return repository, executable, checkpoint


def _checkpoint_digest(checkpoint: Path) -> str:
    digest = hashlib.sha256()
    with checkpoint.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _upstream_revision(repository: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


class MicaModel:
    def __init__(self, config: MicaConfig) -> None:
        self.config = config

    def reconstruct(self, prepared_face: PreparedFace) -> ReconstructedFace:
        repository, executable, checkpoint = validate_mica_setup(self.config)
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        with tempfile.TemporaryDirectory(prefix="face-to-slider-mica-") as directory:
            workspace = Path(directory)
            input_directory = workspace / "input"
            input_directory.mkdir()
            Image.fromarray(prepared_face.crop).save(input_directory / "face.png")
            output_directory = workspace / "output"
            command = [
                str(executable),
                str(repository / "demo.py"),
                "-i",
                str(input_directory),
                "-o",
                str(output_directory),
                "-a",
                str(workspace / "arcface"),
                "-m",
                str(checkpoint),
            ]
            self._run_command(
                [str(executable), "-c", MICA_RUNTIME_CHECK, str(checkpoint)],
                repository,
                environment,
            )
            self._run_command(command, repository, environment)
            geometry = load_mica_geometry(output_directory / "face")
        features = extract_geometric_features(geometry)
        preview = render_frontal_preview(geometry, self.config.preview_size)
        warnings = list(prepared_face.report.quality.warnings)
        warnings.extend(
            [
                "MICA estimates identity shape; hidden geometry may differ from the person.",
                "Preview is an untextured canonical mesh, not a rendered game character.",
                "Real-model reconstruction accuracy has not been evaluated in this application.",
            ]
        )
        report = ReconstructionReport(
            vertex_count=len(geometry.vertices),
            triangle_count=len(geometry.triangles),
            landmark_count=len(geometry.landmarks),
            features=features,
            reliability=("review_required" if prepared_face.report.quality.flags else "unassessed"),
            input_quality_flags=prepared_face.report.quality.flags,
            warnings=warnings,
            checkpoint_sha256=_checkpoint_digest(checkpoint),
            upstream_revision=_upstream_revision(repository),
        )
        return ReconstructedFace(geometry=geometry, preview=preview, report=report)

    def _run_command(
        self, command: list[str], repository: Path, environment: dict[str, str]
    ) -> None:
        try:
            result = subprocess.run(
                command,
                cwd=repository,
                env=environment,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise ReconstructionError("MICA reconstruction timed out.") from error
        except OSError as error:
            raise ReconstructionError(
                "Could not start the configured MICA Python runtime."
            ) from error
        if result.returncode != 0:
            details = (result.stderr or result.stdout).strip()[-2000:]
            raise ReconstructionError(
                "MICA failed. Check its CUDA environment and model assets. " + details
            )
