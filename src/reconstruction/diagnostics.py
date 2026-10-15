import json
import os
import subprocess
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from reconstruction.mica_model import (
    REQUIRED_REPOSITORY_FILES,
    mica_config_from_environment,
    resolve_mica_paths,
)
from reconstruction.models import MicaConfig, ReconstructionError

RUNTIME_DIAGNOSTIC = """
import contextlib
import importlib
import io
import json
import sys
result = {"python_version": sys.version.split()[0], "dependencies": {}, "errors": []}
with contextlib.redirect_stdout(io.StringIO()):
    for name in ("torch", "cv2", "numpy", "insightface", "trimesh", "yacs",
                 "skimage", "loguru", "face_alignment", "micalib.models.mica"):
        try:
            module = importlib.import_module(name)
            result["dependencies"][name] = getattr(module, "__version__", "installed")
        except Exception as error:
            result["errors"].append(name + ": " + str(error))
    try:
        import torch
        result["cuda_available"] = bool(torch.cuda.is_available())
        result["torch_cuda_version"] = torch.version.cuda
        if not result["cuda_available"]:
            result["errors"].append("CUDA is unavailable in this Python environment.")
    except Exception as error:
        result["cuda_available"] = False
        result["errors"].append("CUDA check: " + str(error))
print("MICA_DIAGNOSTIC_JSON=" + json.dumps(result))
"""


class SetupCheck(BaseModel):
    name: str
    path: str
    available: bool


class MicaSetupReport(BaseModel):
    status: Literal["incomplete", "assets_present", "runtime_error", "runtime_checked"]
    checks: list[SetupCheck] = Field(default_factory=list)
    runtime: dict = Field(default_factory=dict)
    issues: list[str] = Field(default_factory=list)
    inference_tested: bool = False


def inspect_mica_setup(
    config: MicaConfig | None = None, check_runtime: bool = False
) -> MicaSetupReport:
    if config is None:
        try:
            config = mica_config_from_environment()
        except ReconstructionError as error:
            return MicaSetupReport(status="incomplete", issues=[str(error)])
    repository, executable, checkpoint = resolve_mica_paths(config)
    checks = [
        SetupCheck(name=name, path=str(repository / name), available=(repository / name).is_file())
        for name in REQUIRED_REPOSITORY_FILES
    ]
    checks.extend(
        [
            SetupCheck(name="python", path=str(executable), available=executable.is_file()),
            SetupCheck(name="checkpoint", path=str(checkpoint), available=checkpoint.is_file()),
        ]
    )
    detector_models = Path.home() / ".insightface/models/antelopev2"
    checks.append(
        SetupCheck(
            name="antelopev2",
            path=str(detector_models),
            available=detector_models.is_dir() and any(detector_models.glob("*.onnx")),
        )
    )
    issues = [f"Missing {check.name}: {check.path}" for check in checks if not check.available]
    if issues:
        return MicaSetupReport(status="incomplete", checks=checks, issues=issues)
    if not check_runtime:
        return MicaSetupReport(status="assets_present", checks=checks)
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    try:
        result = subprocess.run(
            [str(executable), "-c", RUNTIME_DIAGNOSTIC],
            cwd=repository,
            env=environment,
            capture_output=True,
            text=True,
            timeout=min(config.timeout_seconds, 60),
            check=False,
        )
        if result.returncode != 0:
            return MicaSetupReport(
                status="runtime_error",
                checks=checks,
                issues=[
                    (result.stderr or result.stdout).strip()[-2000:] or "Runtime check failed."
                ],
            )
        payloads = [
            line.removeprefix("MICA_DIAGNOSTIC_JSON=")
            for line in result.stdout.splitlines()
            if line.startswith("MICA_DIAGNOSTIC_JSON=")
        ]
        if len(payloads) != 1:
            raise ValueError("Runtime diagnostic returned an invalid response.")
        runtime = json.loads(payloads[0])
        if not isinstance(runtime, dict) or not isinstance(runtime.get("errors"), list):
            raise ValueError("Runtime diagnostic returned invalid metadata.")
        if not isinstance(runtime.get("cuda_available"), bool):
            raise ValueError("Runtime diagnostic returned invalid CUDA metadata.")
        issues = list(runtime["errors"])
        if not runtime["cuda_available"] and not issues:
            issues.append("CUDA is unavailable in this Python environment.")
        if not all(isinstance(issue, str) for issue in issues):
            raise ValueError("Runtime diagnostic returned invalid errors.")
        return MicaSetupReport(
            status="runtime_error" if issues else "runtime_checked",
            checks=checks,
            runtime=runtime,
            issues=issues,
        )
    except (OSError, subprocess.TimeoutExpired, ValueError) as error:
        return MicaSetupReport(status="runtime_error", checks=checks, issues=[str(error)])
