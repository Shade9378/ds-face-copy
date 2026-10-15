import ast
import io
import json
import subprocess
import sys
import tokenize
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from preprocessing.models import AnalysisConfig
from preprocessing.pipeline import prepare_face
from reconstruction.diagnostics import inspect_mica_setup
from reconstruction.export import create_reconstruction_archive, export_reconstruction
from reconstruction.face_geometry import (
    export_mesh_obj,
    extract_geometric_features,
    load_mica_geometry,
    validate_face_geometry,
)
from reconstruction.mica_model import (
    REQUIRED_REPOSITORY_FILES,
    MicaModel,
    mica_config_from_environment,
    validate_mica_setup,
)
from reconstruction.models import (
    FaceGeometry,
    MicaConfig,
    ReconstructedFace,
    ReconstructionError,
    ReconstructionReport,
)
from reconstruction.preview import BACKGROUND_COLOR, render_frontal_preview


@pytest.fixture
def geometry():
    landmarks = np.zeros((68, 3), dtype=np.float64)
    landmarks[:, 0] = np.linspace(-50, 50, 68)
    landmarks[:, 1] = np.sin(np.linspace(0, np.pi, 68)) * 30
    landmarks[36:42] = [
        [-32, 20, 0],
        [-29, 22, 0],
        [-26, 22, 0],
        [-23, 20, 0],
        [-26, 18, 0],
        [-29, 18, 0],
    ]
    landmarks[42:48] = [
        [23, 20, 0],
        [26, 22, 0],
        [29, 22, 0],
        [32, 20, 0],
        [29, 18, 0],
        [26, 18, 0],
    ]
    return FaceGeometry(
        vertices=np.array(
            [[-60.0, -60, 0], [60.0, -60, 0], [60.0, 60, 0], [-60.0, 60, 0], [0.0, 0, 30]]
        ),
        triangles=np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]),
        landmarks=landmarks,
        identity_parameters=np.zeros(300),
    )


def write_geometry(directory, geometry):
    directory.mkdir(parents=True, exist_ok=True)
    export_mesh_obj(geometry, directory / "mesh.obj")
    np.save(directory / "kpt68.npy", geometry.landmarks[None], allow_pickle=False)
    np.save(directory / "identity.npy", geometry.identity_parameters, allow_pickle=False)


def test_geometry_round_trip(tmp_path, geometry):
    write_geometry(tmp_path, geometry)
    restored = load_mica_geometry(tmp_path)
    for name in ("vertices", "triangles", "landmarks", "identity_parameters"):
        np.testing.assert_allclose(getattr(restored, name), getattr(geometry, name))


@pytest.mark.parametrize("problem", ["nan", "index", "landmarks", "identity", "collapsed", "float"])
def test_invalid_geometry_rejected(geometry, problem):
    if problem == "nan":
        geometry.vertices[0, 0] = np.nan
    elif problem == "index":
        geometry.triangles[0, 0] = 1000
    elif problem == "landmarks":
        geometry.landmarks = np.zeros((67, 3))
    elif problem == "identity":
        geometry.identity_parameters = np.zeros((1, 300))
    elif problem == "collapsed":
        geometry.vertices[:] = 0
    else:
        geometry.triangles = geometry.triangles.astype(float)
    with pytest.raises(ReconstructionError):
        validate_face_geometry(geometry)


def test_missing_and_object_array_outputs_rejected(tmp_path, geometry):
    with pytest.raises(ReconstructionError, match="missing or malformed"):
        load_mica_geometry(tmp_path)
    write_geometry(tmp_path, geometry)
    np.save(tmp_path / "identity.npy", np.array([{"unexpected": "object"}], dtype=object))
    with pytest.raises(ReconstructionError, match="missing or malformed"):
        load_mica_geometry(tmp_path)


def test_feature_normalization_invariant_to_scale_and_translation(geometry):
    original = extract_geometric_features(geometry)
    geometry.vertices = geometry.vertices * 3 + np.array([20, 10, -5])
    geometry.landmarks = geometry.landmarks * 3 + np.array([20, 10, -5])
    transformed = extract_geometric_features(geometry)
    np.testing.assert_allclose(list(original.values()), list(transformed.values()))
    assert "forehead" not in original


def test_overlapping_eyes_rejected(geometry):
    geometry.landmarks[42:48] = geometry.landmarks[36:42]
    with pytest.raises(ReconstructionError, match="Eye centers"):
        extract_geometric_features(geometry)


def test_frontal_preview_has_surface_and_background(geometry):
    image = render_frontal_preview(geometry, 128)
    assert image.shape == (128, 128, 3)
    assert image.dtype == np.uint8
    np.testing.assert_array_equal(image[0, 0], BACKGROUND_COLOR)
    assert not np.array_equal(image[64, 64], BACKGROUND_COLOR)


def test_preview_depth_is_independent_of_triangle_order(geometry):
    geometry.vertices = np.array(
        [
            [-50.0, -50, 0],
            [50.0, -50, 0],
            [0.0, 50, 0],
            [-50.0, -50, 10],
            [50.0, -50, 30],
            [0.0, 50, 20],
        ]
    )
    geometry.triangles = np.array([[0, 1, 2], [3, 4, 5]])
    first = render_frontal_preview(geometry, 128)
    geometry.triangles = geometry.triangles[::-1].copy()
    second = render_frontal_preview(geometry, 128)
    np.testing.assert_array_equal(first, second)


def test_export_archive_contains_geometry_and_report(tmp_path, geometry):
    reconstructed = ReconstructedFace(
        geometry=geometry,
        preview=render_frontal_preview(geometry, 64),
        report=ReconstructionReport(
            vertex_count=5,
            triangle_count=4,
            landmark_count=68,
            features=extract_geometric_features(geometry),
            warnings=["experimental"],
            checkpoint_sha256="test-checkpoint",
        ),
    )
    files = export_reconstruction(reconstructed, tmp_path)
    archive_path = create_reconstruction_archive(files, tmp_path / "reconstruction.zip")
    with zipfile.ZipFile(archive_path) as archive:
        assert set(archive.namelist()) == {
            "mesh.obj",
            "landmarks.npy",
            "identity.npy",
            "frontal_preview.png",
            "report.json",
        }
        report = json.loads(archive.read("report.json"))
        assert report["reconstruction"]["units"] == "millimeters"
        assert report["reconstruction"]["status"] == "experimental"
        assert "forehead" in report["reconstruction"]["unavailable_features"][0]


@pytest.fixture
def mica_setup(tmp_path, monkeypatch, geometry):
    repository = tmp_path / "MICA checkout with spaces"
    for name in REQUIRED_REPOSITORY_FILES:
        path = repository / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    checkpoint = repository / "data/pretrained/mica.tar"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"synthetic checkpoint")
    model_directory = tmp_path / ".insightface/models/antelopev2"
    model_directory.mkdir(parents=True)
    (model_directory / "detector.onnx").write_bytes(b"synthetic detector")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    (repository / "torch.py").write_text(
        "class cuda:\n"
        "    @staticmethod\n"
        "    def is_available():\n"
        "        return True\n"
        "def load(path, map_location, weights_only):\n"
        "    return {'arcface': {'weight': 1}, 'flameModel': {'weight': 1}}\n",
        encoding="utf-8",
    )
    write_geometry(repository / "fixture", geometry)
    (repository / "demo.py").write_text(
        "import argparse\n"
        "import shutil\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        "from PIL import Image\n"
        "parser = argparse.ArgumentParser()\n"
        "for option in ('-i', '-o', '-a', '-m'):\n"
        "    parser.add_argument(option, required=True)\n"
        "args = parser.parse_args()\n"
        "output = Path(args.o) / 'face'\n"
        "shutil.copytree('fixture', output)\n"
        "with Image.open(Path(args.i) / 'face.png') as image:\n"
        "    np.save(output / 'identity.npy', np.array(image.size))\n",
        encoding="utf-8",
    )
    return MicaConfig(
        repository_path=repository, python_executable=Path(sys.executable), preview_size=64
    )


@pytest.fixture
def prepared_face():
    class StubDetector:
        def detect_faces(self, image):
            points = np.zeros((478, 3))
            points[:, 0] = np.linspace(20, 80, 478)
            points[:, 1] = np.linspace(20, 80, 478)
            return [points]

    return prepare_face(
        Image.new("RGB", (100, 100), "gray"), StubDetector(), AnalysisConfig(), upscale=True
    )


def test_setup_error_lists_missing_assets(tmp_path):
    config = MicaConfig(repository_path=tmp_path, python_executable=tmp_path / "python")
    with pytest.raises(ReconstructionError, match="MICA setup is incomplete"):
        validate_mica_setup(config)


def test_environment_configuration(monkeypatch, mica_setup):
    monkeypatch.setenv("MICA_REPO", str(mica_setup.repository_path))
    monkeypatch.setenv("MICA_PYTHON", str(mica_setup.python_executable))
    config = mica_config_from_environment()
    assert config.repository_path == mica_setup.repository_path
    monkeypatch.delenv("MICA_REPO")
    with pytest.raises(ReconstructionError, match="MICA_REPO and MICA_PYTHON"):
        mica_config_from_environment()


def test_real_subprocess_contract_uses_native_crop(mica_setup, prepared_face):
    reconstructed = MicaModel(mica_setup).reconstruct(prepared_face)
    height, width = prepared_face.crop.shape[:2]
    np.testing.assert_array_equal(reconstructed.geometry.identity_parameters, [width, height])
    assert reconstructed.preview.shape == (64, 64, 3)
    assert len(reconstructed.report.checkpoint_sha256) == 64
    assert reconstructed.report.warnings[: len(prepared_face.report.quality.warnings)] == (
        prepared_face.report.quality.warnings
    )


def test_failed_process_does_not_return_geometry(mica_setup, prepared_face):
    (mica_setup.repository_path / "demo.py").write_text(
        "import sys\nsys.stderr.write('CUDA unavailable')\nsys.exit(1)\n", encoding="utf-8"
    )
    with pytest.raises(ReconstructionError, match="CUDA unavailable"):
        MicaModel(mica_setup).reconstruct(prepared_face)


def test_successful_process_without_mesh_fails(mica_setup, prepared_face):
    (mica_setup.repository_path / "demo.py").write_text("", encoding="utf-8")
    with pytest.raises(ReconstructionError, match="missing or malformed"):
        MicaModel(mica_setup).reconstruct(prepared_face)


def test_process_timeout(mica_setup, prepared_face):
    (mica_setup.repository_path / "demo.py").write_text(
        "import time\ntime.sleep(10)\n", encoding="utf-8"
    )
    mica_setup.timeout_seconds = 0.1
    with pytest.raises(ReconstructionError, match="timed out"):
        MicaModel(mica_setup).reconstruct(prepared_face)


def test_source_remains_comment_free():
    root = Path(__file__).resolve().parents[1]
    for directory in ("src", "scripts", "tests"):
        for path in (root / directory).rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            assert all(
                token.type != tokenize.COMMENT
                for token in (tokenize.generate_tokens(io.StringIO(source).readline))
            ), path
            for node in ast.walk(ast.parse(source)):
                if isinstance(
                    node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
                ):
                    assert ast.get_docstring(node) is None, path


def test_incomplete_checkpoint_cannot_generate_random_geometry(mica_setup, prepared_face):
    (mica_setup.repository_path / "torch.py").write_text(
        "class cuda:\n"
        "    @staticmethod\n"
        "    def is_available():\n"
        "        return True\n"
        "def load(path, map_location, weights_only):\n"
        "    return {'arcface': {'weight': 1}}\n",
        encoding="utf-8",
    )
    with pytest.raises(ReconstructionError, match="missing weights for flameModel"):
        MicaModel(mica_setup).reconstruct(prepared_face)


def test_setup_diagnostics_without_configuration(monkeypatch):
    monkeypatch.delenv("MICA_REPO", raising=False)
    monkeypatch.delenv("MICA_PYTHON", raising=False)
    report = inspect_mica_setup()
    assert report.status == "incomplete"
    assert report.inference_tested is False
    assert "MICA_REPO and MICA_PYTHON" in report.issues[0]


def test_setup_diagnostics_lists_missing_assets_without_starting_process(tmp_path, monkeypatch):
    def unexpected_process(*args, **kwargs):
        pytest.fail("Incomplete setup must not start a subprocess.")

    monkeypatch.setattr("reconstruction.diagnostics.subprocess.run", unexpected_process)
    report = inspect_mica_setup(
        MicaConfig(repository_path=tmp_path, python_executable=tmp_path / "python"),
        check_runtime=True,
    )
    assert report.status == "incomplete"
    assert report.runtime == {}
    assert any("generic_model.pkl" in issue for issue in report.issues)
    assert any("checkpoint" in issue for issue in report.issues)


def test_asset_presence_is_not_inference_validation(mica_setup):
    report = inspect_mica_setup(mica_setup)
    assert report.status == "assets_present"
    assert report.inference_tested is False
    assert not report.runtime
    assert all(check.available for check in report.checks)


@pytest.mark.parametrize(
    "runtime,expected",
    [
        ({"errors": [], "cuda_available": True}, "runtime_checked"),
        ({"errors": [], "cuda_available": False}, "runtime_error"),
        ({"errors": ["Missing numpy"], "cuda_available": True}, "runtime_error"),
        ({"errors": []}, "runtime_error"),
        ({"errors": [3], "cuda_available": True}, "runtime_error"),
    ],
)
def test_runtime_diagnostics_status(mica_setup, monkeypatch, runtime, expected):
    def probe(command, **kwargs):
        assert command[0] == str(mica_setup.python_executable.absolute())
        assert kwargs["cwd"] == mica_setup.repository_path.resolve()
        assert "PYTHONPATH" not in kwargs["env"]
        assert kwargs["timeout"] <= 60
        return subprocess.CompletedProcess(
            command, 0, "extra library output\nMICA_DIAGNOSTIC_JSON=" + json.dumps(runtime), ""
        )

    monkeypatch.setattr("reconstruction.diagnostics.subprocess.run", probe)
    report = inspect_mica_setup(mica_setup, check_runtime=True)
    assert report.status == expected
    assert report.inference_tested is False
    assert bool(report.issues) == (expected == "runtime_error")


@pytest.mark.parametrize("failure", ["exit", "timeout", "launch", "invalid_json"])
def test_runtime_diagnostics_failures(mica_setup, monkeypatch, failure):
    def probe(command, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 1)
        if failure == "launch":
            raise OSError("Permission denied")
        if failure == "invalid_json":
            return subprocess.CompletedProcess(command, 0, "MICA_DIAGNOSTIC_JSON=invalid", "")
        return subprocess.CompletedProcess(command, 1, "", "Cannot import torch")

    monkeypatch.setattr("reconstruction.diagnostics.subprocess.run", probe)
    report = inspect_mica_setup(mica_setup, check_runtime=True)
    assert report.status == "runtime_error"
    assert report.issues


def test_reconstruction_quality_flags_require_review(mica_setup, prepared_face):
    reconstructed = MicaModel(mica_setup).reconstruct(prepared_face)
    assert "low_resolution" in reconstructed.report.input_quality_flags
    assert reconstructed.report.reliability == "review_required"
    assert reconstructed.report.occlusion_assessed is False


def test_no_input_flags_does_not_claim_reliable_reconstruction(mica_setup, prepared_face):
    prepared_face.report.quality.flags = []
    reconstructed = MicaModel(mica_setup).reconstruct(prepared_face)
    assert reconstructed.report.reliability == "unassessed"
    assert reconstructed.report.occlusion_assessed is False
