import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from preprocessing.models import AnalysisReport
from reconstruction.face_geometry import export_mesh_obj
from reconstruction.models import ReconstructedFace


def export_reconstruction(
    reconstructed_face: ReconstructedFace,
    output_directory: Path,
    analysis_report: AnalysisReport | None = None,
) -> list[Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    geometry = reconstructed_face.geometry
    mesh_path = output_directory / "mesh.obj"
    export_mesh_obj(geometry, mesh_path)
    np.save(output_directory / "landmarks.npy", geometry.landmarks, allow_pickle=False)
    np.save(output_directory / "identity.npy", geometry.identity_parameters, allow_pickle=False)
    Image.fromarray(reconstructed_face.preview).save(output_directory / "frontal_preview.png")
    report = {"reconstruction": reconstructed_face.report.model_dump(mode="json")}
    if analysis_report is not None:
        report["analysis"] = analysis_report.model_dump(mode="json")
    report_path = output_directory / "report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return [
        mesh_path,
        output_directory / "landmarks.npy",
        output_directory / "identity.npy",
        output_directory / "frontal_preview.png",
        report_path,
    ]


def create_reconstruction_archive(files: list[Path], archive_path: Path) -> Path:
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, arcname=path.name)
    return archive_path
