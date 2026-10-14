import argparse
from pathlib import Path

from preprocessing.face_detection import MediaPipeFaceDetector
from preprocessing.pipeline import prepare_face
from reconstruction.export import export_reconstruction
from reconstruction.mica_model import MicaModel
from reconstruction.models import MicaConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct a local photograph with MICA.")
    parser.add_argument("photograph", type=Path)
    parser.add_argument("--face-model", type=Path, required=True)
    parser.add_argument("--mica-repo", type=Path, required=True)
    parser.add_argument("--mica-python", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = MicaConfig(
        repository_path=args.mica_repo,
        python_executable=args.mica_python,
        checkpoint_path=args.checkpoint,
    )
    detector = MediaPipeFaceDetector(args.face_model)
    try:
        prepared = prepare_face(args.photograph, detector)
    finally:
        detector.close()
    reconstructed = MicaModel(config).reconstruct(prepared)
    prepared.report.reconstruction_status = "experimental: MICA canonical mesh"
    export_reconstruction(reconstructed, args.output, prepared.report)
    print(f"Saved reconstruction to {args.output.resolve()}")


if __name__ == "__main__":
    main()
