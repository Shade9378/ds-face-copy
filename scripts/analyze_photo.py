import argparse
from pathlib import Path

from preprocessing.face_detection import MediaPipeFaceDetector
from preprocessing.pipeline import prepare_face


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Command-line local photo analysis; reports are not game slider presets."
    )
    parser.add_argument("photograph", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--upscale", action="store_true")
    args = parser.parse_args()
    detector = MediaPipeFaceDetector(args.model)
    try:
        result = prepare_face(args.photograph, detector, upscale=args.upscale)
        args.report.write_text(result.report.model_dump_json(indent=2), encoding="utf-8")
    finally:
        detector.close()


if __name__ == "__main__":
    main()
