import argparse
from pathlib import Path

from reconstruction.diagnostics import inspect_mica_setup
from reconstruction.models import MicaConfig


def main() -> int:
    parser = argparse.ArgumentParser(description="Check local MICA assets and runtime.")
    parser.add_argument("--mica-repo", type=Path)
    parser.add_argument("--mica-python", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--runtime", action="store_true")
    args = parser.parse_args()
    if bool(args.mica_repo) != bool(args.mica_python):
        parser.error("--mica-repo and --mica-python must be supplied together.")
    if args.checkpoint and not args.mica_repo:
        parser.error("--checkpoint requires --mica-repo and --mica-python.")
    config = (
        MicaConfig(
            repository_path=args.mica_repo,
            python_executable=args.mica_python,
            checkpoint_path=args.checkpoint,
        )
        if args.mica_repo
        else None
    )
    report = inspect_mica_setup(config, check_runtime=args.runtime)
    print(report.model_dump_json(indent=2))
    return 1 if report.issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
