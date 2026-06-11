import argparse
from pathlib import Path


DEFAULT_SUBDIRS = ("depth",)


def collect_names(directory: Path, patterns: list[str]) -> set[str]:
    names: set[str] = set()
    for pattern in patterns:
        for path in directory.glob(pattern):
            if path.is_file():
                names.add(path.name)
    return names


def find_frame_dirs(root: Path, base_dir_name: str, target_dir_names: list[str]) -> list[Path]:
    frame_dirs: list[Path] = []
    for bgr_dir in root.rglob(base_dir_name):
        if not bgr_dir.is_dir():
            continue
        parent = bgr_dir.parent
        if any((parent / name).is_dir() for name in target_dir_names):
            frame_dirs.append(parent)
    return sorted(frame_dirs)


def align_one_dir(
    frame_dir: Path,
    base_dir_name: str,
    target_dir_names: list[str],
    patterns: list[str],
    apply: bool,
) -> tuple[int, int, list[str]]:
    base_dir = frame_dir / base_dir_name
    base_names = collect_names(base_dir, patterns)
    deleted = 0
    missing = 0
    messages: list[str] = []

    for target_name in target_dir_names:
        target_dir = frame_dir / target_name
        if not target_dir.is_dir():
            messages.append(f"missing folder: {target_dir}")
            missing += len(base_names)
            continue

        target_files = [
            path
            for pattern in patterns
            for path in target_dir.glob(pattern)
            if path.is_file()
        ]
        target_names = {path.name for path in target_files}

        extra_files = [path for path in target_files if path.name not in base_names]
        missing_names = sorted(base_names - target_names)

        if extra_files:
            for path in extra_files:
                if apply:
                    path.unlink()
            deleted += len(extra_files)
            action = "deleted" if apply else "would delete"
            messages.append(f"{target_name}: {action} {len(extra_files)} extra files")

        if missing_names:
            missing += len(missing_names)
            messages.append(f"{target_name}: missing {len(missing_names)} files compared with {base_dir_name}")

    return deleted, missing, messages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Align dataset folders by deleting files in depth that do not exist in bgr."
    )
    parser.add_argument("root", type=Path, help="Dataset root, for example dataset")
    parser.add_argument("--base", default="bgr", help="Reference folder name. Default: bgr")
    parser.add_argument(
        "--targets",
        nargs="+",
        default=list(DEFAULT_SUBDIRS),
        help="Folders to align to the base folder. Default: depth",
    )
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=["*.png", "*.pcd"],
        help="File patterns to compare. Default: *.png *.pcd",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete extra files. Without this flag, only prints what would change.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root not found: {root}")

    frame_dirs = find_frame_dirs(root, args.base, args.targets)
    total_deleted = 0
    total_missing = 0
    changed_dirs = 0

    print(f"Root: {root}")
    print(f"Mode: {'APPLY' if args.apply else 'DRY-RUN'}")
    print(f"Base: {args.base}")
    print(f"Targets: {', '.join(args.targets)}")
    print(f"Checked dirs: {len(frame_dirs)}")

    for frame_dir in frame_dirs:
        deleted, missing, messages = align_one_dir(
            frame_dir=frame_dir,
            base_dir_name=args.base,
            target_dir_names=args.targets,
            patterns=args.patterns,
            apply=args.apply,
        )
        total_deleted += deleted
        total_missing += missing
        if messages:
            changed_dirs += 1
            print(f"\n{frame_dir}")
            for message in messages:
                print(f"  - {message}")

    print("\nSummary")
    print(f"Dirs with differences: {changed_dirs}")
    print(f"Extra files {'deleted' if args.apply else 'to delete'}: {total_deleted}")
    print(f"Missing files in targets: {total_missing}")


if __name__ == "__main__":
    main()
