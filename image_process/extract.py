import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


DEFAULT_BAG_DIR = Path("data")
DEFAULT_OUTPUT_ROOT = Path("dataset")
DEFAULT_MERGE_OUTPUT_DIR = Path("dataset_merged")
DEFAULT_MERGE_DIRS = [
    "bgr",
    "depth",
]

# Fallback camera intrinsics. If a bag reports aligned color intrinsics
# successfully, those values are written instead.
DEFAULT_INTRINSICS = {
    "width": 1280,
    "height": 720,
    "fx": 662.636,
    "fy": 662.636,
    "cx": 635.522,
    "cy": 348.738,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract RealSense bag files into the bgr/depth workflow layout, "
            "or merge already extracted workflow datasets."
        )
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge existing extracted datasets instead of extracting bag files.",
    )
    parser.add_argument(
        "--renumber",
        action="store_true",
        help="Rename an existing dataset's bgr/depth/pcd files to sequential frame ids.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Renumber mode: dataset directory containing bgr/depth.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Renumber mode: first output frame index.",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=6,
        help="Renumber mode: zero-padding width for output frame ids.",
    )
    parser.add_argument("--bag-dir", type=Path, default=DEFAULT_BAG_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--save-interval", type=int, default=45)
    parser.add_argument(
        "--merge-after-extract",
        action="store_true",
        help=(
            "After extracting bag files, merge this run's bag output folders "
            "into the configured merge output."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing bag output directory.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Merge mode: root containing extracted dataset folders.",
    )
    parser.add_argument(
        "--merge-output",
        type=Path,
        default=DEFAULT_MERGE_OUTPUT_DIR,
        help=(
            "Merge mode: output dataset directory, or parent directory when "
            "--merge-name is provided."
        ),
    )
    parser.add_argument(
        "--merge-name",
        default=None,
        help=(
            "Merge mode: optional name for the merged dataset folder. "
            "Example: --merge-output ./out --merge-name merged_demo creates "
            "./out/merged_demo."
        ),
    )
    parser.add_argument(
        "--merge-dir",
        action="append",
        default=None,
        help=(
            "Merge mode: relative folder to merge, e.g. bgr. "
            "May be repeated. Defaults to bgr,depth."
        ),
    )
    parser.add_argument(
        "--merge-missing-files",
        action="store_true",
        help=(
            "Deprecated compatibility flag. Merge mode now includes already "
            "merged source folders by default and copies only missing target "
            "files unless --overwrite is set."
        ),
    )
    parser.add_argument(
        "--sources",
        nargs="*",
        default=None,
        help="Merge mode: optional source dataset folder names relative to --source-root.",
    )
    parser.add_argument(
        "--source-path",
        action="append",
        type=Path,
        default=None,
        help=(
            "Merge mode: explicit source dataset directory. May be repeated "
            "and may point anywhere, as long as it contains bgr/depth."
        ),
    )
    parser.add_argument(
        "--separator",
        default="__",
        help="Merge mode: separator between source folder name and original filename.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Merge mode: actually copy files. Without this, only previews the merge plan.",
    )
    return parser.parse_args()


def make_workflow_dirs(bag_output_dir: Path) -> dict[str, Path]:
    dirs = {
        "bgr": bag_output_dir / "bgr",
        "depth": bag_output_dir / "depth",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def intrinsics_to_dict(intr) -> dict:
    if intr is None:
        return dict(DEFAULT_INTRINSICS)
    return {
        "width": int(intr.width),
        "height": int(intr.height),
        "fx": float(intr.fx),
        "fy": float(intr.fy),
        "cx": float(intr.ppx),
        "cy": float(intr.ppy),
        "model": str(intr.model),
        "coeffs": [float(x) for x in intr.coeffs],
    }


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def extract_bag(bag_path: Path, output_root: Path, save_interval: int, overwrite: bool) -> None:
    try:
        import pyrealsense2 as rs
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pyrealsense2 is required to extract .bag files. Install it in this "
            "environment, for example: pip install pyrealsense2"
        ) from exc

    bag_name = bag_path.stem
    bag_output_dir = output_root / bag_name

    if bag_output_dir.exists() and not overwrite:
        print(f"Skip existing same-name bag output: {bag_output_dir}")
        return

    dirs = make_workflow_dirs(bag_output_dir)
    print(f"\nProcessing: {bag_path}")
    print(f"Output: {bag_output_dir}")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(str(bag_path), repeat_playback=False)

    profile = pipeline.start(config)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)
    align = rs.align(rs.stream.color)

    frame_idx = 0
    save_idx = 0
    intrinsics = None

    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                frame_idx += 1
                continue

            if intrinsics is None:
                color_intr = color_frame.profile.as_video_stream_profile().get_intrinsics()
                intrinsics = intrinsics_to_dict(color_intr)

            if frame_idx % save_interval != 0:
                frame_idx += 1
                continue

            color_rgb = np.asanyarray(color_frame.get_data())
            color_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
            depth_mm = np.asanyarray(depth_frame.get_data())

            stem = f"{save_idx:06d}"
            bgr_path = dirs["bgr"] / f"{stem}.png"
            depth_path = dirs["depth"] / f"{stem}.png"

            cv2.imwrite(str(bgr_path), color_bgr)
            cv2.imwrite(str(depth_path), depth_mm)

            print(f"{bag_name}: saved {stem}")
            frame_idx += 1
            save_idx += 1

    except RuntimeError:
        print(f"Finish: {bag_name}")
    finally:
        pipeline.stop()

def merge_relative_dirs(args: argparse.Namespace) -> list[str]:
    rel_dirs = args.merge_dir if args.merge_dir else list(DEFAULT_MERGE_DIRS)
    return sorted(dict.fromkeys(rel_dirs))


def resolve_merge_output(args: argparse.Namespace) -> Path:
    if args.merge_name:
        return args.merge_output / args.merge_name
    return args.merge_output


def discover_merge_sources(
    source_root: Path,
    merge_output: Path,
    source_names: list[str] | None,
    source_paths: list[Path] | None,
) -> list[Path]:
    sources = []
    if source_names:
        sources.extend(source_root / name for name in source_names)
    if source_paths:
        sources.extend(source_paths)
    if not sources:
        sources = sorted(
            path
            for path in source_root.iterdir()
            if path.is_dir()
            and ((path / "bgr").is_dir() or (path / "input" / "bgr").is_dir())
        )

    output_resolved = merge_output.resolve()
    filtered_sources = []
    seen = set()
    for source in sources:
        if not source.exists():
            raise FileNotFoundError(f"source dataset does not exist: {source}")
        source_resolved = source.resolve()
        if source_resolved == output_resolved:
            continue
        if source_resolved in seen:
            continue
        seen.add(source_resolved)
        filtered_sources.append(source)
    return filtered_sources


def merged_filename(source_name: str, src_file: Path, separator: str) -> str:
    return f"{source_name}{separator}{src_file.name}"


def source_relative_dir(source: Path, rel_dir: str) -> Path:
    new_layout_dir = source / rel_dir
    if new_layout_dir.is_dir():
        return new_layout_dir

    # Compatibility for datasets extracted before the layout was flattened.
    old_layout_dir = source / "input" / rel_dir
    return old_layout_dir


def build_merge_plan(
    sources: list[Path],
    merge_output: Path,
    rel_dirs: list[str],
    separator: str,
) -> list[tuple[Path, Path, str, str]]:
    plan = []
    for source in sources:
        source_name = source.name
        for rel_dir in rel_dirs:
            src_dir = source_relative_dir(source, rel_dir)
            if not src_dir.is_dir():
                continue
            dst_dir = merge_output / rel_dir
            for src_file in sorted(path for path in src_dir.iterdir() if path.is_file()):
                dst_file = dst_dir / merged_filename(source_name, src_file, separator)
                plan.append((src_file, dst_file, source_name, rel_dir))
    return plan


def source_already_merged(
    source: Path, merge_output: Path, rel_dirs: list[str], separator: str
) -> bool:
    prefix = f"{source.name}{separator}"
    for rel_dir in rel_dirs:
        dst_dir = merge_output / rel_dir
        if not dst_dir.is_dir():
            continue
        if any(path.is_file() and path.name.startswith(prefix) for path in dst_dir.iterdir()):
            return True
    return False


def filter_already_merged_sources(
    sources: list[Path],
    merge_output: Path,
    rel_dirs: list[str],
    separator: str,
    overwrite: bool,
) -> tuple[list[Path], list[Path]]:
    if overwrite:
        return sources, []

    pending = []
    skipped = []
    for source in sources:
        if source_already_merged(source, merge_output, rel_dirs, separator):
            skipped.append(source)
        else:
            pending.append(source)
    return pending, skipped


def check_merge_plan(plan: list[tuple[Path, Path, str, str]], overwrite: bool) -> None:
    target_counts = Counter(dst for _, dst, _, _ in plan)
    duplicates = [path for path, count in target_counts.items() if count > 1]
    if duplicates:
        raise RuntimeError(
            "duplicate merge targets found:\n"
            + "\n".join(str(path) for path in duplicates[:20])
        )


def copy_file_resumable(src: Path, dst: Path, overwrite: bool) -> str:
    """Copy a file and allow re-running a partially completed merge."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if overwrite:
            shutil.copyfile(src, dst)
            return "overwritten"
        return "skipped_existing"
    # Do not use shutil.copy2 here. On Windows-mounted paths (e.g. WSL /mnt/*
    # drives), copying file timestamps/permission metadata may fail with PermissionError.
    shutil.copyfile(src, dst)
    return "copied"


def copy_merge_meta(
    sources: list[Path],
    merge_output: Path,
    overwrite: bool,
    apply: bool,
) -> list[dict]:
    manifest_sources = []
    first_intrinsic = None

    for source in sources:
        source_meta = source / "meta"
        entry = {"name": source.name, "path": str(source), "meta_files": []}
        if source_meta.is_dir():
            for src_file in sorted(path for path in source_meta.iterdir() if path.is_file()):
                dst_file = merge_output / "meta" / "sources" / source.name / src_file.name
                entry["meta_files"].append(str(dst_file))
                if apply:
                    copy_file_resumable(src_file, dst_file, overwrite)

            intrinsic = source_meta / "intrinsic.json"
            if intrinsic.exists() and first_intrinsic is None:
                first_intrinsic = intrinsic
        manifest_sources.append(entry)

    if first_intrinsic is not None and apply:
        dst_intrinsic = merge_output / "meta" / "intrinsic.json"
        copy_file_resumable(first_intrinsic, dst_intrinsic, overwrite)

    return manifest_sources


def write_merge_manifest(
    merge_output: Path,
    source_root: Path,
    rel_dirs: list[str],
    sources: list[dict],
    plan: list[tuple[Path, Path, str, str]],
    overwrite: bool,
) -> None:
    by_source = Counter(source_name for _, _, source_name, _ in plan)
    by_rel_dir = Counter(rel_dir for _, _, _, rel_dir in plan)
    write_json(
        merge_output / "meta" / "merge_manifest.json",
        {
            "layout": "merged_sam2_2d_to_3d_workflow_v1",
            "source_root": str(source_root),
            "output_dir": str(merge_output),
            "relative_dirs": rel_dirs,
            "overwrite": overwrite,
            "renaming": "target filename = <source_dataset_name>__<original_filename>",
            "file_count": len(plan),
            "file_count_by_source": dict(sorted(by_source.items())),
            "file_count_by_relative_dir": dict(sorted(by_rel_dir.items())),
            "sources": sources,
        },
    )


def merge_datasets(args: argparse.Namespace) -> None:
    rel_dirs = merge_relative_dirs(args)
    merge_output = resolve_merge_output(args)
    sources = discover_merge_sources(
        args.source_root, merge_output, args.sources, args.source_path
    )
    sources, skipped_sources = filter_already_merged_sources(
        sources,
        merge_output,
        rel_dirs,
        args.separator,
        True,
    )
    plan = build_merge_plan(sources, merge_output, rel_dirs, args.separator)
    check_merge_plan(plan, args.overwrite)

    action = "MERGE" if args.apply else "DRY RUN"
    print(f"{action}: {args.source_root} -> {merge_output}")
    print(f"Sources: {len(sources)}")
    for source in sources:
        print(f"  - {source.name}")
    if skipped_sources:
        print(f"Skipped already merged sources: {len(skipped_sources)}")
        for source in skipped_sources:
            print(f"  - {source.name}")
    print(f"Relative dirs: {', '.join(rel_dirs)}")
    print(f"Files to copy: {len(plan)}")

    for rel_dir, count in sorted(Counter(rel_dir for _, _, _, rel_dir in plan).items()):
        print(f"  {rel_dir}: {count}")

    preview = plan[:20]
    for src_file, dst_file, _, _ in preview:
        print(f"{src_file} -> {dst_file}")
    if len(plan) > len(preview):
        print(f"... {len(plan) - len(preview)} more files")

    if not args.apply:
        print("\nDry run only. Re-run with --apply to copy files.")
        return

    copy_counts = Counter()
    for src_file, dst_file, _, _ in plan:
        copy_counts[copy_file_resumable(src_file, dst_file, args.overwrite)] += 1

    print(f"Copied: {copy_counts.get('copied', 0)}")
    print(f"Skipped existing: {copy_counts.get('skipped_existing', 0)}")
    print(f"Overwritten: {copy_counts.get('overwritten', 0)}")
    print("Merge complete.")


def collect_dataset_stems(dataset_dir: Path) -> dict[str, set[str]]:
    suffixes = {
        "bgr": ".png",
        "depth": ".png",
    }
    stems_by_dir = {}
    for rel_dir, suffix in suffixes.items():
        directory = dataset_dir / rel_dir
        if not directory.is_dir():
            raise FileNotFoundError(f"missing required directory: {directory}")
        stems_by_dir[rel_dir] = {
            path.stem for path in directory.iterdir() if path.is_file() and path.suffix == suffix
        }
    return stems_by_dir


def validate_common_stems(stems_by_dir: dict[str, set[str]]) -> list[str]:
    common = set.intersection(*stems_by_dir.values())
    problems = []
    for rel_dir, stems in stems_by_dir.items():
        missing = sorted(common - stems)
        extra = sorted(stems - common)
        if missing:
            problems.append(f"{rel_dir} missing {len(missing)} common stems, e.g. {missing[:5]}")
        if extra:
            problems.append(f"{rel_dir} has {len(extra)} unmatched stems, e.g. {extra[:5]}")
    if problems:
        raise RuntimeError(
            "bgr/depth are not aligned; refusing to renumber:\n"
            + "\n".join(problems)
        )
    return sorted(common)


def build_renumber_plan(
    dataset_dir: Path, stems: list[str], start_index: int, digits: int
) -> list[tuple[Path, Path, Path]]:
    suffixes = {
        "bgr": ".png",
        "depth": ".png",
    }
    temp_prefix = f".renumber_tmp_{start_index}_{len(stems)}_"
    plan = []
    for idx, old_stem in enumerate(stems):
        new_stem = f"{start_index + idx:0{digits}d}"
        for rel_dir, suffix in suffixes.items():
            src = dataset_dir / rel_dir / f"{old_stem}{suffix}"
            tmp = dataset_dir / rel_dir / f"{temp_prefix}{idx:0{digits}d}{suffix}"
            dst = dataset_dir / rel_dir / f"{new_stem}{suffix}"
            plan.append((src, tmp, dst))
    return plan


def check_renumber_plan(plan: list[tuple[Path, Path, Path]]) -> None:
    for src, tmp, _ in plan:
        if not src.exists():
            raise FileNotFoundError(f"missing source file: {src}")
        if tmp.exists():
            raise FileExistsError(f"temporary rename target already exists: {tmp}")


def renumber_dataset(args: argparse.Namespace) -> None:
    if args.dataset_dir is None:
        raise ValueError("--dataset-dir is required with --renumber")
    if args.start_index < 0:
        raise ValueError("--start-index must be >= 0")
    if args.digits < 1:
        raise ValueError("--digits must be >= 1")

    stems_by_dir = collect_dataset_stems(args.dataset_dir)
    stems = validate_common_stems(stems_by_dir)
    plan = build_renumber_plan(args.dataset_dir, stems, args.start_index, args.digits)
    check_renumber_plan(plan)

    action = "RENUMBER" if args.apply else "DRY RUN"
    print(f"{action}: {args.dataset_dir}")
    print(f"Frames: {len(stems)}")
    print(f"Output range: {args.start_index:0{args.digits}d} - {args.start_index + len(stems) - 1:0{args.digits}d}")

    preview_count = min(20, len(stems))
    for idx in range(preview_count):
        print(f"{stems[idx]} -> {args.start_index + idx:0{args.digits}d}")
    if len(stems) > preview_count:
        print(f"... {len(stems) - preview_count} more frames")

    if not args.apply:
        print("\nDry run only. Re-run with --apply to rename files.")
        return

    for src, tmp, _ in plan:
        src.rename(tmp)
    for _, tmp, dst in plan:
        tmp.rename(dst)
    print("Renumber complete.")


def main() -> None:
    args = parse_args()
    if args.merge:
        merge_datasets(args)
        return
    if args.renumber:
        renumber_dataset(args)
        return

    args.output_root.mkdir(parents=True, exist_ok=True)

    bag_files = sorted(args.bag_dir.glob("*.bag"))
    print(f"Found bags: {len(bag_files)}")
    for bag_path in bag_files:
        extract_bag(
            bag_path=bag_path,
            output_root=args.output_root,
            save_interval=args.save_interval,
            overwrite=args.overwrite,
        )

    if args.merge_after_extract:
        if not bag_files:
            print("\nNo bag outputs to merge.")
        else:
            merge_args = argparse.Namespace(**vars(args))
            merge_args.source_root = args.output_root
            merge_args.sources = [bag_path.stem for bag_path in bag_files]
            merge_args.apply = True
            print("\nMERGE AFTER EXTRACT")
            merge_datasets(merge_args)

    print("\nALL DONE")


if __name__ == "__main__":
    main()
