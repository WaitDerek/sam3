"""Generate per-frame object point clouds from mask + depth (+ bgr color).

Sibling of bgr_depth_to_pcd.py, but instead of exporting the whole scene it
keeps only the pixels inside each frame's binary mask (e.g. the segmented
副雨刮器 blades) that also have valid depth, back-projects them with the camera
intrinsics, and colors them from the aligned bgr frame.

This script only generates and saves <stem>.pcd files. Use visualize_pcd.py to
view any saved cloud by name, e.g.:

    python visualize_pcd.py <dataset>/pcd/20260521_154208__000204.pcd
"""

import argparse
import json
import struct
from pathlib import Path

import cv2
import numpy as np


DEFAULT_DATASET_DIR = Path(__file__).resolve().parent / "dataset" / "test"
DEFAULT_INTRINSICS = {
    "width": 1280,
    "height": 720,
    "fx": 662.636,
    "fy": 662.636,
    "cx": 635.522,
    "cy": 348.738,
}


def parse_legacy_intrinsics(text: str) -> dict:
    intrinsics = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key in {"width", "height"}:
            intrinsics[key] = int(float(value))
        elif key in {"fx", "fy", "cx", "cy"}:
            intrinsics[key] = float(value)
    return intrinsics


def load_intrinsics(dataset_dir: Path, intrinsics_json: Path | None) -> dict:
    candidates = []
    if intrinsics_json is not None:
        candidates.append(intrinsics_json)
    candidates.extend(
        [
            dataset_dir / "meta" / "intrinsic.json",
            dataset_dir / "intrinsic.json",
            Path(__file__).resolve().parent / "intrinsic.json",
        ]
    )

    intrinsics = dict(DEFAULT_INTRINSICS)
    for path in candidates:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = parse_legacy_intrinsics(text)
        for key in ["width", "height", "fx", "fy", "cx", "cy"]:
            if key in data:
                intrinsics[key] = data[key]
        return intrinsics
    return intrinsics


def load_bgr(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"could not read bgr image: {path}")
    return bgr


def load_depth(path: Path) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"could not read depth image: {path}")
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    return depth


def load_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"could not read mask image: {path}")
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask > 127


def pack_rgb_as_float(r: int, g: int, b: int) -> float:
    rgb_uint = (int(r) << 16) | (int(g) << 8) | int(b)
    return struct.unpack("f", struct.pack("I", rgb_uint))[0]


def masked_rgbd_to_points(
    bgr: np.ndarray,
    depth_raw: np.ndarray,
    mask: np.ndarray,
    intrinsics: dict,
    depth_scale: float,
    min_depth: float,
    max_depth: float,
    legacy_transform: bool,
) -> tuple[np.ndarray, np.ndarray]:
    depth_m = depth_raw.astype(np.float32) / depth_scale
    valid = mask & (depth_m > 0)
    if min_depth > 0:
        valid &= depth_m >= min_depth
    if max_depth > 0:
        valid &= depth_m <= max_depth

    v, u = np.nonzero(valid)
    if len(u) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    z = depth_m[v, u]
    x = (u.astype(np.float32) - float(intrinsics["cx"])) * z / float(intrinsics["fx"])
    y = (v.astype(np.float32) - float(intrinsics["cy"])) * z / float(intrinsics["fy"])

    if legacy_transform:
        y = -y
        z = -z

    points = np.stack([x, y, z], axis=1).astype(np.float32)
    colors_rgb = bgr[v, u][:, ::-1].astype(np.uint8)
    return points, colors_rgb


def write_colored_pcd(path: Path, points: np.ndarray, colors_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        f.write("FIELDS x y z rgb\n")
        f.write("SIZE 4 4 4 4\n")
        f.write("TYPE F F F F\n")
        f.write("COUNT 1 1 1 1\n")
        f.write(f"WIDTH {len(points)}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {len(points)}\n")
        f.write("DATA ascii\n")
        for (x, y, z), (r, g, b) in zip(points, colors_rgb):
            rgb = pack_rgb_as_float(int(r), int(g), int(b))
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {rgb:.8e}\n")


def write_binary_colored_pcd(path: Path, points: np.ndarray, colors_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb_uint = (
        (colors_rgb[:, 0].astype(np.uint32) << 16)
        | (colors_rgb[:, 1].astype(np.uint32) << 8)
        | colors_rgb[:, 2].astype(np.uint32)
    )
    rgb_float = rgb_uint.view(np.float32)
    rows = np.empty((len(points), 4), dtype=np.float32)
    rows[:, :3] = points.astype(np.float32, copy=False)
    rows[:, 3] = rgb_float

    with path.open("wb") as f:
        header = (
            "# .PCD v0.7 - Point Cloud Data file format\n"
            "VERSION 0.7\n"
            "FIELDS x y z rgb\n"
            "SIZE 4 4 4 4\n"
            "TYPE F F F F\n"
            "COUNT 1 1 1 1\n"
            f"WIDTH {len(points)}\n"
            "HEIGHT 1\n"
            "VIEWPOINT 0 0 0 1 0 0 0\n"
            f"POINTS {len(points)}\n"
            "DATA binary\n"
        )
        f.write(header.encode("ascii"))
        f.write(rows.tobytes())


def find_frames(
    bgr_dir: Path,
    depth_dir: Path,
    mask_dir: Path,
    mask_suffix: str,
    frame: str | None,
) -> list[str]:
    def mask_path(stem: str) -> Path:
        return mask_dir / f"{stem}{mask_suffix}.png"

    if frame:
        stem = Path(frame).stem
        for suffix in (mask_suffix, "_mask", "_det", ""):
            if stem.endswith(suffix) and suffix:
                stem = stem[: -len(suffix)]
                break
        missing = [
            name
            for name, path in [
                ("bgr", bgr_dir / f"{stem}.png"),
                ("depth", depth_dir / f"{stem}.png"),
                ("mask", mask_path(stem)),
            ]
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"frame {stem} missing {', '.join(missing)}"
            )
        return [stem]

    frames = []
    for bgr_path in sorted(bgr_dir.glob("*.png")):
        stem = bgr_path.stem
        if (depth_dir / f"{stem}.png").exists() and mask_path(stem).exists():
            frames.append(stem)
    if not frames:
        raise FileNotFoundError(
            f"no frames with aligned bgr/depth/mask in {bgr_dir}, {depth_dir}, {mask_dir}"
        )
    return frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate per-frame object point clouds from mask + depth (+ bgr color)."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--bgr-dir", type=Path, default=None)
    parser.add_argument("--depth-dir", type=Path, default=None)
    parser.add_argument("--mask-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--mask-suffix",
        default="_mask",
        help="Mask filename suffix before .png (default: _mask, i.e. <stem>_mask.png).",
    )
    parser.add_argument(
        "--frame",
        default=None,
        help="Frame stem or file name. Omit to process all aligned frames.",
    )
    parser.add_argument("--intrinsics-json", type=Path, default=None)
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--min-depth", type=float, default=0.0)
    parser.add_argument("--max-depth", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--binary",
        action="store_true",
        help="Write binary PCD instead of ASCII PCD to reduce file size and write time.",
    )
    parser.add_argument(
        "--camera-frame",
        action="store_true",
        help="Keep x-right/y-down/z-forward instead of the historical y/z flipped PCD convention.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir
    bgr_dir = args.bgr_dir or dataset_dir / "bgr"
    depth_dir = args.depth_dir or dataset_dir / "depth"
    mask_dir = args.mask_dir or dataset_dir / "masks"
    output_dir = args.output_dir or dataset_dir / "pcd"
    intrinsics = load_intrinsics(dataset_dir, args.intrinsics_json)

    for name, directory in [("bgr", bgr_dir), ("depth", depth_dir), ("mask", mask_dir)]:
        if not directory.is_dir():
            raise FileNotFoundError(f"missing {name} dir: {directory}")

    frames = find_frames(bgr_dir, depth_dir, mask_dir, args.mask_suffix, args.frame)
    written = 0
    skipped = 0
    empty = 0
    for frame in frames:
        pcd_path = output_dir / f"{frame}.pcd"
        if pcd_path.exists() and not args.overwrite:
            print(f"skip existing pcd: {pcd_path}")
            skipped += 1
            continue

        bgr = load_bgr(bgr_dir / f"{frame}.png")
        depth = load_depth(depth_dir / f"{frame}.png")
        mask = load_mask(mask_dir / f"{frame}{args.mask_suffix}.png")
        if not (bgr.shape[:2] == depth.shape[:2] == mask.shape[:2]):
            raise ValueError(
                f"shape mismatch for {frame}: "
                f"bgr={bgr.shape[:2]} depth={depth.shape[:2]} mask={mask.shape[:2]}"
            )

        points, colors = masked_rgbd_to_points(
            bgr=bgr,
            depth_raw=depth,
            mask=mask,
            intrinsics=intrinsics,
            depth_scale=args.depth_scale,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            legacy_transform=not args.camera_frame,
        )
        if len(points) == 0:
            print(f"skip empty (mask has no valid depth): {frame}")
            empty += 1
            continue

        if args.binary:
            write_binary_colored_pcd(pcd_path, points, colors)
        else:
            write_colored_pcd(pcd_path, points, colors)
        print(f"PCD: {pcd_path} ({len(points)} points)")
        written += 1

    print(
        f"done: written={written}, skipped={skipped}, "
        f"empty={empty}, total={len(frames)}"
    )


if __name__ == "__main__":
    main()
