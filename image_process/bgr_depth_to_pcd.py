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


def pack_rgb_as_float(r: int, g: int, b: int) -> float:
    rgb_uint = (int(r) << 16) | (int(g) << 8) | int(b)
    return struct.unpack("f", struct.pack("I", rgb_uint))[0]


def rgbd_to_points(
    bgr: np.ndarray,
    depth_raw: np.ndarray,
    intrinsics: dict,
    depth_scale: float,
    min_depth: float,
    max_depth: float,
    legacy_transform: bool,
) -> tuple[np.ndarray, np.ndarray]:
    depth_m = depth_raw.astype(np.float32) / depth_scale
    valid = depth_m > 0
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


def find_frames(bgr_dir: Path, depth_dir: Path, frame: str | None) -> list[str]:
    if frame:
        stem = Path(frame).stem
        bgr_path = bgr_dir / f"{stem}.png"
        depth_path = depth_dir / f"{stem}.png"
        if not bgr_path.exists() or not depth_path.exists():
            raise FileNotFoundError(f"missing aligned bgr/depth frame: {stem}")
        return [stem]

    frames = []
    for bgr_path in sorted(bgr_dir.glob("*.png")):
        if (depth_dir / bgr_path.name).exists():
            frames.append(bgr_path.stem)
    if not frames:
        raise FileNotFoundError(f"no aligned bgr/depth frames found in {bgr_dir} and {depth_dir}")
    return frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate colored PCD files from bgr/depth folders.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--bgr-dir", type=Path, default=None)
    parser.add_argument("--depth-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--frame", default=None, help="Frame stem or png name. Omit to process all aligned frames.")
    parser.add_argument("--intrinsics-json", type=Path, default=None)
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--min-depth", type=float, default=0.0)
    parser.add_argument("--max-depth", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
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
    output_dir = args.output_dir or dataset_dir / "pcd"
    intrinsics = load_intrinsics(dataset_dir, args.intrinsics_json)

    for name, directory in [("bgr", bgr_dir), ("depth", depth_dir)]:
        if not directory.is_dir():
            raise FileNotFoundError(f"missing {name} dir: {directory}")

    frames = find_frames(bgr_dir, depth_dir, args.frame)
    written = 0
    skipped = 0
    for frame in frames:
        pcd_path = output_dir / f"{frame}.pcd"
        if pcd_path.exists() and not args.overwrite:
            print(f"skip existing pcd: {pcd_path}")
            skipped += 1
            continue

        bgr = load_bgr(bgr_dir / f"{frame}.png")
        depth = load_depth(depth_dir / f"{frame}.png")
        points, colors = rgbd_to_points(
            bgr=bgr,
            depth_raw=depth,
            intrinsics=intrinsics,
            depth_scale=args.depth_scale,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            legacy_transform=not args.camera_frame,
        )
        if len(points) == 0:
            print(f"skip empty depth frame: {frame}")
            skipped += 1
            continue

        write_colored_pcd(pcd_path, points, colors)
        print(f"PCD: {pcd_path} ({len(points)} points)")
        written += 1

    print(f"done: written={written}, skipped={skipped}, total={len(frames)}")


if __name__ == "__main__":
    main()
