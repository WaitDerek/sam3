"""Resolve one image pixel to XYZ using aligned depth and camera intrinsics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bgr_depth_to_pcd import DEFAULT_DATASET_DIR, load_depth, load_intrinsics
from mask_depth_to_pcd import load_mask


def parse_pixel(text: str) -> tuple[int, int]:
    try:
        x_text, y_text = text.split(",", 1)
        return int(x_text.strip()), int(y_text.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--pixel must look like x,y") from exc


def normalize_frame_stem(frame: str, mask_suffix: str) -> str:
    stem = Path(frame).stem
    suffixes = [mask_suffix, "_mask", "_det"]
    for suffix in suffixes:
        if suffix and stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Look up the XYZ coordinate of one image pixel from aligned depth."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--depth-dir", type=Path, default=None)
    parser.add_argument("--mask-dir", type=Path, default=None)
    parser.add_argument("--frame", required=True, help="Frame stem, png name, *_det.jpg, or *_mask.png.")
    parser.add_argument("--pixel", type=parse_pixel, required=True, help="Pixel coordinate as x,y.")
    parser.add_argument("--intrinsics-json", type=Path, default=None)
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--min-depth", type=float, default=0.0)
    parser.add_argument("--max-depth", type=float, default=0.0)
    parser.add_argument(
        "--search-radius",
        type=int,
        default=0,
        help="If the exact pixel has invalid depth, search the nearest valid pixel inside this radius.",
    )
    parser.add_argument(
        "--mask-suffix",
        default="_mask",
        help="Mask filename suffix before .png when checking masks.",
    )
    parser.add_argument(
        "--check-mask",
        action="store_true",
        help="Report whether the queried pixel lies inside the frame mask when a mask file exists.",
    )
    parser.add_argument(
        "--camera-frame",
        action="store_true",
        help="Keep x-right/y-down/z-forward instead of the historical y/z flipped convention.",
    )
    parser.add_argument("--json", action="store_true", help="Print the result as JSON.")
    return parser.parse_args()


def pixel_is_valid(depth_m: np.ndarray, x: int, y: int, min_depth: float, max_depth: float) -> bool:
    z = float(depth_m[y, x])
    if z <= 0:
        return False
    if min_depth > 0 and z < min_depth:
        return False
    if max_depth > 0 and z > max_depth:
        return False
    return True


def find_sample_pixel(
    depth_m: np.ndarray,
    x: int,
    y: int,
    radius: int,
    min_depth: float,
    max_depth: float,
) -> tuple[int, int]:
    height, width = depth_m.shape[:2]
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError(f"pixel out of range: ({x}, {y}) for image size {width}x{height}")

    if pixel_is_valid(depth_m, x, y, min_depth, max_depth):
        return x, y

    best: tuple[float, int, int] | None = None
    for yy in range(max(0, y - radius), min(height, y + radius + 1)):
        for xx in range(max(0, x - radius), min(width, x + radius + 1)):
            if not pixel_is_valid(depth_m, xx, yy, min_depth, max_depth):
                continue
            dist2 = float((xx - x) ** 2 + (yy - y) ** 2)
            if best is None or dist2 < best[0]:
                best = (dist2, xx, yy)

    if best is None:
        raise ValueError(
            f"no valid depth found for ({x}, {y}) within radius {radius}"
        )
    return best[1], best[2]


def pixel_to_xyz(
    x: int,
    y: int,
    z: float,
    intrinsics: dict,
    legacy_transform: bool,
) -> tuple[float, float, float]:
    world_x = (float(x) - float(intrinsics["cx"])) * z / float(intrinsics["fx"])
    world_y = (float(y) - float(intrinsics["cy"])) * z / float(intrinsics["fy"])
    world_z = z
    if legacy_transform:
        world_y = -world_y
        world_z = -world_z
    return world_x, world_y, world_z


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir
    depth_dir = args.depth_dir or dataset_dir / "depth"
    mask_dir = args.mask_dir or dataset_dir / "masks"
    stem = normalize_frame_stem(args.frame, args.mask_suffix)

    depth_path = depth_dir / f"{stem}.png"
    if not depth_path.is_file():
        raise FileNotFoundError(f"missing depth image: {depth_path}")

    depth_raw = load_depth(depth_path)
    depth_m = depth_raw.astype(np.float32) / args.depth_scale
    intrinsics = load_intrinsics(dataset_dir, args.intrinsics_json)

    query_x, query_y = args.pixel
    used_x, used_y = find_sample_pixel(
        depth_m=depth_m,
        x=query_x,
        y=query_y,
        radius=max(0, args.search_radius),
        min_depth=args.min_depth,
        max_depth=args.max_depth,
    )
    depth_raw_value = int(depth_raw[used_y, used_x])
    depth_m_value = float(depth_m[used_y, used_x])
    xyz = pixel_to_xyz(
        x=used_x,
        y=used_y,
        z=depth_m_value,
        intrinsics=intrinsics,
        legacy_transform=not args.camera_frame,
    )

    result = {
        "frame": stem,
        "query_pixel": {"x": query_x, "y": query_y},
        "used_pixel": {"x": used_x, "y": used_y},
        "depth_raw": depth_raw_value,
        "depth_m": depth_m_value,
        "xyz": {"x": xyz[0], "y": xyz[1], "z": xyz[2]},
        "coordinate_frame": "camera" if args.camera_frame else "legacy_pcd",
    }

    if args.check_mask and mask_dir.is_dir():
        mask_path = mask_dir / f"{stem}{args.mask_suffix}.png"
        if mask_path.is_file():
            mask = load_mask(mask_path)
            result["mask_hit"] = bool(mask[query_y, query_x]) if (
                0 <= query_x < mask.shape[1] and 0 <= query_y < mask.shape[0]
            ) else False
            result["used_pixel_mask_hit"] = bool(mask[used_y, used_x])
        else:
            result["mask_hit"] = None
            result["used_pixel_mask_hit"] = None

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print(f"frame: {result['frame']}")
    print(f"query_pixel: ({query_x}, {query_y})")
    if (used_x, used_y) != (query_x, query_y):
        print(f"used_pixel: ({used_x}, {used_y})")
    print(f"depth_raw: {depth_raw_value}")
    print(f"depth_m: {depth_m_value:.6f}")
    print(
        "xyz: "
        f"x={xyz[0]:.6f}, "
        f"y={xyz[1]:.6f}, "
        f"z={xyz[2]:.6f} "
        f"({result['coordinate_frame']})"
    )
    if "mask_hit" in result:
        print(f"mask_hit: {result['mask_hit']}")
        print(f"used_pixel_mask_hit: {result['used_pixel_mask_hit']}")


if __name__ == "__main__":
    main()
