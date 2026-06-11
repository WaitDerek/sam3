"""Create RGB-resolution workflow data by directly resizing depth to color size.

This intentionally does NOT use depth/RGB extrinsics or z-buffering. It simply
loads the original color image, resizes the Depth16 PNG to the color image
resolution, and writes bgr/depth/intrinsic.json so existing point-cloud scripts
can run on the RGB-resolution grid.

Example:
    python resize_depth_to_rgb.py \
      --color dataset/demo/20260604095608625_W4000_H3000.jpg \
      --depth dataset/demo_prepared/depth/20260604095608625.png \
      --output-dir dataset/demo_rgb_resized_depth

The output stem is inferred from the color filename, depth_scale defaults to
10000 for this sensor family, and existing outputs are overwritten by default.
"""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

RGB_RAW_RE = re.compile(r"(?P<stem>.+)_W(?P<width>\d+)_H(?P<height>\d+)_RGB8Planar\.raw$")


def load_color(path: Path) -> tuple[np.ndarray, str]:
    if path.suffix.lower() == ".raw":
        match = RGB_RAW_RE.match(path.name)
        if not match:
            raise ValueError(f"cannot parse RGB8Planar raw name: {path.name}")
        width = int(match.group("width"))
        height = int(match.group("height"))
        data = np.fromfile(path, dtype=np.uint8)
        expected = width * height * 3
        if data.size != expected:
            raise ValueError(f"RGB raw size mismatch: got {data.size}, expected {expected}")
        planes = data.reshape((3, height, width))
        rgb = np.moveaxis(planes, 0, -1)
        return rgb[:, :, ::-1].copy(), match.group("stem")

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"could not read color image: {path}")
    stem = re.sub(r"_W\d+_H\d+$", "", path.stem)
    return bgr, stem


def load_rgb_intrinsics(mfa_path: Path | None, rgb_w: int, rgb_h: int) -> dict:
    if mfa_path is None or not mfa_path.exists():
        return {"width": rgb_w, "height": rgb_h}
    root = ET.parse(mfa_path).getroot()
    for block in root:
        rgb_camera = block.find("RGB_Camera")
        if rgb_camera is None:
            continue
        if int(rgb_camera.attrib.get("img_w", -1)) != rgb_w:
            continue
        if int(rgb_camera.attrib.get("img_h", -1)) != rgb_h:
            continue
        intr = block.find("RGB_correct/Intrins") or block.find("RGB/Intrins")
        if intr is None:
            continue
        return {
            "width": rgb_w,
            "height": rgb_h,
            "fx": float(intr.attrib["fx"]),
            "fy": float(intr.attrib["fy"]),
            "cx": float(intr.attrib["cx"]),
            "cy": float(intr.attrib["cy"]),
            "source": str(mfa_path),
        }
    return {"width": rgb_w, "height": rgb_h, "source": str(mfa_path)}


def find_default_mfa(paths: list[Path]) -> Path | None:
    checked: list[Path] = []
    for path in paths:
        for parent in [path, *path.parents]:
            if parent in checked:
                continue
            checked.append(parent)
            matches = sorted(parent.glob("Sensor_*.mfa"))
            if matches:
                return matches[0]
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Directly resize depth to RGB resolution.")
    parser.add_argument("--color", type=Path, required=True, help="JPG/PNG/BMP or *_RGB8Planar.raw")
    parser.add_argument("--depth", type=Path, required=True, help="16-bit depth PNG")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", default=None)
    parser.add_argument("--mfa", type=Path, default=None, help="Optional calibration file. If omitted, searches nearby Sensor_*.mfa.")
    parser.add_argument("--depth-scale", type=float, default=10000.0)
    parser.add_argument("--interpolation", choices=["nearest", "linear"], default="nearest")
    parser.add_argument("--no-overwrite", action="store_true", help="Fail if output files already exist. Default overwrites.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bgr, parsed_stem = load_color(args.color)
    depth = cv2.imread(str(args.depth), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"could not read depth image: {args.depth}")
    if depth.ndim == 3:
        depth = depth[:, :, 0]

    rgb_h, rgb_w = bgr.shape[:2]
    interp = cv2.INTER_NEAREST if args.interpolation == "nearest" else cv2.INTER_LINEAR
    resized_depth = cv2.resize(depth, (rgb_w, rgb_h), interpolation=interp)
    if resized_depth.dtype != depth.dtype:
        resized_depth = np.clip(resized_depth, 0, np.iinfo(depth.dtype).max).astype(depth.dtype)

    stem = args.stem or parsed_stem
    bgr_dir = args.output_dir / "bgr"
    depth_dir = args.output_dir / "depth"
    bgr_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    bgr_out = bgr_dir / f"{stem}.png"
    depth_out = depth_dir / f"{stem}.png"
    if args.no_overwrite and (bgr_out.exists() or depth_out.exists()):
        raise FileExistsError("output exists; omit --no-overwrite to replace it")
    cv2.imwrite(str(bgr_out), bgr)
    cv2.imwrite(str(depth_out), resized_depth)

    mfa_path = args.mfa or find_default_mfa([args.color.parent, args.depth.parent, args.output_dir])
    intrinsics = load_rgb_intrinsics(mfa_path, rgb_w, rgb_h)
    intrinsics.update(
        {
            "alignment_mode": "depth_resized_to_rgb_no_extrinsics",
            "depth_scale": args.depth_scale,
            "depth_resize_interpolation": args.interpolation,
            "original_depth_width": int(depth.shape[1]),
            "original_depth_height": int(depth.shape[0]),
        }
    )
    if mfa_path is not None:
        intrinsics["mfa"] = str(mfa_path)
    intrinsic_out = args.output_dir / "intrinsic.json"
    intrinsic_out.write_text(json.dumps(intrinsics, indent=2), encoding="utf-8")
    print(f"bgr: {bgr_out} shape={rgb_w}x{rgb_h}")
    print(f"depth: {depth_out} resized_from={depth.shape[1]}x{depth.shape[0]} interpolation={args.interpolation}")
    print(f"intrinsics: {intrinsic_out}")


if __name__ == "__main__":
    main()
