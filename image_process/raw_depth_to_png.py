"""Convert vendor Depth16 RAW files to workflow depth PNGs.

Example:
    python raw_depth_to_png.py \
      dataset/demo/20260604095608625_W1920_H1456_Depth16.raw \
      --output-dir dataset/demo_prepared/depth \
      --mfa dataset/demo/Sensor_00DA2021501.mfa \
      --write-intrinsics dataset/demo_prepared/intrinsic.json
"""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

RAW_RE = re.compile(r"(?P<stem>.+)_W(?P<width>\d+)_H(?P<height>\d+)_Depth16\.raw$")


def parse_raw_name(path: Path) -> tuple[str, int, int]:
    match = RAW_RE.match(path.name)
    if not match:
        raise ValueError(f"cannot parse width/height from depth raw name: {path.name}")
    return match.group("stem"), int(match.group("width")), int(match.group("height"))


def attrs_to_float_dict(node: ET.Element, keys: tuple[str, ...]) -> dict[str, float]:
    return {key: float(node.attrib[key]) for key in keys if key in node.attrib}


def load_depth_intrinsics(mfa_path: Path | None, width: int, height: int) -> dict:
    if mfa_path is None or not mfa_path.exists():
        return {"width": width, "height": height}

    root = ET.parse(mfa_path).getroot()
    for block in root:
        camera = block.find("Camera")
        left_correct = block.find("Left_correct")
        if camera is None or left_correct is None:
            continue
        if int(camera.attrib.get("img_w", -1)) != width:
            continue
        if int(camera.attrib.get("img_h", -1)) != height:
            continue
        intr = left_correct.find("Intrins")
        if intr is None:
            continue
        data = attrs_to_float_dict(intr, ("fx", "fy", "cx", "cy"))
        data.update({"width": width, "height": height, "source": str(mfa_path)})
        return data

    return {"width": width, "height": height, "source": str(mfa_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Depth16 RAW to 16-bit PNG depth.")
    parser.add_argument("raw", type=Path, help="Input *_W<width>_H<height>_Depth16.raw file")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for depth PNG")
    parser.add_argument("--output", type=Path, default=None, help="Explicit output PNG path")
    parser.add_argument("--stem", default=None, help="Output stem. Defaults to timestamp prefix from filename")
    parser.add_argument("--endian", choices=["little", "big"], default="little")
    parser.add_argument("--mfa", type=Path, default=None, help="Optional Sensor_*.mfa calibration file")
    parser.add_argument("--write-intrinsics", type=Path, default=None, help="Optional intrinsic.json output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stem, width, height = parse_raw_name(args.raw)
    stem = args.stem or stem
    dtype = "<u2" if args.endian == "little" else ">u2"
    data = np.fromfile(args.raw, dtype=np.dtype(dtype))
    expected = width * height
    if data.size != expected:
        raise ValueError(f"raw size mismatch: got {data.size} uint16 values, expected {expected}")
    depth = data.reshape((height, width))

    if args.output is not None:
        out_path = args.output
    else:
        out_dir = args.output_dir or args.raw.parent / "depth"
        out_path = out_dir / f"{stem}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), depth)
    print(f"depth: {out_path} shape={width}x{height} dtype=uint16 min={int(depth.min())} max={int(depth.max())}")

    if args.write_intrinsics is not None:
        intrinsics = load_depth_intrinsics(args.mfa, width, height)
        args.write_intrinsics.parent.mkdir(parents=True, exist_ok=True)
        args.write_intrinsics.write_text(json.dumps(intrinsics, indent=2), encoding="utf-8")
        print(f"intrinsics: {args.write_intrinsics}")


if __name__ == "__main__":
    main()
