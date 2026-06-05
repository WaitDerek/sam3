"""Restore the original empty-state outputs for images in a repair empty-list.

For each image stem in image_process/out/_logs/_repair/<material>_empty.txt:
  - rewrites metadata to {"image": ..., "label": ..., "prompts": [], "detections": []}
  - rewrites mask as all-zero PNG
  - rewrites overlay as the source image with no drawing
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from material_paths import TOOL_DIR

ROOT = TOOL_DIR


def revert(material: str, label: str, input_dir: Path, output_dir: Path) -> None:
    empty_list = ROOT / "out/_logs/_repair" / f"{material}_empty.txt"
    stems = [
        line.strip()
        for line in empty_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    overlay_dir = output_dir / "overlays"
    mask_dir = output_dir / "masks"
    meta_dir = output_dir / "metadata"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    restored = 0
    missing = 0
    for stem in stems:
        image_path = input_dir / f"{stem}.png"
        if not image_path.exists():
            missing += 1
            continue
        rgb = np.array(Image.open(image_path).convert("RGB"))
        h, w = rgb.shape[:2]
        Image.fromarray(rgb).save(overlay_dir / f"{stem}_det.jpg", quality=92)
        Image.fromarray(np.zeros((h, w), dtype=np.uint8)).save(
            mask_dir / f"{stem}_mask.png"
        )
        (meta_dir / f"{stem}.json").write_text(
            json.dumps(
                {
                    "image": str(image_path),
                    "label": label,
                    "prompts": [],
                    "detections": [],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        restored += 1
    print(f"{material}: restored={restored}, missing={missing}, total={len(stems)}")


MATERIALS = {
    "fresh_air_vent": {
        "label": "空调新风口",
        "input_dir": ROOT / "dataset/空调新风口总成_906/bgr",
        "output_dir": ROOT / "out/fresh_air_vent",
    },
    "bumper_624": {
        "label": "保险杠",
        "input_dir": ROOT / "dataset/前保险杠侧安装支架总成_624/bgr",
        "output_dir": ROOT / "out/bumper_563",
    },
    "power_tailgate_system_854": {
        "label": "背门自动开闭系统",
        "input_dir": ROOT / "dataset/背门自动开闭系统ECU控制器总成_854/bgr",
        "output_dir": ROOT / "out/power_tailgate_system_797",
    },
    "screw_cover": {
        "label": "螺钉盖板",
        "input_dir": ROOT / "dataset/前门内开手柄盒螺钉盖板",
        "output_dir": ROOT / "out/screw_cover",
    },
    "passenger_wiper_1113": {
        "label": "副雨刮器",
        "input_dir": ROOT / "dataset/副雨刮器总成_1113/bgr",
        "output_dir": ROOT / "out/passenger_wiper_1100",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("materials", nargs="+", choices=sorted(MATERIALS.keys()))
    args = parser.parse_args()
    for material in args.materials:
        cfg = MATERIALS[material]
        revert(
            material=material,
            label=cfg["label"],
            input_dir=cfg["input_dir"],
            output_dir=cfg["output_dir"],
        )


if __name__ == "__main__":
    main()
