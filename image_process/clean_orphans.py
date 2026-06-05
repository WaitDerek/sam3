"""Delete image_process/out detection files whose source image is gone.

Mirrors INPUT_DIRS from clean_container_fps.py. For each material's metadata
folder, every stem whose source PNG no longer exists under input_dir is removed
from all three output subtrees. Prints a per-material count.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from material_paths import TOOL_DIR

ROOT = TOOL_DIR
sys.path.insert(0, str(TOOL_DIR))
from clean_container_fps import INPUT_DIRS  # noqa: E402


def clean_material(material: str, dry_run: bool) -> tuple[int, int]:
    input_dir = INPUT_DIRS[material]
    out_dir = ROOT / "out" / material
    meta_dir = out_dir / "metadata"
    overlay_dir = out_dir / "overlays"
    mask_dir = out_dir / "masks"

    # Safety: if the source directory itself doesn't exist (likely renamed),
    # refuse to remove anything for this material. Otherwise a stale INPUT_DIRS
    # entry would wipe the entire output tree.
    if not input_dir.exists():
        print(
            f"{material}: WARNING input_dir missing ({input_dir}) — skipping; "
            "fix INPUT_DIRS before re-running"
        )
        return 0, 0

    total = 0
    removed = 0
    for f in sorted(meta_dir.glob("*.json")):
        total += 1
        stem = f.stem
        if (input_dir / f"{stem}.png").exists():
            continue
        removed += 1
        if dry_run:
            continue
        for path in (
            overlay_dir / f"{stem}_det.jpg",
            mask_dir / f"{stem}_mask.png",
            meta_dir / f"{stem}.json",
        ):
            if path.exists():
                path.unlink()
    return total, removed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--material", action="append", dest="materials")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    materials = args.materials or sorted(INPUT_DIRS.keys())

    grand_total = 0
    grand_removed = 0
    for m in materials:
        total, removed = clean_material(m, args.dry_run)
        grand_total += total
        grand_removed += removed
        verb = "would remove" if args.dry_run else "removed"
        print(f"{m}: scanned={total} {verb}={removed}")
    print(f"-- total: scanned={grand_total} removed={grand_removed}")


if __name__ == "__main__":
    main()
