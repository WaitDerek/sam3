"""Drop container-FP detections and regenerate mask/overlay from remaining ones.

Filters applied per detection:
  - prompt is in PURE_CONTAINER_BLACKLIST -> drop
  - area_fraction > per-material MAX_AREA_FRAC -> drop (catches whole-crate masks)

If an image has detections left, mask = union(kept) and overlay is rebuilt.
If nothing is left, metadata becomes empty and overlay/mask become the no-detection
baseline (original image / all-zero mask).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parent

PURE_CONTAINER_BLACKLIST = {
    "gray plastic storage crate",
    "gray plastic grid storage crate",
    "cardboard storage box with dividers",
    "tan cardboard crate with compartments",
    "any object inside or on top of crate",
    "any object inside crate",
    "any object inside cardboard crate",
    "any object on table",
    "any object in storage tray",
    "any red object",
}

# Per-material minimum acceptable detection score. Detections below this are
# treated as background/edge false positives even if their area/prompt is fine.
# Distributions show clean gaps in the histogram around these values.
MIN_SCORE = {
    "power_tailgate_system_854": 0.10,
    "air_filter": 0.12,
    "foam_block_979": 0.20,
}


# Per-material upper bound on bbox_fill_fraction. For the wiper bin views the
# gray plastic comb-rack scores well but produces masks that fill 18%+ of
# their bbox, while real thin curved rubber blades sit at 5-15%. Set via
# prefix overrides so the table_view close-ups stay unaffected.
MAX_BBOX_FILL: dict[str, float] = {}

MAX_BBOX_FILL_PREFIX_OVERRIDES = {
    ("passenger_wiper_1113", "20260521_153324"): 0.18,
    ("passenger_wiper_1113", "20260521_154208"): 0.18,
}


def max_bbox_fill_for(material: str, stem: str = "") -> float:
    for (mat, prefix), cap in MAX_BBOX_FILL_PREFIX_OVERRIDES.items():
        if mat == material and stem.startswith(prefix):
            return cap
    return MAX_BBOX_FILL.get(material, 1.0)


# Materials where a detection touching the image border is always a FP (the real
# target is never at the edge of the frame for these capture setups).
DROP_BORDER_TOUCHING = {
    "foam_block_979": 2,  # pixels of slack on each border
}


def min_score_for(material: str) -> float:
    return MIN_SCORE.get(material, 0.0)


# Per-material cap on detection area_fraction. Anything above is almost certainly
# the crate/box itself rather than the target part.
MAX_AREA_FRAC = {
    "air_filter": 0.20,
    "bumper_624": 0.02,
    "washer_filler_769": 0.20,
    "foam_block_979": 0.15,
    "fresh_air_vent": 0.08,
    "passenger_window_switch": 0.10,
    "passenger_wiper_1113": 0.08,
    "power_tailgate_system_854": 0.06,
    "screw_cover": 0.02,
    "warning_triangle_486": 0.15,
}

# Prefix overrides — close-up table_view shots have the part filling 20-40% of
# the frame, so the conservative bin-view caps would drop valid detections.
# Keys are (material, image_stem prefix); values override MAX_AREA_FRAC.
PREFIX_AREA_OVERRIDES = {
    ("fresh_air_vent", "20260521_143611"): 0.45,
    ("fresh_air_vent", "20260521_144208"): 0.45,
    ("fresh_air_vent", "20260521_144718"): 0.45,
    ("bumper_624", "20260521_112611"): 0.10,
    ("bumper_624", "20260521_113058"): 0.10,
    ("bumper_624", "20260521_113346"): 0.10,
    ("bumper_624", "20260521_113506"): 0.20,
    ("bumper_624", "20260522_133438"): 0.10,
    ("power_tailgate_system_854", "20260521_140702"): 0.20,
    ("power_tailgate_system_854", "20260521_141129"): 0.20,
    ("power_tailgate_system_854", "20260521_141640"): 0.20,
    ("power_tailgate_system_854", "20260521_142146"): 0.20,
    ("air_filter", "20260522_114115"): 0.08,
    ("air_filter", "20260522_114208"): 0.08,
    ("washer_filler_769", "20260520_165857"): 0.05,
    ("washer_filler_769", "20260520_165906"): 0.05,
    ("passenger_wiper_1113", "20260521_153324"): 0.022,
    ("passenger_wiper_1113", "20260521_154208"): 0.022,
}


def cap_for(material: str, stem: str) -> float:
    for (mat, prefix), cap in PREFIX_AREA_OVERRIDES.items():
        if mat == material and stem.startswith(prefix):
            return cap
    return MAX_AREA_FRAC.get(material, 0.25)

# Where the source images live (mirrors REPAIRS in repair_material_segmentation.py).
INPUT_DIRS = {
    "air_filter": ROOT / "data/空滤器进气连接管总成_560",
    "bumper_624": ROOT / "data/前保险杠侧安装支架总成_624/bgr",
    "washer_filler_769": ROOT / "data/洗涤器水壶加注管总成_769",
    "foam_block_979": ROOT / "data/后轮鼓包内后侧泡沫块_979/bgr",
    "fresh_air_vent": ROOT / "data/空调新风口总成_906/bgr",
    "passenger_window_switch": ROOT / "data/副电动车窗开关总成_697",
    "passenger_wiper_1113": ROOT / "data/副雨刮器总成_1113/bgr",
    "power_tailgate_system_854": ROOT / "data/背门自动开闭系统ECU控制器总成_854/bgr",
    "screw_cover": ROOT / "data/前门内开手柄盒螺钉盖板_908",
    "warning_triangle_486": ROOT / "data/三角警告牌_486/bgr",
}

OUTPUT_DIRS = {
    "air_filter": ROOT / "out/air_filter",
    "bumper_624": ROOT / "out/bumper_563",
    "washer_filler_769": ROOT / "out/washer_filler_768",
    "foam_block_979": ROOT / "out/foam_block_979",
    "fresh_air_vent": ROOT / "out/fresh_air_vent",
    "passenger_window_switch": ROOT / "out/passenger_window_switch",
    "passenger_wiper_1113": ROOT / "out/passenger_wiper_1100",
    "power_tailgate_system_854": ROOT / "out/power_tailgate_system_797",
    "screw_cover": ROOT / "out/screw_cover",
    "warning_triangle_486": ROOT / "out/warning_triangle_486",
}

COLOR = np.array((0, 130, 255), dtype=np.uint8)


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    out = rgb.copy()
    m = mask.astype(bool)
    out[m] = (alpha * COLOR + (1 - alpha) * rgb[m]).astype(np.uint8)
    return out


def draw_contour(rgb: np.ndarray, mask: np.ndarray, thickness: int = 2) -> np.ndarray:
    m = mask.astype(np.uint8)
    pad = np.pad(m, thickness, mode="constant")
    border = (pad[thickness:-thickness, thickness:-thickness] == 1) & ~(
        (pad[: -2 * thickness, thickness:-thickness] == 1)
        & (pad[2 * thickness :, thickness:-thickness] == 1)
        & (pad[thickness:-thickness, : -2 * thickness] == 1)
        & (pad[thickness:-thickness, 2 * thickness :] == 1)
    )
    out = rgb.copy()
    out[border] = COLOR
    return out


def load_mask_png(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L")) > 127


def reconstruct_kept_mask(
    image_path: Path,
    metadata: dict,
    kept_indices: list[int],
    rgb_shape: tuple[int, int],
) -> np.ndarray:
    """Approximate the kept-detection mask using each detection's bbox.

    The per-detection binary masks are not stored individually (only the union is
    persisted as out/<material>/masks/<stem>_mask.png). To rebuild a clean mask
    after dropping FP detections we intersect that union mask with the bounding
    boxes of the kept detections. This is exact when kept regions do not overlap
    dropped regions inside the same bbox, which holds in practice because the
    container FPs are much larger than the per-part bboxes.
    """
    h, w = rgb_shape
    union_path = image_path.parent.parent / "masks" / f"{image_path.stem.split('_det')[0]}_mask.png"
    # union_path is computed differently — see callers
    union = np.zeros((h, w), dtype=bool)
    return union


def clean_metadata(meta_path: Path, material: str) -> tuple[int, int]:
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    dets = data.get("detections", [])
    cap = cap_for(material, meta_path.stem)
    min_s = min_score_for(material)
    fill_cap = max_bbox_fill_for(material, meta_path.stem)
    border_slack = DROP_BORDER_TOUCHING.get(material)

    kept = []
    dropped = []
    for det in dets:
        prompt = det.get("prompt", "")
        af = det.get("area_fraction", 0.0)
        score = det.get("score", 0.0)
        fill = det.get("bbox_fill_fraction", 0.0)
        bbox = det.get("bbox_xyxy") or [0, 0, 0, 0]
        touches_border = bool(
            border_slack is not None
            and (bbox[0] <= border_slack or bbox[1] <= border_slack)
        )
        if (
            prompt in PURE_CONTAINER_BLACKLIST
            or af > cap
            or score < min_s
            or fill > fill_cap
            or touches_border
        ):
            dropped.append(det)
        else:
            kept.append(det)

    if not dropped:
        return 0, 0  # nothing to do

    data["detections"] = [
        dict(d, instance=i) for i, d in enumerate(kept, start=1)
    ]
    meta_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return len(dropped), len(kept)


def rebuild_outputs(material: str, stems_to_rebuild: list[str]) -> None:
    """Recompute mask + overlay PNGs from the (already cleaned) metadata.

    The original union mask cannot be split, so we approximate each kept
    detection's contribution as its bbox rectangle clipped to the union mask.
    """
    out_dir = OUTPUT_DIRS.get(material, ROOT / "out" / material)
    input_dir = INPUT_DIRS[material]
    overlay_dir = out_dir / "overlays"
    mask_dir = out_dir / "masks"
    meta_dir = out_dir / "metadata"

    for stem in stems_to_rebuild:
        meta = json.loads((meta_dir / f"{stem}.json").read_text(encoding="utf-8"))
        kept = meta.get("detections", [])
        image_path = input_dir / f"{stem}.png"
        if not image_path.exists():
            # Source image was moved/removed since the metadata was written —
            # leave existing overlay/mask alone rather than crashing the run.
            continue
        rgb = np.array(Image.open(image_path).convert("RGB"))
        h, w = rgb.shape[:2]
        union_path = mask_dir / f"{stem}_mask.png"
        old_union = (
            load_mask_png(union_path)
            if union_path.exists()
            else np.zeros((h, w), dtype=bool)
        )

        if not kept:
            new_mask = np.zeros((h, w), dtype=bool)
        else:
            new_mask = np.zeros((h, w), dtype=bool)
            for det in kept:
                bbox = det.get("bbox_xyxy")
                if not bbox:
                    continue
                x0, y0, x1, y1 = bbox
                x0 = max(0, min(w, int(x0)))
                y0 = max(0, min(h, int(y0)))
                x1 = max(0, min(w, int(x1)))
                y1 = max(0, min(h, int(y1)))
                if x1 <= x0 or y1 <= y0:
                    continue
                # Pad the tile by `margin` pixels so binary_closing has real
                # neighbors at the bbox edge. Without padding, the closing's
                # erosion step treats outside-tile pixels as 0 and eats ~2 px
                # off every side of the mask — visibly trimming clean
                # detections (washer_filler, foam, etc.).
                margin = 4
                xp0 = max(0, x0 - margin)
                yp0 = max(0, y0 - margin)
                xp1 = min(w, x1 + margin)
                yp1 = min(h, y1 + margin)
                padded_tile = old_union[yp0:yp1, xp0:xp1]
                tile_closed = ndimage.binary_closing(
                    padded_tile,
                    structure=np.ones((3, 3), dtype=bool),
                    iterations=2,
                )
                # Only fill holes that are clearly small (≤ 1% of the bbox
                # area). Larger "holes" usually are real background, not part
                # of the target.
                bbox_area = (y1 - y0) * (x1 - x0)
                max_hole_area = max(1, int(0.01 * bbox_area))
                filled = ndimage.binary_fill_holes(tile_closed)
                holes = filled & ~tile_closed
                labeled, n_components = ndimage.label(holes)
                if n_components:
                    sizes = ndimage.sum(holes, labeled, range(1, n_components + 1))
                    keep_holes = np.zeros_like(tile_closed)
                    for idx, sz in enumerate(sizes, start=1):
                        if sz <= max_hole_area:
                            keep_holes |= labeled == idx
                    tile_closed = tile_closed | keep_holes
                # Crop the padded tile back to the original bbox region.
                inner = tile_closed[
                    (y0 - yp0) : (y0 - yp0) + (y1 - y0),
                    (x0 - xp0) : (x0 - xp0) + (x1 - x0),
                ]
                new_mask[y0:y1, x0:x1] |= inner

        Image.fromarray((new_mask.astype(np.uint8) * 255)).save(union_path)
        overlay = overlay_mask(rgb, new_mask)
        overlay = draw_contour(overlay, new_mask)
        Image.fromarray(overlay).save(overlay_dir / f"{stem}_det.jpg", quality=92)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--material", action="append", dest="materials")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--rebuild-all",
        action="store_true",
        help="Regenerate every mask/overlay (apply morphology) even if no dets were dropped.",
    )
    args = parser.parse_args()

    materials = args.materials or sorted(INPUT_DIRS.keys())

    summary = {}
    for material in materials:
        md_dir = OUTPUT_DIRS.get(material, ROOT / "out" / material) / "metadata"
        if not md_dir.exists():
            continue
        total_files = 0
        affected_files = 0
        total_dropped = 0
        rebuild_stems = []
        empty_after = 0
        for f in sorted(md_dir.glob("*.json")):
            total_files += 1
            data = json.loads(f.read_text(encoding="utf-8"))
            before = len(data["detections"])
            cap = cap_for(material, f.stem)
            min_s = min_score_for(material)
            fill_cap = max_bbox_fill_for(material, f.stem)
            border_slack = DROP_BORDER_TOUCHING.get(material)

            def _keep(d: dict) -> bool:
                if d.get("prompt", "") in PURE_CONTAINER_BLACKLIST:
                    return False
                if d.get("area_fraction", 0.0) > cap:
                    return False
                if d.get("score", 0.0) < min_s:
                    return False
                if d.get("bbox_fill_fraction", 0.0) > fill_cap:
                    return False
                if border_slack is not None:
                    bbox = d.get("bbox_xyxy") or [0, 0, 0, 0]
                    if bbox[0] <= border_slack or bbox[1] <= border_slack:
                        return False
                return True

            kept = [d for d in data["detections"] if _keep(d)]
            dropped = before - len(kept)
            if dropped > 0:
                affected_files += 1
                total_dropped += dropped
                rebuild_stems.append(f.stem)
                if not args.dry_run:
                    data["detections"] = [
                        dict(d, instance=i) for i, d in enumerate(kept, start=1)
                    ]
                    f.write_text(
                        json.dumps(data, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
            elif args.rebuild_all and kept:
                # No drops, but caller asked to regenerate anyway so morphology
                # is applied to old, fragmented masks.
                rebuild_stems.append(f.stem)
            if not kept:
                empty_after += 1

        if not args.dry_run and rebuild_stems:
            rebuild_outputs(material, rebuild_stems)

        summary[material] = {
            "total_files": total_files,
            "affected_files": affected_files,
            "dropped_detections": total_dropped,
            "empty_after_cleanup": empty_after,
        }
        print(
            f"{material}: files={total_files} affected={affected_files} "
            f"dropped_dets={total_dropped} empty_after={empty_after}"
        )

    out_path = ROOT / "out" / "_logs" / "_repair" / "_cleanup_summary.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
