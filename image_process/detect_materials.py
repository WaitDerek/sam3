"""Run SAM 3.1 multiplex predictor on image_process/dataset/*.jpg.

Each image is treated as a single-frame "video" so there is no temporal
propagation across the 5 unrelated photos. For every image we run several
text-prompt variants per category, take the union of their detections
(deduplicated by IoU), and draw masks color-coded by category. Outputs go to
image_process/out/.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

from material_paths import DATA_DIR, OUT_DIR, REPO_ROOT

# Keep imports bound to this fork's SAM3 package when the script is launched
# from another working directory.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from sam3.model_builder import build_sam3_multiplex_video_predictor

# Upstream bug: sam3_base_predictor.start_session always passes
# offload_state_to_cpu=False to model.init_state(), but the multiplex tracking
# model's init_state doesn't accept that kwarg. Wrap init_state to swallow it.
from sam3.model.sam3_multiplex_tracking import Sam3MultiplexTrackingWithInteractivity

_orig_init_state = Sam3MultiplexTrackingWithInteractivity.init_state


def _init_state_compat(self, *args, **kwargs):
    kwargs.pop("offload_state_to_cpu", None)
    return _orig_init_state(self, *args, **kwargs)


Sam3MultiplexTrackingWithInteractivity.init_state = _init_state_compat

CKPT = REPO_ROOT / "sam3.1_multiplex.pt"
IN_DIR = DATA_DIR
OUT_DIR.mkdir(exist_ok=True)

# Each category: (label, color, [prompt variants], min_score).
# Per-category min_score is required because prompt scores are NOT comparable
# across classes — measured calibration on these images:
#   bracket prompt: 0.7-0.91 on real, 0.83-0.91 even on false positives (wiper, ECU)
#   wiper prompt:   0.34-0.59 on real (no windshield context → systematically low)
#   hose prompt:    0.64-0.80
#   washer prompt:  0.78-0.91
# So we floor each class at a value chosen from its observed range.
# Order also defines DRAW order: earlier = bottom, later = on top. The
# over-eager bracket class is drawn first so more-specific classes paint
# over its false positives on wiper / hose etc.
CATEGORIES: list[tuple[str, tuple[int, int, int], list[str], float]] = [
    (
        "mounting bracket",
        (255, 200, 0),  # yellow — drawn at the bottom
        [
            "black plastic bracket with multiple mounting holes",
            "ribbed automotive plastic bracket",
            "black plastic part with screw holes and ribs",
        ],
        0.55,
    ),
    (
        "air intake hose",
        (230, 25, 75),  # red
        [
            "air intake hose",
            "engine air intake duct",
            "corrugated air intake tube",
        ],
        0.4,
    ),
    (
        "washer fluid bottle",
        (0, 130, 255),  # blue
        [
            "windshield washer fluid reservoir",
            "white plastic bottle with blue cap",
            "white plastic windshield washer tank",
        ],
        0.5,
    ),
    (
        "wiper blade",
        (60, 180, 75),  # green — drawn last so it covers bracket false-fires on wiper
        [
            "windshield wiper blade with rubber strip",
            "curved windshield wiper blade",
            "windshield wiper assembly",
        ],
        0.3,
    ),
    (
        "foam block",
        (180, 60, 200),  # magenta
        [
            "black foam block",
            "black sponge block",
            "rectangular black foam",
            "automotive sound deadening foam pad",
        ],
        0.4,
    ),
]

IOU_DEDUP_THRESH = 0.4   # within a category, suppress duplicates from variants
MIN_AREA_FRAC = 5e-5     # drop tiny noise masks (<0.005% of image)

# Manual per-image overrides keyed by image stem ("1", "2", ...).
#   exclude_boxes: drop any mask whose centroid falls inside (x1,y1,x2,y2)
#   merge_categories: union all kept masks of these category labels into ONE
MANUAL_OVERRIDES: dict[str, dict] = {
    "3": {
        "exclude_boxes": [(0, 750, 380, 1280)],   # electronic device w/ cable, bottom-left
        "merge_categories": ["air intake hose"],  # merge duct + corrugated tube into one
    },
}


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, color: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    out = rgb.copy()
    m = mask.astype(bool)
    out[m] = (alpha * color + (1 - alpha) * rgb[m]).astype(np.uint8)
    return out


def draw_contour(rgb: np.ndarray, mask: np.ndarray, color: np.ndarray, thickness: int = 2) -> np.ndarray:
    m = mask.astype(np.uint8)
    pad = np.pad(m, thickness, mode="constant")
    border = (pad[thickness:-thickness, thickness:-thickness] == 1) & ~(
        (pad[: -2 * thickness, thickness:-thickness] == 1)
        & (pad[2 * thickness :, thickness:-thickness] == 1)
        & (pad[thickness:-thickness, : -2 * thickness] == 1)
        & (pad[thickness:-thickness, 2 * thickness :] == 1)
    )
    out = rgb.copy()
    out[border] = color
    return out


def to_binary_mask(t) -> np.ndarray:
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().numpy()
    t = np.asarray(t)
    if t.ndim == 3:
        t = t.squeeze()
    return t > 0.5


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 0.0


def mask_centroid(m: np.ndarray) -> tuple[int, int]:
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return (-1, -1)
    return (int(xs.mean()), int(ys.mean()))


def run_text_prompt(predictor, session_id: str, prompt: str) -> list[tuple[np.ndarray, float]]:
    predictor.handle_request(
        request=dict(type="reset_session", session_id=session_id)
    )
    resp = predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=0,
            text=prompt,
        )
    )
    outputs = resp.get("outputs", {}) or {}
    masks_raw = outputs.get("out_binary_masks", [])
    obj_ids = outputs.get("out_obj_ids", [])
    probs = outputs.get("out_probs", [])
    if hasattr(obj_ids, "tolist"):
        obj_ids = obj_ids.tolist()
    if hasattr(probs, "tolist"):
        probs = probs.tolist()
    out: list[tuple[np.ndarray, float]] = []
    for i in range(len(obj_ids)):
        m = to_binary_mask(masks_raw[i])
        p = float(probs[i]) if i < len(probs) else 0.0
        out.append((m, p))
    return out


def run_one_image(predictor, img_path: Path) -> None:
    img = np.array(Image.open(img_path).convert("RGB"))
    H, W = img.shape[:2]
    min_area = int(H * W * MIN_AREA_FRAC)
    overlay = img.copy()
    per_category_count: list[tuple[str, int, np.ndarray]] = []

    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(img_path, Path(tmp) / "0.jpg")
        resp = predictor.handle_request(
            request=dict(type="start_session", resource_path=tmp)
        )
        session_id = resp["session_id"]

        try:
            # Phase 1: collect detections per category (intra-category dedup,
            # category-specific min score).
            per_cat_kept: list[list[tuple[np.ndarray, float]]] = []
            for label, _, variants, min_score in CATEGORIES:
                kept: list[tuple[np.ndarray, float]] = []
                for variant in variants:
                    for m, p in run_text_prompt(predictor, session_id, variant):
                        if m.shape != (H, W) or m.sum() < min_area or p < min_score:
                            continue
                        if any(mask_iou(m, k[0]) >= IOU_DEDUP_THRESH for k in kept):
                            continue
                        kept.append((m, p))
                per_cat_kept.append(kept)

            # Phase 1.5: apply per-image manual overrides.
            override = MANUAL_OVERRIDES.get(img_path.stem, {})
            exclude_boxes = override.get("exclude_boxes", [])
            merge_categories = set(override.get("merge_categories", []))
            for ci, (label, _, _, _) in enumerate(CATEGORIES):
                kept = per_cat_kept[ci]
                # Drop masks whose centroid falls inside any exclude box.
                if exclude_boxes:
                    filtered: list[tuple[np.ndarray, float]] = []
                    for m, p in kept:
                        cx, cy = mask_centroid(m)
                        if any(x1 <= cx < x2 and y1 <= cy < y2 for x1, y1, x2, y2 in exclude_boxes):
                            print(f"  [{img_path.name}] dropping {label} mask at ({cx},{cy}) (excluded)")
                            continue
                        filtered.append((m, p))
                    kept = filtered
                # Merge all kept masks of this category into a single union.
                if label in merge_categories and len(kept) > 1:
                    union = np.zeros((H, W), dtype=bool)
                    best_p = 0.0
                    for m, p in kept:
                        union |= m
                        best_p = max(best_p, p)
                    print(f"  [{img_path.name}] merged {len(kept)} {label} masks into 1")
                    kept = [(union, best_p)]
                per_cat_kept[ci] = kept

            # Phase 2: paint each category in CATEGORIES order — later categories
            # overdraw earlier ones, so the bracket class (which over-fires on
            # wipers) goes first and gets covered by wiper green on the wiper.
            for ci, (label, color_tuple, _, _) in enumerate(CATEGORIES):
                color = np.array(color_tuple, dtype=np.uint8)
                kept = per_cat_kept[ci]
                for m, _ in kept:
                    overlay = overlay_mask(overlay, m, color, alpha=0.45)
                    overlay = draw_contour(overlay, m, color, thickness=2)
                per_category_count.append((label, len(kept), color))
                print(f"  [{img_path.name}] {label}: {len(kept)} instance(s)")
        finally:
            predictor.handle_request(
                request=dict(type="close_session", session_id=session_id)
            )

    # Bottom legend strip.
    legend_h = 28
    legend = np.full((legend_h, W, 3), 30, dtype=np.uint8)
    composed = np.concatenate([overlay, legend], axis=0)
    out_img = Image.fromarray(composed)

    draw = ImageDraw.Draw(out_img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    x = 8
    y = H + 6
    for label, n, color in per_category_count:
        draw.rectangle([x, y, x + 22, y + 16], fill=tuple(int(c) for c in color))
        x += 28
        text = f"{label} ({n})"
        draw.text((x, y), text, fill=(255, 255, 255), font=font)
        x += int(draw.textlength(text, font=font)) + 16

    out_path = OUT_DIR / f"{img_path.stem}_det.jpg"
    out_img.save(out_path, quality=92)
    print(f"  -> saved {out_path}")


def main() -> None:
    assert CKPT.exists(), f"checkpoint not found: {CKPT}"
    images = sorted(IN_DIR.glob("*.jpg"))
    assert images, f"no jpgs in {IN_DIR}"

    print(f"loading SAM 3.1 multiplex predictor from {CKPT.name} ...")
    predictor = build_sam3_multiplex_video_predictor(
        checkpoint_path=str(CKPT),
        use_fa3=False,
        use_rope_real=True,
        compile=False,
        warm_up=False,
    )

    # Lower detection thresholds so smaller / lower-confidence objects fire
    # (defaults are tuned for video tracking confidence, too strict for one-shot
    # image detection of unfamiliar parts).
    predictor.model.score_threshold_detection = 0.2
    predictor.model.new_det_thresh = 0.3
    print(
        f"thresholds: score_threshold_detection="
        f"{predictor.model.score_threshold_detection}, "
        f"new_det_thresh={predictor.model.new_det_thresh}"
    )
    print("ready.\n")

    for img_path in images:
        print(f"-> {img_path.name}")
        run_one_image(predictor, img_path)


if __name__ == "__main__":
    main()
