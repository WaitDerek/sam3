"""Segment the washer-fluid filler pipe assembly in data/bgr images.

This is a single-target variant of detect_materials.py. It keeps the same
SAM 3.1 text-prompt flow, but is scoped to the PNG images under
data/bgr and saves both visual overlays and binary masks.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Keep imports bound to this fork's SAM3 package when the script is launched
# from another working directory.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image

from sam3.model.sam3_multiplex_tracking import Sam3MultiplexTrackingWithInteractivity
from sam3.model_builder import build_sam3_multiplex_video_predictor

_orig_init_state = Sam3MultiplexTrackingWithInteractivity.init_state


def _init_state_compat(self, *args, **kwargs):
    """Compatibility shim for the current multiplex predictor build."""
    kwargs.pop("offload_state_to_cpu", None)
    return _orig_init_state(self, *args, **kwargs)


Sam3MultiplexTrackingWithInteractivity.init_state = _init_state_compat

CKPT = ROOT / "sam3.1_multiplex.pt"
DEFAULT_INPUT_DIR = ROOT / "data" / "bgr"
DEFAULT_OUTPUT_DIR = ROOT / "out" / "washer_filler_769"

DEFAULT_LABEL = "洗涤液加注管总成"
DEFAULT_PROMPTS = [
    "white plastic washer fluid filler neck with blue cap",
    "windshield washer fluid filler neck",
    "washer fluid filler pipe assembly",
    "blue capped washer fluid filler pipe",
    "white plastic automotive filler tube",
    "white plastic windshield washer filler tube",
]

COLOR = np.array((0, 130, 255), dtype=np.uint8)
IOU_DEDUP_THRESH = 0.4
MIN_AREA_FRAC = 3e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment washer-fluid filler pipe assembly in data/bgr images."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pattern", default="*.png")
    parser.add_argument(
        "--include-prefix",
        action="append",
        dest="include_prefixes",
        help="Only process images whose filename starts with this prefix. Repeat for multiple prefixes.",
    )
    parser.add_argument(
        "--min-stem",
        help="Only process images whose filename stem is lexicographically >= this value.",
    )
    parser.add_argument(
        "--max-stem",
        help="Only process images whose filename stem is lexicographically <= this value.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        help="Process one explicit image instead of scanning --input-dir.",
    )
    parser.add_argument(
        "--image-list",
        type=Path,
        help="Process explicit images listed one per line. Lines may be stems, filenames, or paths.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process only the first N images after sorting. Use --limit 1 for prompt checks.",
    )
    parser.add_argument("--label", default=DEFAULT_LABEL)
    parser.add_argument(
        "--text-prompt",
        action="append",
        dest="prompts",
        help="Override prompt list. Repeat this option to test multiple variants.",
    )
    parser.add_argument("--min-score", type=float, default=0.35)
    parser.add_argument("--min-area-frac", type=float, default=MIN_AREA_FRAC)
    parser.add_argument(
        "--max-area-frac",
        type=float,
        help="Reject candidates whose mask area exceeds this fraction of the image.",
    )
    parser.add_argument(
        "--max-bbox-fill-frac",
        type=float,
        help="Reject candidates whose mask fills too much of its bounding box.",
    )
    parser.add_argument("--iou-dedup", type=float, default=IOU_DEDUP_THRESH)
    parser.add_argument(
        "--max-instances",
        type=int,
        help="Keep at most this many detections after score sorting and IoU deduplication.",
    )
    parser.add_argument(
        "--min-relative-score",
        type=float,
        help="Reject candidates whose score is below this fraction of the top kept score.",
    )
    parser.add_argument("--score-threshold-detection", type=float, default=0.2)
    parser.add_argument("--new-det-thresh", type=float, default=0.3)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip images whose overlay, mask, and metadata outputs already exist.",
    )
    return parser.parse_args()


def overlay_mask(
    rgb: np.ndarray, mask: np.ndarray, color: np.ndarray, alpha: float = 0.45
) -> np.ndarray:
    out = rgb.copy()
    m = mask.astype(bool)
    out[m] = (alpha * color + (1 - alpha) * rgb[m]).astype(np.uint8)
    return out


def draw_contour(
    rgb: np.ndarray, mask: np.ndarray, color: np.ndarray, thickness: int = 2
) -> np.ndarray:
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


def to_binary_mask(mask) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask.squeeze()
    return mask > 0.5


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 0.0


def mask_bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def bbox_fill_fraction(mask: np.ndarray, bbox: list[int] | None = None) -> float:
    if bbox is None:
        bbox = mask_bbox(mask)
    if bbox is None:
        return 0.0
    x0, y0, x1, y1 = bbox
    bbox_area = max(1, (x1 - x0) * (y1 - y0))
    return float(mask.sum()) / float(bbox_area)


def run_text_prompt(
    predictor, session_id: str, prompt: str
) -> list[tuple[np.ndarray, float]]:
    predictor.handle_request(request=dict(type="reset_session", session_id=session_id))
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

    results: list[tuple[np.ndarray, float]] = []
    for i in range(len(obj_ids)):
        prob = float(probs[i]) if i < len(probs) else 0.0
        results.append((to_binary_mask(masks_raw[i]), prob))
    return results


def collect_detections(
    predictor,
    image_path: Path,
    prompts: list[str],
    min_score: float,
    min_area_frac: float,
    max_area_frac: float | None,
    max_bbox_fill_frac: float | None,
    iou_dedup: float,
    max_instances: int | None = None,
    min_relative_score: float | None = None,
) -> tuple[np.ndarray, list[dict]]:
    image = Image.open(image_path).convert("RGB")
    rgb = np.array(image)
    height, width = rgb.shape[:2]
    image_area = height * width
    min_area = int(height * width * min_area_frac)
    max_area = int(image_area * max_area_frac) if max_area_frac is not None else None

    with tempfile.TemporaryDirectory() as tmp:
        frame_path = Path(tmp) / "0.jpg"
        image.save(frame_path, quality=95)
        resp = predictor.handle_request(
            request=dict(type="start_session", resource_path=tmp)
        )
        session_id = resp["session_id"]

        try:
            candidates: list[tuple[np.ndarray, float, str]] = []
            for prompt in prompts:
                for mask, prob in run_text_prompt(predictor, session_id, prompt):
                    if mask.shape != (height, width):
                        continue
                    area = int(mask.sum())
                    bbox = mask_bbox(mask)
                    fill_frac = bbox_fill_fraction(mask, bbox)
                    if area < min_area or prob < min_score:
                        continue
                    if max_area is not None and area > max_area:
                        continue
                    if (
                        max_bbox_fill_frac is not None
                        and fill_frac > max_bbox_fill_frac
                    ):
                        continue
                    candidates.append((mask, prob, prompt))
        finally:
            predictor.handle_request(
                request=dict(type="close_session", session_id=session_id)
            )

    candidates.sort(key=lambda item: item[1], reverse=True)
    if candidates and min_relative_score is not None:
        top_score = candidates[0][1]
        candidates = [
            item
            for item in candidates
            if item[1] >= top_score * min_relative_score
        ]

    kept: list[tuple[np.ndarray, float, str]] = []
    for mask, prob, prompt in candidates:
        if any(mask_iou(mask, kept_mask) >= iou_dedup for kept_mask, _, _ in kept):
            continue
        kept.append((mask, prob, prompt))
        if max_instances is not None and len(kept) >= max_instances:
            break

    metadata = []
    for index, (mask, prob, prompt) in enumerate(kept, start=1):
        bbox = mask_bbox(mask)
        area = int(mask.sum())
        metadata.append(
            {
                "instance": index,
                "score": prob,
                "prompt": prompt,
                "area": area,
                "area_fraction": float(area) / float(height * width),
                "bbox_xyxy": bbox,
                "bbox_fill_fraction": bbox_fill_fraction(mask, bbox),
            }
        )
    union = np.zeros((height, width), dtype=bool)
    for mask, _, _ in kept:
        union |= mask
    return union, metadata


def save_outputs(
    image_path: Path,
    output_dir: Path,
    label: str,
    prompts: list[str],
    union_mask: np.ndarray,
    metadata: list[dict],
) -> None:
    overlay_dir = output_dir / "overlays"
    mask_dir = output_dir / "masks"
    meta_dir = output_dir / "metadata"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    rgb = np.array(Image.open(image_path).convert("RGB"))
    overlay = overlay_mask(rgb, union_mask, COLOR, alpha=0.45)
    overlay = draw_contour(overlay, union_mask, COLOR, thickness=2)
    out_img = Image.fromarray(overlay)

    overlay_path = overlay_dir / f"{image_path.stem}_det.jpg"
    mask_path = mask_dir / f"{image_path.stem}_mask.png"
    meta_path = meta_dir / f"{image_path.stem}.json"

    out_img.save(overlay_path, quality=92)
    Image.fromarray((union_mask.astype(np.uint8) * 255)).save(mask_path)
    meta_path.write_text(
        json.dumps(
            {
                "image": str(image_path),
                "label": label,
                "prompts": prompts,
                "detections": metadata,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"  {label}: {len(metadata)} instance(s)")
    for det in metadata:
        print(
            "    "
            f"score={det['score']:.3f} "
            f"area={det['area']} "
            f"bbox={det['bbox_xyxy']} "
            f"prompt={det['prompt']!r}"
        )
    print(f"  overlay: {overlay_path}")
    print(f"  mask:    {mask_path}")
    print(f"  meta:    {meta_path}")


def outputs_exist(image_path: Path, output_dir: Path) -> bool:
    return (
        (output_dir / "overlays" / f"{image_path.stem}_det.jpg").exists()
        and (output_dir / "masks" / f"{image_path.stem}_mask.png").exists()
        and (output_dir / "metadata" / f"{image_path.stem}.json").exists()
    )


def build_predictor(args: argparse.Namespace):
    assert CKPT.exists(), f"checkpoint not found: {CKPT}"
    print(f"loading SAM 3.1 multiplex predictor from {CKPT.name} ...")
    predictor = build_sam3_multiplex_video_predictor(
        checkpoint_path=str(CKPT),
        use_fa3=False,
        use_rope_real=True,
        compile=False,
        warm_up=False,
    )
    predictor.model.score_threshold_detection = args.score_threshold_detection
    predictor.model.new_det_thresh = args.new_det_thresh
    print(
        "thresholds: "
        f"score_threshold_detection={predictor.model.score_threshold_detection}, "
        f"new_det_thresh={predictor.model.new_det_thresh}"
    )
    return predictor


def resolve_images(args: argparse.Namespace) -> list[Path]:
    if args.image:
        images = [args.image]
    elif args.image_list:
        image_lines = [
            line.strip()
            for line in args.image_list.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        images = []
        for line in image_lines:
            path = Path(line)
            if path.is_absolute():
                images.append(path)
            elif path.suffix:
                images.append(args.input_dir / path.name)
            else:
                images.append(args.input_dir / f"{line}.png")
    else:
        images = sorted(args.input_dir.glob(args.pattern))
        if args.include_prefixes:
            prefixes = tuple(args.include_prefixes)
            images = [image for image in images if image.name.startswith(prefixes)]
    if args.min_stem:
        images = [image for image in images if image.stem >= args.min_stem]
    if args.max_stem:
        images = [image for image in images if image.stem <= args.max_stem]
    if args.limit is not None:
        images = images[: args.limit]
    assert images, f"no images matched {args.input_dir / args.pattern}"
    return images


def main() -> None:
    args = parse_args()
    prompts = args.prompts or DEFAULT_PROMPTS
    images = resolve_images(args)

    print(f"label: {args.label}")
    print("prompts:")
    for prompt in prompts:
        print(f"  - {prompt}")
    print(f"images: {len(images)}")

    predictor = build_predictor(args)
    print("ready.\n")

    skipped = 0
    missing = 0
    for image_path in images:
        if not image_path.exists():
            missing += 1
            print(f"-> {image_path.name}")
            print("  skipped: input image no longer exists")
            continue

        if args.skip_existing and outputs_exist(image_path, args.output_dir):
            skipped += 1
            continue

        print(f"-> {image_path.name}")
        union_mask, metadata = collect_detections(
            predictor=predictor,
            image_path=image_path,
            prompts=prompts,
            min_score=args.min_score,
            min_area_frac=args.min_area_frac,
            max_area_frac=args.max_area_frac,
            max_bbox_fill_frac=args.max_bbox_fill_frac,
            iou_dedup=args.iou_dedup,
            max_instances=args.max_instances,
            min_relative_score=args.min_relative_score,
        )
        save_outputs(
            image_path=image_path,
            output_dir=args.output_dir,
            label=args.label,
            prompts=prompts,
            union_mask=union_mask,
            metadata=metadata,
        )

    if skipped:
        print(f"skipped existing outputs: {skipped}")
    if missing:
        print(f"skipped missing inputs: {missing}")


if __name__ == "__main__":
    main()
