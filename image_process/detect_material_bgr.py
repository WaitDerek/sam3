"""Segment material images from image_process/dataset and save masks to image_process/out.

This is a single-target variant of detect_materials.py. It keeps the same
SAM 3.1 text-prompt flow, but is scoped to the PNG images under
the configured bgr directory and saves both visual overlays and binary masks.
"""

from __future__ import annotations

import argparse
import json
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
from PIL import Image
from scipy import ndimage
import cv2

from sam3.model.sam3_multiplex_tracking import Sam3MultiplexTrackingWithInteractivity
from sam3.model_builder import build_sam3_multiplex_video_predictor

_orig_init_state = Sam3MultiplexTrackingWithInteractivity.init_state


def _init_state_compat(self, *args, **kwargs):
    """Compatibility shim for the current multiplex predictor build."""
    kwargs.pop("offload_state_to_cpu", None)
    return _orig_init_state(self, *args, **kwargs)


Sam3MultiplexTrackingWithInteractivity.init_state = _init_state_compat

CKPT = REPO_ROOT / "sam3.1_multiplex.pt"
DEFAULT_INPUT_DIR = DATA_DIR / "bgr"
DEFAULT_OUTPUT_DIR = OUT_DIR / "washer_filler_769"

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
        description="Segment material images and save overlay/mask/metadata outputs."
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
        "--image-stem",
        action="append",
        dest="image_stems",
        help="Process an explicit image stem from --input-dir. Repeat for multiple stems.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process a bounded prefix of sorted images for prompt checks.",
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
    parser.add_argument(
        "--min-bbox-fill-frac",
        type=float,
        help="Reject candidates whose mask fills too little of its bounding box.",
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
        help="Reject candidates whose score is below the configured relative score fraction.",
    )
    parser.add_argument(
        "--reject-border-touching",
        action="store_true",
        help="Reject candidates whose bbox touches any image border.",
    )
    parser.add_argument(
        "--border-slack",
        type=int,
        default=0,
        help="Pixel slack used with --reject-border-touching.",
    )
    parser.add_argument(
        "--max-aspect",
        type=float,
        help="Reject candidates whose bbox aspect ratio max(w,h)/min(w,h) "
        "exceeds this (drops elongated crate-rim/edge masks).",
    )
    parser.add_argument(
        "--centerline-trim",
        action="store_true",
        help="Trim elongated masks to a smoothed centerline band.",
    )
    parser.add_argument(
        "--centerline-half-height",
        type=int,
        default=14,
        help="Half-height in pixels kept around the smoothed centerline.",
    )
    parser.add_argument(
        "--centerline-window",
        type=int,
        default=51,
        help="Sliding median window in pixels for centerline smoothing.",
    )
    parser.add_argument(
        "--centerline-min-component-area",
        type=int,
        default=500,
        help="Drop smaller connected components during centerline trimming.",
    )
    parser.add_argument(
        "--centerline-min-aspect",
        type=float,
        default=3.0,
        help="Only trim connected components with bbox width/height above this ratio.",
    )
    parser.add_argument(
        "--mask-min-x",
        type=int,
        help="Drop mask pixels left of this column (keep x >= value). "
        "Used to clip a fused left-side bracket/holder out of an elongated mask.",
    )
    parser.add_argument(
        "--mask-max-x",
        type=int,
        help="Drop mask pixels right of this column (keep x <= value).",
    )
    parser.add_argument(
        "--min-component-bbox-fill-frac",
        type=float,
        help="Drop connected mask components whose bbox fill is below this value.",
    )
    parser.add_argument(
        "--split-components",
        action="store_true",
        help="Save each kept connected component as its own detection instance.",
    )
    parser.add_argument(
        "--kmeans-split-instances",
        type=int,
        help="Split one fused mask into this many instances by clustering mask "
        "pixel coordinates. Intended for touching objects that SAM keeps as a "
        "single connected component.",
    )
    parser.add_argument(
        "--kmeans-split-y-weight",
        type=float,
        default=1.0,
        help="Weight applied to y coordinates for --kmeans-split-instances.",
    )
    parser.add_argument(
        "--kmeans-split-gap-kernel",
        type=int,
        default=0,
        help="Remove a narrow boundary between clustered mask instances using "
        "this dilation kernel size. 0 keeps the full mask area.",
    )
    parser.add_argument(
        "--fill-holes",
        action="store_true",
        help="Fill enclosed holes in each kept mask (e.g. printed labels on a "
        "solid case left as holes by the segmentation).",
    )
    parser.add_argument(
        "--mask-close-kernel",
        type=int,
        default=0,
        help="Square structuring-element size for binary closing of each kept "
        "mask. 0 disables closing.",
    )
    parser.add_argument(
        "--mask-close-iterations",
        type=int,
        default=1,
        help="Iterations for --mask-close-kernel binary closing.",
    )
    parser.add_argument(
        "--fill-convex",
        action="store_true",
        help="Replace each mask component with its convex hull (cleans printed "
        "label/recess concavities on rigid rectangular cases). Guarded by "
        "--fill-convex-max-ratio so non-convex components (e.g. two touching "
        "cases) are left untouched.",
    )
    parser.add_argument(
        "--fill-convex-max-ratio",
        type=float,
        default=1.35,
        help="Only convex-hull a component when hull_area <= ratio * area.",
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


def connected_components(mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    height, width = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    components: list[tuple[np.ndarray, np.ndarray]] = []
    ys, xs = np.nonzero(mask)

    for start_y, start_x in zip(ys, xs):
        if seen[start_y, start_x]:
            continue

        queue = [(int(start_y), int(start_x))]
        seen[start_y, start_x] = True
        points_y: list[int] = []
        points_x: list[int] = []

        for y, x in queue:
            points_y.append(y)
            points_x.append(x)
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if (
                    0 <= ny < height
                    and 0 <= nx < width
                    and mask[ny, nx]
                    and not seen[ny, nx]
                ):
                    seen[ny, nx] = True
                    queue.append((ny, nx))

        components.append((np.array(points_y), np.array(points_x)))
    return components


def smooth_nan_median(values: np.ndarray, window: int) -> np.ndarray:
    if window < 1:
        return values

    half = window // 2
    smoothed = np.full_like(values, np.nan, dtype=float)
    for index in range(len(values)):
        lo = max(0, index - half)
        hi = min(len(values), index + half + 1)
        valid = values[lo:hi]
        valid = valid[~np.isnan(valid)]
        if len(valid):
            smoothed[index] = float(np.median(valid))
    return smoothed


def trim_mask_to_centerline(
    mask: np.ndarray,
    half_height: int,
    window: int,
    min_component_area: int,
    min_aspect: float,
) -> np.ndarray:
    trimmed = np.zeros(mask.shape, dtype=bool)

    for ys, xs in connected_components(mask):
        area = len(xs)
        if area < min_component_area:
            continue

        x0 = int(xs.min())
        x1 = int(xs.max()) + 1
        y0 = int(ys.min())
        y1 = int(ys.max()) + 1
        aspect = (x1 - x0) / max(1, y1 - y0)

        component = np.zeros(mask.shape, dtype=bool)
        component[ys, xs] = True
        if aspect < min_aspect:
            trimmed |= component
            continue

        centers = np.full(x1 - x0, np.nan, dtype=float)
        for offset, x in enumerate(range(x0, x1)):
            col_ys = np.flatnonzero(component[:, x])
            if len(col_ys):
                centers[offset] = float(np.median(col_ys))
        centers = smooth_nan_median(centers, window)

        kept = np.zeros(mask.shape, dtype=bool)
        for offset, x in enumerate(range(x0, x1)):
            center = centers[offset]
            if np.isnan(center):
                continue
            col_ys = np.flatnonzero(component[:, x])
            if len(col_ys) == 0:
                continue
            col_ys = col_ys[np.abs(col_ys - center) <= half_height]
            kept[col_ys, x] = True

        for comp_ys, comp_xs in connected_components(kept):
            if len(comp_xs) >= min_component_area:
                trimmed[comp_ys, comp_xs] = True

    return trimmed


def refine_mask(
    mask: np.ndarray,
    close_kernel: int,
    close_iterations: int,
    fill_holes: bool,
    fill_convex: bool = False,
    fill_convex_max_ratio: float = 1.35,
) -> np.ndarray:
    """Bridge small gaps (binary closing), fill enclosed holes, optionally
    replace each connected component with its convex hull.

    Makes a fragmented mask coherent again, e.g. a red warning-triangle case
    whose printed label / arrow / specular / recess regions were left as holes
    or edge concavities by the text-prompt segmentation.

    fill_convex solidifies each component to its convex hull, which cleans
    boundary concavities (an unmasked printed-label notch at the case end) that
    binary_fill_holes cannot reach. It is guarded by fill_convex_max_ratio:
    the hull is only applied when hull_area <= ratio * component_area, so a
    component that is far from convex (e.g. two touching cases forming a V) is
    left untouched instead of having the wedge between them filled with
    background.
    """
    out = mask
    if close_kernel and close_kernel > 0 and close_iterations > 0:
        # Pad so the closing's erosion step does not eat pixels off masks that
        # reach the image border.
        pad = close_kernel * close_iterations + 1
        padded = np.pad(out, pad, mode="constant")
        padded = ndimage.binary_closing(
            padded,
            structure=np.ones((close_kernel, close_kernel), dtype=bool),
            iterations=close_iterations,
        )
        out = padded[pad:-pad, pad:-pad]
    if fill_holes:
        out = ndimage.binary_fill_holes(out)
    if fill_convex:
        filled = np.zeros(out.shape, dtype=np.uint8)
        for ys, xs in connected_components(out):
            if len(xs) < 50:
                filled[ys, xs] = 1
                continue
            pts = np.stack([xs, ys], axis=1).astype(np.int32)
            hull = cv2.convexHull(pts)
            hull_area = cv2.contourArea(hull)
            if hull_area <= fill_convex_max_ratio * len(xs):
                cv2.fillConvexPoly(filled, hull, 1)
            else:
                filled[ys, xs] = 1
        out = filled > 0
    return out



def bbox_fill_fraction(mask: np.ndarray, bbox: list[int] | None = None) -> float:
    if bbox is None:
        bbox = mask_bbox(mask)
    if bbox is None:
        return 0.0
    x0, y0, x1, y1 = bbox
    bbox_area = max(1, (x1 - x0) * (y1 - y0))
    return float(mask.sum()) / float(bbox_area)


def split_or_filter_components(
    mask: np.ndarray,
    min_component_area: int,
    min_component_bbox_fill_frac: float | None,
    split_components: bool,
) -> list[np.ndarray]:
    components: list[np.ndarray] = []
    union = np.zeros(mask.shape, dtype=bool)

    for ys, xs in connected_components(mask):
        if len(xs) < min_component_area:
            continue
        component = np.zeros(mask.shape, dtype=bool)
        component[ys, xs] = True
        fill_frac = bbox_fill_fraction(component)
        if (
            min_component_bbox_fill_frac is not None
            and fill_frac < min_component_bbox_fill_frac
        ):
            continue
        if split_components:
            components.append(component)
        else:
            union |= component

    if split_components:
        return components
    if int(union.sum()) == 0:
        return []
    return [union]


def split_mask_by_kmeans(
    mask: np.ndarray,
    instances: int,
    y_weight: float,
    gap_kernel: int,
    min_component_area: int,
) -> list[np.ndarray]:
    ys, xs = np.nonzero(mask)
    if instances < 2 or len(xs) < max(instances, min_component_area):
        return [mask] if int(mask.sum()) else []

    coords = np.stack([xs.astype(np.float32), ys.astype(np.float32) * y_weight], axis=1)
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        50,
        0.2,
    )
    attempts = 8
    flags = cv2.KMEANS_PP_CENTERS
    compactness, labels, centers = cv2.kmeans(
        coords, instances, None, criteria, attempts, flags
    )
    del compactness

    labels = labels.reshape(-1)
    order = np.argsort(centers[:, 0])
    ordered_components: list[np.ndarray] = []
    for label_id in order:
        component = np.zeros(mask.shape, dtype=bool)
        keep = labels == label_id
        component[ys[keep], xs[keep]] = True
        if int(component.sum()) >= min_component_area:
            ordered_components.append(component)

    if len(ordered_components) != instances:
        return [mask] if int(mask.sum()) else []

    if gap_kernel and gap_kernel > 0:
        kernel = np.ones((gap_kernel, gap_kernel), dtype=np.uint8)
        dilated = [
            cv2.dilate(component.astype(np.uint8), kernel, iterations=1) > 0
            for component in ordered_components
        ]
        boundary = np.zeros(mask.shape, dtype=bool)
        for i, left in enumerate(dilated):
            for right in dilated[i + 1 :]:
                boundary |= left & right & mask
        ordered_components = [
            component & ~boundary for component in ordered_components
        ]
        if any(int(component.sum()) < min_component_area for component in ordered_components):
            return [mask] if int(mask.sum()) else []

    return ordered_components


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
    min_bbox_fill_frac: float | None,
    iou_dedup: float,
    max_instances: int | None = None,
    min_relative_score: float | None = None,
    reject_border_touching: bool = False,
    border_slack: int = 0,
    max_aspect: float | None = None,
    centerline_trim: bool = False,
    centerline_half_height: int = 14,
    centerline_window: int = 51,
    centerline_min_component_area: int = 500,
    centerline_min_aspect: float = 3.0,
    mask_min_x: int | None = None,
    mask_max_x: int | None = None,
    min_component_bbox_fill_frac: float | None = None,
    split_components: bool = False,
    fill_holes: bool = False,
    mask_close_kernel: int = 0,
    mask_close_iterations: int = 1,
    fill_convex: bool = False,
    fill_convex_max_ratio: float = 1.35,
    kmeans_split_instances: int | None = None,
    kmeans_split_y_weight: float = 1.0,
    kmeans_split_gap_kernel: int = 0,
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
                    if mask_min_x is not None or mask_max_x is not None:
                        mask = mask.copy()
                        if mask_min_x is not None:
                            mask[:, :mask_min_x] = False
                        if mask_max_x is not None:
                            mask[:, mask_max_x + 1 :] = False
                    area = int(mask.sum())
                    bbox = mask_bbox(mask)
                    fill_frac = bbox_fill_fraction(mask, bbox)
                    if area < min_area or prob < min_score:
                        continue
                    if max_area is not None and area > max_area:
                        continue
                    if max_aspect is not None and bbox is not None:
                        bw = bbox[2] - bbox[0]
                        bh = bbox[3] - bbox[1]
                        if max(bw, bh) / max(1, min(bw, bh)) > max_aspect:
                            continue
                    if reject_border_touching and bbox is not None:
                        x0, y0, x1, y1 = bbox
                        if (
                            x0 <= border_slack
                            or y0 <= border_slack
                            or x1 >= width - border_slack
                            or y1 >= height - border_slack
                        ):
                            continue
                    if (
                        max_bbox_fill_frac is not None
                        and fill_frac > max_bbox_fill_frac
                    ):
                        continue
                    if (
                        min_bbox_fill_frac is not None
                        and fill_frac < min_bbox_fill_frac
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

    processed: list[tuple[np.ndarray, float, str]] = []
    for mask, prob, prompt in kept:
        if centerline_trim:
            mask = trim_mask_to_centerline(
                mask=mask,
                half_height=centerline_half_height,
                window=centerline_window,
                min_component_area=centerline_min_component_area,
                min_aspect=centerline_min_aspect,
            )
        if fill_holes or mask_close_kernel or fill_convex:
            mask = refine_mask(
                mask,
                close_kernel=mask_close_kernel,
                close_iterations=mask_close_iterations,
                fill_holes=fill_holes,
                fill_convex=fill_convex,
                fill_convex_max_ratio=fill_convex_max_ratio,
            )
        if kmeans_split_instances is not None:
            masks = split_mask_by_kmeans(
                mask,
                instances=kmeans_split_instances,
                y_weight=kmeans_split_y_weight,
                gap_kernel=kmeans_split_gap_kernel,
                min_component_area=min_area,
            )
        elif min_component_bbox_fill_frac is not None or split_components:
            masks = split_or_filter_components(
                mask,
                min_component_area=min_area,
                min_component_bbox_fill_frac=min_component_bbox_fill_frac,
                split_components=split_components,
            )
        else:
            masks = [mask] if int(mask.sum()) else []
        for component_mask in masks:
            if int(component_mask.sum()) == 0:
                continue
            processed.append((component_mask, prob, prompt))

    metadata = []
    for index, (mask, prob, prompt) in enumerate(processed, start=1):
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
    for mask, _, _ in processed:
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
    elif args.image_stems:
        images = [args.input_dir / f"{stem}.png" for stem in args.image_stems]
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
            min_bbox_fill_frac=args.min_bbox_fill_frac,
            iou_dedup=args.iou_dedup,
            max_instances=args.max_instances,
            min_relative_score=args.min_relative_score,
            reject_border_touching=args.reject_border_touching,
            border_slack=args.border_slack,
            max_aspect=args.max_aspect,
            centerline_trim=args.centerline_trim,
            centerline_half_height=args.centerline_half_height,
            centerline_window=args.centerline_window,
            centerline_min_component_area=args.centerline_min_component_area,
            centerline_min_aspect=args.centerline_min_aspect,
            mask_min_x=args.mask_min_x,
            mask_max_x=args.mask_max_x,
            min_component_bbox_fill_frac=args.min_component_bbox_fill_frac,
            split_components=args.split_components,
            fill_holes=args.fill_holes,
            mask_close_kernel=args.mask_close_kernel,
            mask_close_iterations=args.mask_close_iterations,
            fill_convex=args.fill_convex,
            fill_convex_max_ratio=args.fill_convex_max_ratio,
            kmeans_split_instances=args.kmeans_split_instances,
            kmeans_split_y_weight=args.kmeans_split_y_weight,
            kmeans_split_gap_kernel=args.kmeans_split_gap_kernel,
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
