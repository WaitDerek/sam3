"""Segment material images from dataset and save masks to out.

This script runs the SAM 3.1 text-prompt flow for one target label, scoped to the PNG images under
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
        "--exclude-stem",
        action="append",
        dest="exclude_stems",
        help="Skip an explicit image stem after resolving the input image set. Repeat for multiple stems.",
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
    parser.add_argument(
        "--fill-outer-contour",
        action="store_true",
        help="Solidify each mask component to the region inside its external "
        "contour. Fills a hollow crate (e.g. an open silver liner that "
        "fill_holes cannot reach) while preserving the true crate outline, "
        "unlike --fill-convex which cuts the rotated crate's corners.",
    )
    parser.add_argument(
        "--front-priority",
        action="store_true",
        help="For multiple crate-like instances, let the lower/front instance "
        "claim ambiguous pixels before rear instances. Useful for occluded crates.",
    )
    parser.add_argument(
        "--front-priority-convex",
        action="store_true",
        help="With --front-priority, convex-hull only the front-most instance "
        "before subtracting it from rear instances.",
    )
    parser.add_argument(
        "--front-priority-convex-max-ratio",
        type=float,
        default=2.5,
        help="Convex hull guard used by --front-priority-convex.",
    )
    parser.add_argument("--score-threshold-detection", type=float, default=0.2)
    parser.add_argument("--new-det-thresh", type=float, default=0.3)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip images whose overlay, mask, and metadata outputs already exist.",
    )
    parser.add_argument(
        "--save-instance-masks",
        action="store_true",
        help="Also save non-overlapping per-instance label masks and colored instance overlays.",
    )
    parser.add_argument(
        "--instance-boundary-gap",
        type=int,
        default=0,
        help="When saving instance masks, clear this many pixels around boundaries "
        "between touching instances. 0 keeps every foreground pixel assigned.",
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


INSTANCE_COLORS = [
    np.array((255, 64, 64), dtype=np.uint8),
    np.array((30, 144, 255), dtype=np.uint8),
    np.array((40, 190, 90), dtype=np.uint8),
    np.array((255, 180, 30), dtype=np.uint8),
    np.array((180, 90, 255), dtype=np.uint8),
]


def make_instance_label_map(
    masks: list[np.ndarray],
    boundary_gap: int = 0,
) -> np.ndarray:
    """Create a non-overlapping 8-bit label image from per-instance masks."""
    if not masks:
        return np.zeros((0, 0), dtype=np.uint8)

    height, width = masks[0].shape
    label_map = np.zeros((height, width), dtype=np.uint8)
    if len(masks) == 1:
        label_map[masks[0]] = 1
        return label_map

    stack = np.stack([mask.astype(bool) for mask in masks], axis=0)
    cover_count = stack.sum(axis=0)
    unique = cover_count == 1
    for index, mask in enumerate(stack, start=1):
        label_map[unique & mask] = index

    overlap = cover_count > 1
    if np.any(overlap):
        yy, xx = np.indices((height, width))
        centers = []
        for mask in stack:
            bbox = mask_bbox(mask)
            if bbox is None:
                centers.append((width / 2.0, height / 2.0))
                continue
            x0, y0, x1, y1 = bbox
            centers.append(((x0 + x1) / 2.0, (y0 + y1) / 2.0))
        distances = []
        for cx, cy in centers:
            distances.append((xx - cx) ** 2 + (yy - cy) ** 2)
        nearest = np.argmin(np.stack(distances, axis=0), axis=0).astype(np.uint8) + 1
        label_map[overlap] = nearest[overlap]

    if boundary_gap and boundary_gap > 0 and len(masks) > 1:
        boundary = np.zeros((height, width), dtype=bool)
        foreground = label_map > 0
        boundary[:, 1:] |= (label_map[:, 1:] != label_map[:, :-1]) & foreground[:, 1:] & foreground[:, :-1]
        boundary[1:, :] |= (label_map[1:, :] != label_map[:-1, :]) & foreground[1:, :] & foreground[:-1, :]
        kernel = np.ones((boundary_gap, boundary_gap), dtype=np.uint8)
        boundary = cv2.dilate(boundary.astype(np.uint8), kernel, iterations=1) > 0
        label_map[boundary] = 0

    return label_map


def overlay_instance_labels(
    rgb: np.ndarray,
    label_map: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    out = rgb.copy()
    for label_id in range(1, int(label_map.max()) + 1):
        mask = label_map == label_id
        if not np.any(mask):
            continue
        color = INSTANCE_COLORS[(label_id - 1) % len(INSTANCE_COLORS)]
        out[mask] = (alpha * color + (1 - alpha) * rgb[mask]).astype(np.uint8)
        out = draw_contour(out, mask, color, thickness=2)
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
    fill_outer_contour: bool = False,
) -> np.ndarray:
    """Bridge small gaps (binary closing), fill enclosed holes, optionally
    replace each connected component with its convex hull or solidify it to its
    outer contour.

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

    fill_outer_contour fills the region enclosed by each component's external
    contour. Unlike fill_convex it follows the mask's true boundary instead of
    a convex hull, so it makes a hollow crate solid (filling an open silver
    liner that binary_fill_holes cannot, because the liner opening leaks to the
    exterior at the rim) WITHOUT cutting the rotated crate's corners off with a
    convex diagonal.
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
    if fill_outer_contour:
        filled = np.zeros(out.shape, dtype=np.uint8)
        contours, _ = cv2.findContours(
            out.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(filled, contours, -1, color=1, thickness=cv2.FILLED)
        out = filled > 0
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


def apply_front_priority(
    processed: list[tuple[np.ndarray, float, str]],
    convex_front: bool,
    convex_front_max_ratio: float,
) -> list[tuple[np.ndarray, float, str]]:
    """Prioritize the lower/front crate when two crates occlude each other."""
    if len(processed) < 2:
        return processed

    bboxes = [mask_bbox(mask) for mask, _, _ in processed]
    if any(bbox is None for bbox in bboxes):
        return processed

    order = sorted(
        range(len(processed)),
        key=lambda index: (
            bboxes[index][3],
            (bboxes[index][0] + bboxes[index][2]) / 2.0,
        ),
        reverse=True,
    )

    allocated: list[tuple[np.ndarray, float, str] | None] = [None] * len(processed)
    occupied = np.zeros(processed[0][0].shape, dtype=bool)
    for rank, index in enumerate(order):
        mask, prob, prompt = processed[index]
        mask = mask.copy()
        if rank == 0 and convex_front:
            mask = refine_mask(
                mask,
                close_kernel=0,
                close_iterations=1,
                fill_holes=False,
                fill_convex=True,
                fill_convex_max_ratio=convex_front_max_ratio,
            )
        mask &= ~occupied
        if int(mask.sum()) == 0:
            continue
        allocated[index] = (mask, prob, prompt)
        occupied |= mask

    front_first = []
    for index in order:
        item = allocated[index]
        if item is not None:
            front_first.append(item)
    return front_first


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
    fill_outer_contour: bool = False,
    kmeans_split_instances: int | None = None,
    kmeans_split_y_weight: float = 1.0,
    kmeans_split_gap_kernel: int = 0,
    front_priority: bool = False,
    front_priority_convex: bool = False,
    front_priority_convex_max_ratio: float = 2.5,
) -> tuple[np.ndarray, list[dict], list[np.ndarray]]:
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
        if fill_holes or mask_close_kernel or fill_convex or fill_outer_contour:
            mask = refine_mask(
                mask,
                close_kernel=mask_close_kernel,
                close_iterations=mask_close_iterations,
                fill_holes=fill_holes,
                fill_convex=fill_convex,
                fill_convex_max_ratio=fill_convex_max_ratio,
                fill_outer_contour=fill_outer_contour,
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

    if front_priority:
        processed = apply_front_priority(
            processed,
            convex_front=front_priority_convex,
            convex_front_max_ratio=front_priority_convex_max_ratio,
        )

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
    instance_masks = [mask for mask, _, _ in processed]
    return union, metadata, instance_masks


def save_outputs(
    image_path: Path,
    output_dir: Path,
    label: str,
    prompts: list[str],
    union_mask: np.ndarray,
    metadata: list[dict],
    instance_masks: list[np.ndarray] | None = None,
    save_instance_masks: bool = False,
    instance_boundary_gap: int = 0,
) -> None:
    overlay_dir = output_dir / "overlays"
    mask_dir = output_dir / "masks"
    meta_dir = output_dir / "metadata"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    instance_mask_dir = output_dir / "instance_masks"
    instance_overlay_dir = output_dir / "instance_overlays"
    if save_instance_masks:
        instance_mask_dir.mkdir(parents=True, exist_ok=True)
        instance_overlay_dir.mkdir(parents=True, exist_ok=True)

    rgb = np.array(Image.open(image_path).convert("RGB"))
    overlay_path = overlay_dir / f"{image_path.stem}_det.jpg"
    mask_path = mask_dir / f"{image_path.stem}_mask.png"
    meta_path = meta_dir / f"{image_path.stem}.json"

    Image.fromarray((union_mask.astype(np.uint8) * 255)).save(mask_path)
    label_map = None
    if save_instance_masks:
        if instance_masks:
            label_map = make_instance_label_map(
                instance_masks,
                boundary_gap=instance_boundary_gap,
            )
        else:
            label_map = np.zeros(union_mask.shape, dtype=np.uint8)
        instance_mask_path = instance_mask_dir / f"{image_path.stem}_instances.png"
        instance_overlay_path = instance_overlay_dir / f"{image_path.stem}_instances.jpg"
        Image.fromarray(label_map).save(instance_mask_path)
        instance_overlay = overlay_instance_labels(rgb, label_map, alpha=0.45)
        Image.fromarray(instance_overlay).save(instance_overlay_path, quality=92)
    if label_map is not None and int(label_map.max()) > 1:
        overlay = overlay_instance_labels(rgb, label_map, alpha=0.45)
    else:
        overlay = overlay_mask(rgb, union_mask, COLOR, alpha=0.45)
        overlay = draw_contour(overlay, union_mask, COLOR, thickness=2)
    Image.fromarray(overlay).save(overlay_path, quality=92)
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
    if args.exclude_stems:
        excluded = set(args.exclude_stems)
        images = [image for image in images if image.stem not in excluded]
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
        union_mask, metadata, instance_masks = collect_detections(
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
            fill_outer_contour=args.fill_outer_contour,
            kmeans_split_instances=args.kmeans_split_instances,
            kmeans_split_y_weight=args.kmeans_split_y_weight,
            kmeans_split_gap_kernel=args.kmeans_split_gap_kernel,
            front_priority=args.front_priority,
            front_priority_convex=args.front_priority_convex,
            front_priority_convex_max_ratio=args.front_priority_convex_max_ratio,
        )
        save_outputs(
            image_path=image_path,
            output_dir=args.output_dir,
            label=args.label,
            prompts=prompts,
            union_mask=union_mask,
            metadata=metadata,
            instance_masks=instance_masks,
            save_instance_masks=args.save_instance_masks,
            instance_boundary_gap=args.instance_boundary_gap,
        )

    if skipped:
        print(f"skipped existing outputs: {skipped}")
    if missing:
        print(f"skipped missing inputs: {missing}")


if __name__ == "__main__":
    main()
