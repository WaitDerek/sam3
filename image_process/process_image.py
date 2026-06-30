import argparse
import contextlib
import json
import struct
from pathlib import Path

import cv2
import numpy as np
import torch
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor import SAM2ImagePredictor


DEFAULT_DATASET_DIR = Path(__file__).resolve().parent / "sam3demo"
DEFAULT_BOX = (760, 350, 990, 510)
DEFAULT_INTRINSICS = {
    "width": 1280,
    "height": 720,
    "fx": 662.636,
    "fy": 662.636,
    "cx": 635.522,
    "cy": 348.738,
}


def parse_box(value: str) -> np.ndarray:
    parts = [float(x.strip()) for x in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("box must be x1,y1,x2,y2")
    x1, y1, x2, y2 = parts
    if x2 <= x1 or y2 <= y1:
        raise argparse.ArgumentTypeError("box must satisfy x2>x1 and y2>y1")
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def parse_point(value: str) -> tuple[float, float]:
    parts = [float(x.strip()) for x in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("point must be x,y")
    return parts[0], parts[1]


def parse_legacy_intrinsics(text: str) -> dict:
    intrinsics = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key in {"width", "height"}:
            intrinsics[key] = int(float(value))
        elif key in {"fx", "fy", "cx", "cy"}:
            intrinsics[key] = float(value)
    return intrinsics


def load_intrinsics(dataset_dir: Path, intrinsics_json: Path | None) -> dict:
    candidates = []
    if intrinsics_json is not None:
        candidates.append(intrinsics_json)
    candidates.extend(
        [
            dataset_dir / "meta" / "intrinsic.json",
            dataset_dir / "intrinsic.json",
            Path(__file__).resolve().parent / "intrinsic.json",
        ]
    )

    intrinsics = dict(DEFAULT_INTRINSICS)
    for path in candidates:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = parse_legacy_intrinsics(text)
        for key in ["width", "height", "fx", "fy", "cx", "cy"]:
            if key in data:
                intrinsics[key] = data[key]
        return intrinsics
    return intrinsics


def load_bgr(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"could not read bgr image: {path}")
    return bgr


def load_depth(path: Path) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"could not read depth image: {path}")
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    return depth


def apply_depth_filter(
    depth: np.ndarray,
    mode: str,
    median_ksize: int,
    bilateral_d: int,
    bilateral_sigma_color: float,
    bilateral_sigma_space: float,
) -> np.ndarray:
    if mode == "none":
        return depth

    valid = depth > 0
    if mode == "median":
        ksize = max(3, int(median_ksize))
        if ksize % 2 == 0:
            ksize += 1
        filtered = cv2.medianBlur(depth, ksize)
    elif mode == "bilateral":
        filtered = cv2.bilateralFilter(
            depth.astype(np.float32),
            int(bilateral_d),
            float(bilateral_sigma_color),
            float(bilateral_sigma_space),
        )
        if np.issubdtype(depth.dtype, np.integer):
            filtered = np.rint(filtered).clip(0, np.iinfo(depth.dtype).max).astype(depth.dtype)
    else:
        raise ValueError(f"unsupported depth filter: {mode}")

    filtered[~valid] = 0
    return filtered


def choose_frame(bgr_dir: Path, depth_dir: Path, frame: str | None) -> str:
    if frame:
        return frame
    for bgr_path in sorted(bgr_dir.glob("*.png")):
        if (depth_dir / bgr_path.name).exists():
            return bgr_path.stem
    raise FileNotFoundError(f"no aligned bgr/depth frame found in {bgr_dir} and {depth_dir}")


def choose_auto_mask(
    annotations: list[dict],
    image_shape: tuple[int, int],
    min_area: int,
    max_area_ratio: float,
) -> tuple[np.ndarray, float]:
    height, width = image_shape
    max_area = int(height * width * max_area_ratio)
    candidates = [
        ann
        for ann in annotations
        if min_area <= int(ann["area"]) <= max_area
    ]
    if not candidates:
        candidates = annotations
    if not candidates:
        raise RuntimeError("SAM2 automatic mode produced no masks.")

    def score(ann: dict) -> float:
        return float(ann.get("predicted_iou", 0.0)) * float(ann.get("stability_score", 0.0))

    best = max(candidates, key=score)
    return best["segmentation"].astype(bool), score(best)


def sam2_auto_mask(
    image_rgb: np.ndarray,
    model_id: str,
    min_area: int,
    max_area_ratio: float,
) -> tuple[np.ndarray, float]:
    generator = SAM2AutomaticMaskGenerator.from_pretrained(
        model_id,
        points_per_side=32,
        pred_iou_thresh=0.8,
        stability_score_thresh=0.9,
        min_mask_region_area=100,
    )
    annotations = generator.generate(image_rgb)
    return choose_auto_mask(annotations, image_rgb.shape[:2], min_area, max_area_ratio)


def auto_detect_foreground_prompt(
    image_rgb: np.ndarray,
    min_area: int,
    max_area_ratio: float,
    padding_ratio: float = 0.18,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Find the most salient dark foreground object and turn it into a SAM2 prompt."""
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    _, saturation, value = cv2.split(hsv)
    height, width = value.shape

    # The datasets are usually objects on a green/gray table. Dark, saturated regions
    # isolate the object better than SAM's fully automatic mask ranking.
    dark_cutoff = min(100, max(55, int(np.percentile(value, 28))))
    candidate = ((value <= dark_cutoff) & (saturation >= 25)).astype(np.uint8) * 255

    border = max(8, min(width, height) // 80)
    candidate[:border, :] = 0
    candidate[-max(border, height // 22):, :] = 0
    candidate[:, :border] = 0
    candidate[:, -border:] = 0

    kernel = np.ones((5, 5), np.uint8)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel, iterations=1)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel, iterations=2)

    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate, 8)
    max_area = int(height * width * max_area_ratio)
    best = None
    for label in range(1, component_count):
        x, y, w, h, area = stats[label]
        if area < min_area or area > max_area:
            continue
        aspect = max(w / max(h, 1), h / max(w, 1))
        if aspect > 8.0:
            continue

        cx, cy = centroids[label]
        center_score = 1.0 - 0.5 * (
            ((cx - width / 2) / (width / 2)) ** 2
            + ((cy - height / 2) / (height / 2)) ** 2
        )
        darkness_score = (255.0 - float(value[labels == label].mean())) / 255.0
        score = float(area) * max(0.2, center_score) * (0.6 + darkness_score)
        if best is None or score > best[0]:
            best = (score, x, y, w, h, cx, cy)

    if best is None:
        raise RuntimeError(
            "could not auto-detect an obvious foreground object; pass --box or --point for this frame"
        )

    _, x, y, w, h, cx, cy = best
    pad = int(max(w, h) * padding_ratio)
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(width - 1, x + w + pad)
    y2 = min(height - 1, y + h + pad)
    box = np.array([x1, y1, x2, y2], dtype=np.float32)
    return box, [(float(cx), float(cy))]


def sam2_predict_mask(
    predictor: SAM2ImagePredictor,
    image_rgb: np.ndarray,
    box: np.ndarray | None,
    positive_points: list[tuple[float, float]],
    negative_points: list[tuple[float, float]],
) -> tuple[np.ndarray, float]:
    predictor.set_image(image_rgb)

    point_coords = None
    point_labels = None
    if positive_points or negative_points:
        points = list(positive_points) + list(negative_points)
        labels = [1] * len(positive_points) + [0] * len(negative_points)
        point_coords = np.array(points, dtype=np.float32)
        point_labels = np.array(labels, dtype=np.int32)

    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=box,
        multimask_output=True,
    )
    best_idx = int(np.argmax(scores))
    return masks[best_idx].astype(bool), float(scores[best_idx])


def make_colored_mask_overlay(
    bgr: np.ndarray,
    mask: np.ndarray,
    color_bgr: tuple[int, int, int] = (0, 180, 0),
    alpha: float = 0.55,
) -> np.ndarray:
    overlay = bgr.copy()
    color = np.array(color_bgr, dtype=np.uint8)
    overlay[mask] = ((1.0 - alpha) * overlay[mask] + alpha * color).astype(np.uint8)

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color_bgr, 2)
    return overlay


def bgr_depth_mask_to_points(
    bgr: np.ndarray,
    depth_raw: np.ndarray,
    mask: np.ndarray,
    intrinsics: dict,
    depth_scale: float,
    min_depth: float,
    max_depth: float,
    legacy_transform: bool,
) -> tuple[np.ndarray, np.ndarray]:
    depth_m = depth_raw.astype(np.float32) / depth_scale
    valid = mask & (depth_m > 0)
    if min_depth > 0:
        valid &= depth_m >= min_depth
    if max_depth > 0:
        valid &= depth_m <= max_depth

    v, u = np.nonzero(valid)
    if len(u) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    z = depth_m[v, u]
    x = (u.astype(np.float32) - float(intrinsics["cx"])) * z / float(intrinsics["fx"])
    y = (v.astype(np.float32) - float(intrinsics["cy"])) * z / float(intrinsics["fy"])

    if legacy_transform:
        y = -y
        z = -z

    points = np.stack([x, y, z], axis=1).astype(np.float32)
    colors_rgb = bgr[v, u][:, ::-1].astype(np.uint8)
    return points, colors_rgb


def write_pcd(path: Path, points: np.ndarray, colors_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="ascii") as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        f.write("FIELDS x y z\n")
        f.write("SIZE 4 4 4\n")
        f.write("TYPE F F F\n")
        f.write("COUNT 1 1 1\n")
        f.write(f"WIDTH {len(points)}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {len(points)}\n")
        f.write("DATA ascii\n")
        for x, y, z in points:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def pack_rgb_as_float(r: int, g: int, b: int) -> float:
    rgb_uint = (int(r) << 16) | (int(g) << 8) | int(b)
    return struct.unpack("f", struct.pack("I", rgb_uint))[0]


def write_colored_pcd(path: Path, points: np.ndarray, colors_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="ascii") as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        f.write("FIELDS x y z rgb\n")
        f.write("SIZE 4 4 4 4\n")
        f.write("TYPE F F F F\n")
        f.write("COUNT 1 1 1 1\n")
        f.write(f"WIDTH {len(points)}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {len(points)}\n")
        f.write("DATA ascii\n")
        for (x, y, z), (r, g, b) in zip(points, colors_rgb):
            rgb = pack_rgb_as_float(int(r), int(g), int(b))
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {rgb:.8e}\n")


def bgr_depth_to_highlight_points(
    bgr: np.ndarray,
    depth_raw: np.ndarray,
    mask: np.ndarray,
    intrinsics: dict,
    depth_scale: float,
    min_depth: float,
    max_depth: float,
    legacy_transform: bool,
) -> tuple[np.ndarray, np.ndarray]:
    all_mask = np.ones(depth_raw.shape[:2], dtype=bool)
    points, colors_rgb = bgr_depth_mask_to_points(
        bgr=bgr,
        depth_raw=depth_raw,
        mask=all_mask,
        intrinsics=intrinsics,
        depth_scale=depth_scale,
        min_depth=min_depth,
        max_depth=max_depth,
        legacy_transform=legacy_transform,
    )

    depth_m = depth_raw.astype(np.float32) / depth_scale
    valid = depth_m > 0
    if min_depth > 0:
        valid &= depth_m >= min_depth
    if max_depth > 0:
        valid &= depth_m <= max_depth

    v, u = np.nonzero(valid)
    selected = mask[v, u]
    colors_rgb[selected] = np.array([255, 0, 0], dtype=np.uint8)
    return points, colors_rgb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SAM2 mask and target PCD from bgr/depth folders."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--bgr-dir", type=Path, default=None)
    parser.add_argument("--depth-dir", type=Path, default=None)
    parser.add_argument("--mask-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--frame", default=None)
    parser.add_argument("--box", type=parse_box, default=None)
    parser.add_argument("--point", type=parse_point, action="append", default=[])
    parser.add_argument("--negative-point", type=parse_point, action="append", default=[])
    parser.add_argument("--model-id", default="facebook/sam2-hiera-large")
    parser.add_argument("--auto-min-area", type=int, default=1000)
    parser.add_argument("--auto-max-area-ratio", type=float, default=0.25)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--intrinsics-json", type=Path, default=None)
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--min-depth", type=float, default=0.0)
    parser.add_argument("--max-depth", type=float, default=0.0)
    parser.add_argument("--depth-filter", choices=["none", "median", "bilateral"], default="none")
    parser.add_argument("--median-ksize", type=int, default=5)
    parser.add_argument("--bilateral-d", type=int, default=7)
    parser.add_argument("--bilateral-sigma-color", type=float, default=35.0)
    parser.add_argument("--bilateral-sigma-space", type=float, default=35.0)
    parser.add_argument(
        "--camera-frame",
        action="store_true",
        help="Keep x-right/y-down/z-forward instead of the historical y/z flipped PCD convention.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir
    bgr_dir = args.bgr_dir or dataset_dir / "bgr"
    depth_dir = args.depth_dir or dataset_dir / "depth"
    mask_dir = args.mask_dir or dataset_dir / "mask"
    pcd_dir = args.output_dir or dataset_dir / "pcd"
    intrinsics = load_intrinsics(dataset_dir, args.intrinsics_json)

    for name, directory in [("bgr", bgr_dir), ("depth", depth_dir)]:
        if not directory.is_dir():
            raise FileNotFoundError(f"missing {name} dir: {directory}")

    frame = choose_frame(bgr_dir, depth_dir, args.frame)
    bgr_path = bgr_dir / f"{frame}.png"
    depth_path = depth_dir / f"{frame}.png"
    mask_path = mask_dir / f"{frame}.png"
    pcd_stem = f"{frame}_filter" if args.depth_filter != "none" else frame
    pcd_path = pcd_dir / f"{pcd_stem}.pcd"

    if pcd_path.exists() and not args.overwrite:
        print(f"skip existing pcd: {pcd_path}")
        return

    bgr = load_bgr(bgr_path)
    image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    depth = load_depth(depth_path)
    depth = apply_depth_filter(
        depth=depth,
        mode=args.depth_filter,
        median_ksize=args.median_ksize,
        bilateral_d=args.bilateral_d,
        bilateral_sigma_color=args.bilateral_sigma_color,
        bilateral_sigma_space=args.bilateral_sigma_space,
    )

    autocast_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if torch.cuda.is_available()
        else contextlib.nullcontext()
    )
    with torch.inference_mode(), autocast_ctx:
        predictor = SAM2ImagePredictor.from_pretrained(args.model_id)
        if args.box is None and not args.point and not args.negative_point:
            box, points = auto_detect_foreground_prompt(
                image_rgb=image_rgb,
                min_area=args.auto_min_area,
                max_area_ratio=args.auto_max_area_ratio,
            )
            box_text = ",".join(str(int(v)) for v in box.tolist())
            print(f"Auto prompt: box={box_text}, point={points[0][0]:.1f},{points[0][1]:.1f}")
            mask, score = sam2_predict_mask(
                predictor=predictor,
                image_rgb=image_rgb,
                box=box,
                positive_points=points,
                negative_points=[],
            )
        else:
            mask, score = sam2_predict_mask(
                predictor=predictor,
                image_rgb=image_rgb,
                box=args.box,
                positive_points=args.point,
                negative_points=args.negative_point,
            )

    mask_dir.mkdir(parents=True, exist_ok=True)
    colored_mask = make_colored_mask_overlay(bgr, mask)
    cv2.imwrite(str(mask_path), colored_mask)

    points, colors = bgr_depth_mask_to_points(
        bgr=bgr,
        depth_raw=depth,
        mask=mask,
        intrinsics=intrinsics,
        depth_scale=args.depth_scale,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        legacy_transform=not args.camera_frame,
    )
    if len(points) == 0:
        raise RuntimeError("SAM2 mask selected no valid depth pixels; check prompt/depth alignment.")

    highlight_points, highlight_colors = bgr_depth_to_highlight_points(
        bgr=bgr,
        depth_raw=depth,
        mask=mask,
        intrinsics=intrinsics,
        depth_scale=args.depth_scale,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        legacy_transform=not args.camera_frame,
    )
    write_colored_pcd(pcd_path, highlight_points, highlight_colors)

    print(f"Image: {bgr_path}")
    print(f"Depth: {depth_path}")
    print(f"Colored mask: {mask_path}")
    print(f"PCD: {pcd_path}")
    print(f"Depth filter: {args.depth_filter}")
    print(f"SAM2 score: {score:.4f}")
    print(f"Mask pixels: {int(mask.sum())}")
    print(f"Mask depth points: {len(points)}")
    print(f"PCD points: {len(highlight_points)}")


if __name__ == "__main__":
    main()
