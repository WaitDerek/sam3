"""Convert Label Studio or COCO JSON annotations to masks/metadata/overlays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import cv2
import numpy as np


COLOR = np.array((0, 130, 255), dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Label Studio JSON exports to masks, metadata and overlays."
    )
    parser.add_argument("--json", type=Path, required=True, help="Label Studio or COCO JSON export.")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Dataset root containing bgr/.")
    parser.add_argument("--image-root", type=Path, default=None, help="Image folder. Default: <dataset-dir>/bgr.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output root. Default: <dataset-dir>.")
    parser.add_argument("--label", default=None, help="Override label written to metadata.")
    parser.add_argument("--image-data-key", default=None, help="Label Studio data key for image path.")
    parser.add_argument("--annotation-index", type=int, default=-1, help="Label Studio annotation index. -1 uses all.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def stem_from_mask_name(name: str) -> str:
    stem = Path(name).stem
    for suffix in ("_mask", "_det"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def normalize_image_ref(ref: str) -> str:
    if ref.startswith("/data/local-files/"):
        query = parse_qs(urlparse(ref).query)
        if query.get("d"):
            return unquote(query["d"][0])
    if ref.startswith("file://"):
        return unquote(urlparse(ref).path)
    return unquote(ref)


def resolve_image_path(task: dict, image_root: Path, image_data_key: str | None) -> Path:
    data = task.get("data", {})
    keys = [image_data_key] if image_data_key else list(data.keys())
    for key in keys:
        if not key or key not in data:
            continue
        value = data[key]
        if not isinstance(value, str):
            continue
        ref = normalize_image_ref(value)
        path = Path(ref)
        if path.is_absolute() and path.exists():
            return path
        candidate = path if path.exists() else image_root / path.name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"could not resolve image for task id={task.get('id')}")


def load_bgr(path: Path, fallback_size: tuple[int, int] | None = None) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is not None:
        return image
    if fallback_size:
        width, height = fallback_size
        return np.zeros((height, width, 3), dtype=np.uint8)
    raise FileNotFoundError(f"could not read image: {path}")


def percent_points_to_pixels(points: list[list[float]], width: int, height: int) -> np.ndarray:
    pts = []
    for x_pct, y_pct in points:
        x = int(round(float(x_pct) * width / 100.0))
        y = int(round(float(y_pct) * height / 100.0))
        pts.append([np.clip(x, 0, width - 1), np.clip(y, 0, height - 1)])
    return np.asarray(pts, dtype=np.int32)


def rect_to_mask(value: dict, width: int, height: int) -> np.ndarray:
    x0 = int(round(float(value["x"]) * width / 100.0))
    y0 = int(round(float(value["y"]) * height / 100.0))
    x1 = int(round((float(value["x"]) + float(value["width"])) * width / 100.0))
    y1 = int(round((float(value["y"]) + float(value["height"])) * height / 100.0))
    x0, x1 = sorted((np.clip(x0, 0, width), np.clip(x1, 0, width)))
    y0, y1 = sorted((np.clip(y0, 0, height), np.clip(y1, 0, height)))
    mask = np.zeros((height, width), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def polygon_to_mask(points: list[list[float]], width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    pts = percent_points_to_pixels(points, width, height)
    if len(pts) >= 3:
        cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def brush_rle_to_mask(value: dict, width: int, height: int) -> np.ndarray:
    try:
        from label_studio_converter.brush import decode_rle
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError(
            "BrushLabels RLE requires label-studio-converter. Install it in the "
            "active environment or export Label Studio annotations as polygons/COCO."
        ) from exc

    decoded = np.asarray(decode_rle(value["rle"]))
    if decoded.ndim == 1:
        if decoded.size == width * height * 4:
            decoded = decoded.reshape((height, width, 4))
            return decoded[:, :, 3] > 0
        if decoded.size == width * height:
            return decoded.reshape((height, width)) > 0
    if decoded.ndim == 2:
        return decoded > 0
    if decoded.ndim == 3:
        return decoded.any(axis=2)
    raise ValueError(f"unsupported brush RLE decoded shape: {decoded.shape}")


def labels_from_value(value: dict, result_type: str, default_label: str | None) -> list[str]:
    for key in ("brushlabels", "polygonlabels", "rectanglelabels", "labels"):
        labels = value.get(key)
        if labels:
            return [str(item) for item in labels]
    return [default_label or result_type]


def bbox_from_mask(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]


def overlay_mask(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = bgr.copy()
    rgb_color = COLOR[::-1]
    out[mask] = (0.45 * rgb_color + 0.55 * out[mask]).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, tuple(int(c) for c in rgb_color), 2)
    return out


def detection_record(mask: np.ndarray, label: str, source_type: str, instance: int) -> dict:
    h, w = mask.shape
    area = int(mask.sum())
    return {
        "instance": instance,
        "label": label,
        "source": "label_studio",
        "source_type": source_type,
        "bbox_xyxy": bbox_from_mask(mask),
        "area_pixels": area,
        "area_fraction": float(area / max(1, h * w)),
    }


def iter_label_studio_items(task: dict, bgr: np.ndarray, label_override: str | None, annotation_index: int) -> list[tuple[np.ndarray, dict]]:
    height, width = bgr.shape[:2]
    annotations = task.get("annotations") or task.get("completions") or []
    if annotation_index >= 0:
        annotations = annotations[annotation_index : annotation_index + 1]
    items = []
    for annotation in annotations:
        if annotation.get("was_cancelled"):
            continue
        for result in annotation.get("result", []):
            value = result.get("value", {})
            result_type = result.get("type", "")
            original_width = int(result.get("original_width") or width)
            original_height = int(result.get("original_height") or height)
            if (original_width, original_height) != (width, height):
                # Label Studio coordinates are percentages, so polygon/rect stay valid.
                original_width, original_height = width, height
            if result_type == "polygonlabels":
                mask = polygon_to_mask(value.get("points", []), original_width, original_height)
            elif result_type == "rectanglelabels":
                mask = rect_to_mask(value, original_width, original_height)
            elif result_type == "brushlabels":
                mask = brush_rle_to_mask(value, original_width, original_height)
            else:
                continue
            if mask.shape != (height, width):
                mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
            if not mask.any():
                continue
            label = label_override or labels_from_value(value, result_type, None)[0]
            items.append((mask, detection_record(mask, label, result_type, len(items) + 1)))
    return items


def decode_coco_segmentation(segmentation, height: int, width: int) -> np.ndarray:
    if isinstance(segmentation, list):
        mask = np.zeros((height, width), dtype=np.uint8)
        for polygon in segmentation:
            pts = np.asarray(polygon, dtype=np.float32).reshape(-1, 2).round().astype(np.int32)
            if len(pts) >= 3:
                cv2.fillPoly(mask, [pts], 1)
        return mask.astype(bool)
    if isinstance(segmentation, dict):
        try:
            from pycocotools import mask as mask_utils
        except Exception as exc:  # pragma: no cover - depends on optional package
            raise RuntimeError("COCO RLE requires pycocotools.") from exc
        return np.asarray(mask_utils.decode(segmentation)).astype(bool)
    return np.zeros((height, width), dtype=bool)


def convert_coco(data: dict, args: argparse.Namespace, image_root: Path, output_dir: Path) -> int:
    categories = {cat["id"]: cat.get("name", args.label or "object") for cat in data.get("categories", [])}
    images = {img["id"]: img for img in data.get("images", [])}
    anns_by_image: dict[int, list[dict]] = {}
    for ann in data.get("annotations", []):
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    written = 0
    for image_id, anns in anns_by_image.items():
        image_info = images[image_id]
        image_path = image_root / Path(image_info["file_name"]).name
        bgr = load_bgr(image_path, (int(image_info["width"]), int(image_info["height"])))
        h, w = bgr.shape[:2]
        masks = []
        detections = []
        for ann in anns:
            mask = decode_coco_segmentation(ann.get("segmentation"), h, w)
            if not mask.any():
                continue
            label = args.label or categories.get(ann.get("category_id"), "object")
            masks.append(mask)
            detections.append(detection_record(mask, label, "coco", len(detections) + 1))
        written += write_outputs(image_path.stem, bgr, masks, detections, output_dir, args.overwrite)
    return written


def write_outputs(
    stem: str,
    bgr: np.ndarray,
    masks: list[np.ndarray],
    detections: list[dict],
    output_dir: Path,
    overwrite: bool,
) -> int:
    masks_dir = output_dir / "masks"
    metadata_dir = output_dir / "metadata"
    overlays_dir = output_dir / "overlays"
    masks_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    mask_path = masks_dir / f"{stem}_mask.png"
    metadata_path = metadata_dir / f"{stem}.json"
    overlay_path = overlays_dir / f"{stem}_det.jpg"
    if not overwrite and (mask_path.exists() or metadata_path.exists() or overlay_path.exists()):
        print(f"skip existing: {stem}")
        return 0

    union = np.zeros(bgr.shape[:2], dtype=bool)
    for mask in masks:
        union |= mask
    cv2.imwrite(str(mask_path), (union.astype(np.uint8) * 255))
    cv2.imwrite(str(overlay_path), overlay_mask(bgr, union))
    prompt_suggestions = sorted({det["label"] for det in detections if det.get("label")})
    metadata = {
        "image": f"{stem}.png",
        "label": prompt_suggestions[0] if len(prompt_suggestions) == 1 else prompt_suggestions,
        "source": "label_studio_json_to_masks.py",
        "prompt_suggestions": prompt_suggestions,
        "detections": detections,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote: {stem} instances={len(detections)}")
    return 1


def convert_label_studio(tasks: list[dict], args: argparse.Namespace, image_root: Path, output_dir: Path) -> int:
    written = 0
    for task in tasks:
        image_path = resolve_image_path(task, image_root, args.image_data_key)
        bgr = load_bgr(image_path)
        items = iter_label_studio_items(task, bgr, args.label, args.annotation_index)
        masks = [mask for mask, _ in items]
        detections = [det for _, det in items]
        if not masks:
            print(f"skip no mask annotations: {image_path.name}")
            continue
        written += write_outputs(stem_from_mask_name(image_path.name), bgr, masks, detections, output_dir, args.overwrite)
    return written


def main() -> None:
    args = parse_args()
    data = json.loads(args.json.read_text(encoding="utf-8"))
    image_root = args.image_root or args.dataset_dir / "bgr"
    output_dir = args.output_dir or args.dataset_dir

    if isinstance(data, dict) and {"images", "annotations"}.issubset(data):
        written = convert_coco(data, args, image_root, output_dir)
    elif isinstance(data, list):
        written = convert_label_studio(data, args, image_root, output_dir)
    else:
        raise ValueError("unsupported JSON format: expected Label Studio task list or COCO dict")
    print(f"done: written={written}")


if __name__ == "__main__":
    main()
