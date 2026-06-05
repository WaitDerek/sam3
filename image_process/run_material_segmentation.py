"""Run material segmentation from the JSON parameter file."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from material_paths import (
    CONFIG_DIR,
    OUT_DIR,
    REPO_ROOT,
    TOOL_DIR,
    relative_to_repo,
    resolve_config_path,
    resolve_repo_path,
)

DEFAULT_CONFIG = CONFIG_DIR / "object_segmentation_params.json"
LOG_DIR = OUT_DIR / "_logs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run image_process/detect_material_bgr.py for every object in the config."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--label",
        action="append",
        dest="labels",
        help="Only run this label. Repeat for multiple labels.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove configured output directories before running.",
    )
    parser.add_argument(
        "--write-params-only",
        action="store_true",
        help="Only write image_process/out/<label>.json parameter snapshots.",
    )
    return parser.parse_args()


def merge_thresholds(obj: dict, profile: dict | None = None) -> dict:
    thresholds = dict(obj.get("thresholds", {}))
    if profile:
        thresholds.update(profile.get("threshold_overrides", {}))
    return thresholds


def load_parameter_files(obj: dict) -> list[dict]:
    profiles = []
    for file_ref in obj.get("parameter_files", []):
        path = resolve_config_path(file_ref)
        payload = json.loads(path.read_text(encoding="utf-8"))
        try:
            parameter_file = str(path.relative_to(TOOL_DIR))
        except ValueError:
            parameter_file = relative_to_repo(path)
        if "profiles" in payload:
            defaults = {key: value for key, value in payload.items() if key != "profiles"}
            for profile_payload in payload["profiles"]:
                profile = dict(defaults)
                profile.update(profile_payload)
                profile.setdefault("name", path.stem)
                profile["parameter_file"] = parameter_file
                profile["parameter_group"] = path.stem
                profiles.append(profile)
        else:
            payload.setdefault("name", path.stem)
            payload["parameter_file"] = parameter_file
            profiles.append(payload)
    if profiles:
        return profiles
    return obj.get("profiles") or ([None] if obj.get("prompts") else [])


def object_matches(obj: dict, requested: set[str]) -> bool:
    names = {
        obj["label"],
        obj.get("material", ""),
        obj.get("material_name", ""),
        obj.get("source_name", ""),
    }
    names.update(obj.get("legacy_labels", []))
    return bool(requested & {name for name in names if name})


def build_command(obj: dict, profile: dict | None = None) -> list[str]:
    thresholds = merge_thresholds(obj, profile)
    prompts = profile.get("prompts") if profile else obj.get("prompts", [])
    if not prompts:
        raise ValueError(f"{obj['label']} has no prompts")
    input_dir = profile.get("input_dir", obj["input_dir"]) if profile else obj["input_dir"]
    output_dir = (
        profile.get("output_dir", obj["output_dir"]) if profile else obj["output_dir"]
    )

    cmd = [
        sys.executable,
        str(TOOL_DIR / "detect_material_bgr.py"),
        "--input-dir",
        str(resolve_repo_path(input_dir)),
        "--output-dir",
        str(resolve_repo_path(output_dir)),
        "--label",
        obj["label"],
        "--min-score",
        str(thresholds["min_score"]),
        "--min-area-frac",
        str(thresholds["min_area_frac"]),
        "--iou-dedup",
        str(thresholds.get("iou_dedup", 0.4)),
        "--score-threshold-detection",
        str(thresholds.get("score_threshold_detection", 0.2)),
        "--new-det-thresh",
        str(thresholds.get("new_det_thresh", 0.3)),
    ]

    if thresholds.get("max_area_frac") is not None:
        cmd.extend(["--max-area-frac", str(thresholds["max_area_frac"])])
    if thresholds.get("max_bbox_fill_frac") is not None:
        cmd.extend(["--max-bbox-fill-frac", str(thresholds["max_bbox_fill_frac"])])
    if thresholds.get("min_bbox_fill_frac") is not None:
        cmd.extend(["--min-bbox-fill-frac", str(thresholds["min_bbox_fill_frac"])])
    if thresholds.get("max_instances") is not None:
        cmd.extend(["--max-instances", str(thresholds["max_instances"])])
    if thresholds.get("min_relative_score") is not None:
        cmd.extend(["--min-relative-score", str(thresholds["min_relative_score"])])
    if thresholds.get("reject_border_touching"):
        cmd.append("--reject-border-touching")
    if thresholds.get("border_slack") is not None:
        cmd.extend(["--border-slack", str(thresholds["border_slack"])])
    if thresholds.get("max_aspect") is not None:
        cmd.extend(["--max-aspect", str(thresholds["max_aspect"])])
    if thresholds.get("centerline_trim"):
        cmd.append("--centerline-trim")
    if thresholds.get("centerline_half_height") is not None:
        cmd.extend(
            ["--centerline-half-height", str(thresholds["centerline_half_height"])]
        )
    if thresholds.get("centerline_window") is not None:
        cmd.extend(["--centerline-window", str(thresholds["centerline_window"])])
    if thresholds.get("centerline_min_component_area") is not None:
        cmd.extend(
            [
                "--centerline-min-component-area",
                str(thresholds["centerline_min_component_area"]),
            ]
        )
    if thresholds.get("centerline_min_aspect") is not None:
        cmd.extend(["--centerline-min-aspect", str(thresholds["centerline_min_aspect"])])
    if thresholds.get("mask_min_x") is not None:
        cmd.extend(["--mask-min-x", str(thresholds["mask_min_x"])])
    if thresholds.get("mask_max_x") is not None:
        cmd.extend(["--mask-max-x", str(thresholds["mask_max_x"])])
    if thresholds.get("min_component_bbox_fill_frac") is not None:
        cmd.extend(
            [
                "--min-component-bbox-fill-frac",
                str(thresholds["min_component_bbox_fill_frac"]),
            ]
        )
    if thresholds.get("split_components"):
        cmd.append("--split-components")
    if thresholds.get("kmeans_split_instances") is not None:
        cmd.extend(
            ["--kmeans-split-instances", str(thresholds["kmeans_split_instances"])]
        )
    if thresholds.get("kmeans_split_y_weight") is not None:
        cmd.extend(
            ["--kmeans-split-y-weight", str(thresholds["kmeans_split_y_weight"])]
        )
    if thresholds.get("kmeans_split_gap_kernel") is not None:
        cmd.extend(
            ["--kmeans-split-gap-kernel", str(thresholds["kmeans_split_gap_kernel"])]
        )
    if thresholds.get("fill_holes"):
        cmd.append("--fill-holes")
    if thresholds.get("mask_close_kernel") is not None:
        cmd.extend(["--mask-close-kernel", str(thresholds["mask_close_kernel"])])
    if thresholds.get("mask_close_iterations") is not None:
        cmd.extend(
            ["--mask-close-iterations", str(thresholds["mask_close_iterations"])]
        )
    if thresholds.get("fill_convex"):
        cmd.append("--fill-convex")
    if thresholds.get("fill_convex_max_ratio") is not None:
        cmd.extend(
            ["--fill-convex-max-ratio", str(thresholds["fill_convex_max_ratio"])]
        )

    if profile:
        if profile.get("pattern"):
            cmd.extend(["--pattern", profile["pattern"]])
        if profile.get("image_list"):
            cmd.extend(["--image-list", str(resolve_config_path(profile["image_list"]))])
        for stem in profile.get("stems", []):
            cmd.extend(["--image-stem", stem])
        for prefix in profile.get("sequence_prefixes", []):
            cmd.extend(["--include-prefix", prefix])
        if profile.get("min_stem"):
            cmd.extend(["--min-stem", profile["min_stem"]])
        if profile.get("max_stem"):
            cmd.extend(["--max-stem", profile["max_stem"]])

    for prompt in prompts:
        cmd.extend(["--text-prompt", prompt])

    return cmd


def write_parameter_snapshots(config: dict, objects: list[dict]) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    for obj in objects:
        path = resolve_repo_path(obj["parameter_file"])
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": config["version"],
            "updated_for": config["updated_for"],
            "runner": config["runner"],
            "material": obj["label"],
            "source_name": obj.get("source_name"),
            "input_dir": obj["input_dir"],
            "output_dir": obj["output_dir"],
            "parameters": obj,
            "loaded_parameter_files": load_parameter_files(obj),
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def clean_outputs(objects: list[dict]) -> None:
    for obj in objects:
        output_dir = resolve_repo_path(obj["output_dir"])
        if output_dir.exists():
            shutil.rmtree(output_dir)


def run_objects(objects: list[dict]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    total = sum(max(1, len(load_parameter_files(obj))) for obj in objects)
    index = 0

    for obj in objects:
        profiles = load_parameter_files(obj) or [None]
        for profile in profiles:
            index += 1
            profile_name = profile["name"] if profile else "all"
            log_name = f"{index:02d}_{obj['label']}_{profile_name}.log"
            log_path = LOG_DIR / log_name
            print(f"[{index}/{total}] {obj['label']} {profile_name}", flush=True)
            with log_path.open("w", encoding="utf-8") as log_file:
                subprocess.run(
                    build_command(obj, profile),
                    cwd=REPO_ROOT,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    check=True,
                )


def main() -> None:
    args = parse_args()
    config_path = (
        args.config if args.config.is_absolute() else resolve_config_path(args.config)
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    labels = set(args.labels or [])
    objects = [
        obj
        for obj in config["objects"]
        if not labels or object_matches(obj, labels)
    ]
    if labels and len(objects) != len(labels):
        found = {name for obj in objects for name in labels if object_matches(obj, {name})}
        missing = sorted(labels - found)
        raise SystemExit(f"unknown label(s): {', '.join(missing)}")

    write_parameter_snapshots(config, objects)
    if args.clean:
        clean_outputs(objects)
    if not args.write_params_only:
        run_objects(objects)


if __name__ == "__main__":
    main()
