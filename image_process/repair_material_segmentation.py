"""Re-run detection on previously-empty images with material-specific repair prompts.

Loads SAM3.1 once, then for each material listed in REPAIRS:
  - reads the list of empty image stems from out/_logs/_repair/<material>_empty.txt
  - deletes existing empty outputs (overlay/mask/metadata) so runs are idempotent
  - re-runs collect_detections() with repair prompts/thresholds
  - writes new outputs into the same out/<material>/ directory tree
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from material_paths import REPO_ROOT, TOOL_DIR

ROOT = TOOL_DIR
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from detect_material_bgr import (  # noqa: E402  — its import already fixes sys.path for sam3
    build_predictor,
    collect_detections,
    save_outputs,
)


class _Args:
    """Minimal stand-in for argparse.Namespace expected by build_predictor."""

    def __init__(self, score_threshold_detection: float, new_det_thresh: float):
        self.score_threshold_detection = score_threshold_detection
        self.new_det_thresh = new_det_thresh


REPAIRS: list[dict] = [
    {
        "label": "空调新风口",
        "material": "fresh_air_vent",
        "input_dir": ROOT / "dataset/空调新风口总成_906/bgr",
        "output_dir": ROOT / "out/fresh_air_vent",
        "empty_list": ROOT / "out/_logs/_repair/fresh_air_vent_empty.txt",
        "thresholds": {
            "min_score": 0.08,
            "min_area_frac": 0.0005,
            "max_area_frac": 0.45,
            "max_bbox_fill_frac": 0.95,
            "iou_dedup": 0.5,
            "score_threshold_detection": 0.05,
            "new_det_thresh": 0.08,
        },
        "prompts": [
            "black plastic air duct on green table",
            "black plastic vent duct close-up on workbench",
            "black automotive plastic duct housing on table",
            "single black plastic duct on green surface",
            "two black plastic air ducts side by side on table",
            "large black plastic duct in foreground",
        ],
    },
    {
        "label": "保险杠",
        "material": "bumper_624",
        "input_dir": ROOT / "dataset/前保险杠侧安装支架总成_624/bgr",
        "output_dir": ROOT / "out/bumper_563",
        "empty_list": ROOT / "out/_logs/_repair/bumper_624_empty.txt",
        "thresholds": {
            "min_score": 0.05,
            "min_area_frac": 0.0002,
            "max_area_frac": 0.09,
            "max_bbox_fill_frac": 0.95,
            "iou_dedup": 0.5,
            "score_threshold_detection": 0.03,
            "new_det_thresh": 0.05,
        },
        "prompts": [
            "small black plastic bumper bracket on green table",
            "small dark plastic part centered on teal background",
            "small black plastic part on green-teal floor",
            "small black automotive bracket close-up",
            "black plastic bumper mounting bracket on flat surface",
        ],
    },
    {
        "label": "背门自动开闭系统",
        "material": "power_tailgate_system_854",
        "input_dir": ROOT / "dataset/背门自动开闭系统ECU控制器总成_854/bgr",
        "output_dir": ROOT / "out/power_tailgate_system_797",
        "empty_list": ROOT / "out/_logs/_repair/power_tailgate_system_854_empty.txt",
        "thresholds": {
            "min_score": 0.1,
            "min_area_frac": 0.0003,
            "max_area_frac": 0.055,
            "max_bbox_fill_frac": 0.9,
            "iou_dedup": 0.5,
            "score_threshold_detection": 0.05,
            "new_det_thresh": 0.08,
        },
        "prompts": [
            "two black ECU modules with white labels in cardboard storage box",
            "black ECU module with white label at left side of cardboard tray",
            "black ECU module with white label at right side of cardboard tray",
            "small black control module with white label sticker in cardboard tray",
            "black power tailgate ECU module with white label",
            "small dark plastic part at corner of cardboard tray",
        ],
    },
    {
        "label": "三角牌",
        "material": "warning_triangle_486",
        "input_dir": ROOT / "dataset/三角警告牌_486/bgr",
        "output_dir": ROOT / "out/warning_triangle_486",
        "empty_list": ROOT / "out/_logs/_repair/warning_triangle_486_empty.txt",
        "thresholds": {
            "min_score": 0.15,
            "min_area_frac": 0.001,
            "max_area_frac": 0.10,
            "max_bbox_fill_frac": 0.95,
            "iou_dedup": 0.2,
            "score_threshold_detection": 0.08,
            "new_det_thresh": 0.12,
        },
        "prompts": [
            "complete long red rectangular plastic warning triangle case",
            "full length red emergency triangle storage case with printed labels",
            "elongated red plastic case from end to end",
            "red plastic warning triangle case including printed text section",
            "long thin red plastic case with end caps",
            "long red automotive emergency triangle case full body",
            "red rectangular warning triangle storage case complete",
        ],
    },
    {
        "label": "空滤器",
        "material": "air_filter",
        "input_dir": ROOT / "dataset/空滤器进气连接管总成561",
        "output_dir": ROOT / "out/air_filter",
        "empty_list": ROOT / "out/_logs/_repair/air_filter_empty.txt",
        "thresholds": {
            "min_score": 0.15,
            "min_area_frac": 0.001,
            "max_area_frac": 0.06,
            "max_bbox_fill_frac": 0.7,
            "iou_dedup": 0.4,
            "score_threshold_detection": 0.1,
            "new_det_thresh": 0.15,
        },
        "prompts": [
            "black S-curved automotive air intake hose with corrugated middle",
            "black ribbed flexible automotive intake duct",
            "black trumpet-shaped intake horn with corrugated hose",
            "single black plastic air intake duct lying in tray",
            "black plastic intake duct with flared mouth",
            "black plastic intake duct with vertical hose and angled horn",
            "black plastic air intake elbow with corrugated hose",
        ],
    },
    {
        "label": "副雨刮器",
        "material": "passenger_wiper_1113",
        "input_dir": ROOT / "dataset/副雨刮器总成_1113/bgr",
        "output_dir": ROOT / "out/passenger_wiper_1100",
        "empty_list": ROOT / "out/_logs/_repair/passenger_wiper_1113_empty.txt",
        "thresholds": {
            "min_score": 0.10,
            "min_area_frac": 0.0005,
            "max_area_frac": 0.08,
            "max_bbox_fill_frac": 0.55,
            "iou_dedup": 0.3,
            "score_threshold_detection": 0.05,
            "new_det_thresh": 0.10,
        },
        "prompts": [
            "black automotive windshield wiper blade",
            "passenger side windshield wiper blade",
            "long horizontal black rubber wiper with central pivot hinge",
            "rear windshield wiper blade with central black hinge mechanism",
            "thin curved black rubber wiper blade strip",
            "long thin black rubber wiper rubber",
            "front windshield wiper blade",
            "two windshield wiper blades on green table",
            "black wiper blade with rounded end caps",
        ],
    },
    {
        "label": "洗涤器水壶加注管总成",
        "material": "washer_filler_769",
        "input_dir": ROOT / "dataset/洗涤器水壶加注管总成_769",
        "output_dir": ROOT / "out/washer_filler_768",
        "empty_list": ROOT / "out/_logs/_repair/washer_filler_769_empty.txt",
        "thresholds": {
            "min_score": 0.40,
            "min_area_frac": 0.001,
            "max_area_frac": 0.04,
            "max_bbox_fill_frac": 0.85,
            "iou_dedup": 0.4,
            "score_threshold_detection": 0.20,
            "new_det_thresh": 0.30,
        },
        "prompts": [
            "small white plastic cylindrical filler neck with blue cap",
            "white plastic short tube with blue rubber cap end",
            "blue plastic cap on white plastic tube",
            "small white plastic washer fluid filler neck and blue cap only",
            "blue capped white plastic nozzle",
        ],
    },
    {
        "label": "泡沫块",
        "material": "foam_block_979",
        "input_dir": ROOT / "dataset/后轮鼓包内后侧泡沫块_979/bgr",
        "output_dir": ROOT / "out/foam_block_979",
        "empty_list": ROOT / "out/_logs/_repair/foam_block_979_empty.txt",
        "thresholds": {
            "min_score": 0.20,
            "min_area_frac": 0.001,
            "max_area_frac": 0.10,
            "max_bbox_fill_frac": 0.98,
            "iou_dedup": 0.2,
            "score_threshold_detection": 0.12,
            "new_det_thresh": 0.18,
        },
        "prompts": [
            "small black foam cube on shiny silver foil",
            "dark foam sponge cube on reflective metal surface",
            "small dark cubic block on silver background",
            "small blue rectangular foam pad on silver foil",
            "black cubic foam pad in metal-lined tray",
            "two foam blocks in tray one dark one blue",
            "small black sponge cube",
            "small blue sponge foam pad",
        ],
    },
]


def delete_existing_outputs(output_dir: Path, stem: str) -> None:
    for sub, suffix in (
        ("overlays", "_det.jpg"),
        ("masks", "_mask.png"),
        ("metadata", ".json"),
    ):
        path = output_dir / sub / f"{stem}{suffix}"
        if path.exists():
            path.unlink()


def run_repair(predictor, repair: dict, log_fp) -> tuple[int, int, int]:
    input_dir: Path = repair["input_dir"]
    output_dir: Path = repair["output_dir"]
    stems = [
        line.strip()
        for line in repair["empty_list"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    thresholds = repair["thresholds"]
    prompts = repair["prompts"]

    total = len(stems)
    recovered = 0
    still_empty = 0
    missing = 0

    print(f"\n=== {repair['material']} — {total} images ===", flush=True)
    log_fp.write(f"\n=== {repair['material']} — {total} images ===\n")

    for idx, stem in enumerate(stems, 1):
        image_path = input_dir / f"{stem}.png"
        if not image_path.exists():
            missing += 1
            line = f"[{idx}/{total}] MISSING {image_path.name}\n"
            log_fp.write(line)
            continue

        delete_existing_outputs(output_dir, stem)

        try:
            union_mask, metadata = collect_detections(
                predictor=predictor,
                image_path=image_path,
                prompts=prompts,
                min_score=thresholds["min_score"],
                min_area_frac=thresholds["min_area_frac"],
                max_area_frac=thresholds.get("max_area_frac"),
                max_bbox_fill_frac=thresholds.get("max_bbox_fill_frac"),
                iou_dedup=thresholds["iou_dedup"],
            )
        except Exception as exc:  # pragma: no cover — surface any runtime errors
            line = f"[{idx}/{total}] ERROR {stem}: {exc}\n"
            print(line.rstrip(), flush=True)
            log_fp.write(line)
            continue

        save_outputs(
            image_path=image_path,
            output_dir=output_dir,
            label=repair["label"],
            prompts=prompts,
            union_mask=union_mask,
            metadata=metadata,
        )

        n = len(metadata)
        status = "RECOVERED" if n > 0 else "still_empty"
        if n > 0:
            recovered += 1
        else:
            still_empty += 1
        log_fp.write(
            f"[{idx}/{total}] {status} {stem} ({n} det)\n"
        )
        if idx % 25 == 0:
            print(
                f"  progress {idx}/{total}: recovered={recovered} still_empty={still_empty}",
                flush=True,
            )
            log_fp.flush()

    summary = (
        f"=== {repair['material']} done — recovered={recovered}, "
        f"still_empty={still_empty}, missing={missing}, total={total}\n"
    )
    print(summary, flush=True)
    log_fp.write(summary)
    log_fp.flush()
    return recovered, still_empty, missing


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--material",
        action="append",
        dest="materials",
        help="Only run this material (e.g. fresh_air_vent). Repeat for multiple.",
    )
    args = parser.parse_args()

    selected = (
        [r for r in REPAIRS if r["material"] in set(args.materials)]
        if args.materials
        else REPAIRS
    )
    if not selected:
        raise SystemExit(f"no matching materials in: {args.materials}")

    log_dir = ROOT / "out" / "_logs" / "_repair"
    log_dir.mkdir(parents=True, exist_ok=True)

    first = selected[0]
    predictor = build_predictor(
        _Args(
            score_threshold_detection=first["thresholds"]["score_threshold_detection"],
            new_det_thresh=first["thresholds"]["new_det_thresh"],
        )
    )

    summary_path = log_dir / "_summary.json"
    summary: dict = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary = {}

    for repair in selected:
        predictor.model.score_threshold_detection = repair["thresholds"][
            "score_threshold_detection"
        ]
        predictor.model.new_det_thresh = repair["thresholds"]["new_det_thresh"]
        log_path = log_dir / f"{repair['material']}.log"
        with log_path.open("w", encoding="utf-8") as log_fp:
            log_fp.write(
                f"prompts: {json.dumps(repair['prompts'], ensure_ascii=False)}\n"
            )
            log_fp.write(
                f"thresholds: {json.dumps(repair['thresholds'])}\n"
            )
            log_fp.write(
                f"predictor thresholds: score_threshold_detection="
                f"{predictor.model.score_threshold_detection}, "
                f"new_det_thresh={predictor.model.new_det_thresh}\n"
            )
            t0 = time.time()
            recovered, still_empty, missing = run_repair(predictor, repair, log_fp)
            elapsed = time.time() - t0
            log_fp.write(f"elapsed_seconds: {elapsed:.1f}\n")
        summary[repair["material"]] = {
            "recovered": recovered,
            "still_empty": still_empty,
            "missing": missing,
            "elapsed_seconds": round(elapsed, 1),
            "log": str(log_path.relative_to(ROOT)),
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
