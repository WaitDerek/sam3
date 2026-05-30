import argparse
from pathlib import Path

import open3d as o3d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a colored PCD point cloud with Open3D.")
    parser.add_argument("pcd", type=Path, help="Path to the .pcd file.")
    parser.add_argument(
        "--screenshot",
        type=Path,
        default=None,
        help="Save a PNG render to this path instead of (or in addition to) showing a window.",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Offscreen render only. Requires --screenshot. Use this on headless machines.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=1.5,
        help="Render point size. Default: 1.5",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.0,
        help="Optional voxel downsample size in meters. 0 disables downsampling.",
    )
    parser.add_argument(
        "--background",
        choices=["black", "white"],
        default="black",
        help="Render background color. Default: black",
    )
    return parser.parse_args()


def load_cloud(path: Path, voxel_size: float) -> o3d.geometry.PointCloud:
    if not path.exists():
        raise FileNotFoundError(f"pcd not found: {path}")
    cloud = o3d.io.read_point_cloud(str(path))
    if cloud.is_empty():
        raise RuntimeError(f"pcd has no points: {path}")
    if voxel_size > 0:
        cloud = cloud.voxel_down_sample(voxel_size)
    return cloud


def show_window(cloud: o3d.geometry.PointCloud, point_size: float, background: str) -> None:
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="PCD viewer")
    vis.add_geometry(cloud)
    opt = vis.get_render_option()
    opt.point_size = point_size
    opt.background_color = (0, 0, 0) if background == "black" else (1, 1, 1)
    vis.run()
    vis.destroy_window()


def save_screenshot(
    cloud: o3d.geometry.PointCloud,
    out_path: Path,
    point_size: float,
    background: str,
    visible: bool,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="PCD render", visible=visible)
    vis.add_geometry(cloud)
    opt = vis.get_render_option()
    opt.point_size = point_size
    opt.background_color = (0, 0, 0) if background == "black" else (1, 1, 1)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(str(out_path), do_render=True)
    vis.destroy_window()
    print(f"Saved screenshot: {out_path}")


def main() -> None:
    args = parse_args()
    if args.no_window and args.screenshot is None:
        raise SystemExit("--no-window requires --screenshot to choose an output path.")

    cloud = load_cloud(args.pcd, args.voxel_size)
    print(f"Loaded: {args.pcd}")
    print(f"Points: {len(cloud.points)}")
    print(f"Has color: {cloud.has_colors()}")

    if args.screenshot is not None:
        save_screenshot(
            cloud=cloud,
            out_path=args.screenshot,
            point_size=args.point_size,
            background=args.background,
            visible=not args.no_window,
        )
    if not args.no_window and args.screenshot is None:
        show_window(cloud, args.point_size, args.background)


if __name__ == "__main__":
    main()
