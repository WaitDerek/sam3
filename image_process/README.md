# image_process 标准运行框架

`image_process` 是本仓库中物料检测、深度图整理、点云生成和结果查看的工作区。后续本地运行只使用下面的目录布局：

```text
./
  dataset/        # 输入和中间数据，不提交 Git
    raw/          # 可选：RealSense .bag 或原始采集数据
    extracted/    # 可选：extract.py 抽帧结果
    merged/       # 可选：extract.py 合并结果
    test/         # 本地示例数据
    <物料目录>/
      bgr/
      depth/
  out/            # 检测、临时点云、试验结果，不提交 Git
  configs/        # 可提交的稳定检测参数
  changan/        # 已验收交付产物，不提交 Git
```

`changan` 是交付归档位置，只有确认完整的数据才放入这里。日常检测和调参输出先写入 `out`，不要直接覆盖 `changan`。

环境准备见 `环境配置说明.md`。

## 数据准备脚本

### `extract.py`

从 RealSense `.bag` 抽取 `bgr/depth`，也可合并和重编号已抽帧数据。默认路径已经收敛到 `dataset`：

```bash
python extract.py \
  --bag-dir dataset/raw \
  --output-root dataset/extracted \
  --save-interval 10
```

合并抽帧结果：

```bash
python extract.py \
  --merge \
  --source-root dataset/extracted \
  --merge-output dataset \
  --merge-name merged \
  --apply
```

重编号：

```bash
python extract.py \
  --renumber \
  --dataset-dir dataset/merged \
  --start-index 0 \
  --digits 6
```

### `align_dataset.py`

按 `bgr` 对齐 `depth`、`pcd` 等目录，删除目标目录里没有对应 RGB 的多余文件。默认是 dry-run，真正删除必须加 `--apply`。

```bash
python align_dataset.py dataset/<物料目录> --apply
```

## 检测脚本

### `run_material_segmentation.py`

配置驱动的推荐入口，读取 `configs/object_segmentation_params.json`，再调用 `detect_material_bgr.py`。

```bash
python run_material_segmentation.py --label 洗涤器水壶加注管总成
python run_material_segmentation.py
```

### `detect_material_bgr.py`

底层单物料检测入口。适合临时检测单张或少量图片，输出 `masks/metadata/overlays` 到 `out/<结果目录>`。

```bash
python detect_material_bgr.py \
  --input-dir dataset/<物料目录>/bgr \
  --output-dir out/<结果目录> \
  --label <物料名> \
  --image-stem <图片stem> \
  --text-prompt "<prompt>"
```

### `repair_material_segmentation.py`

历史补检批处理脚本，内置若干物料的修复范围和阈值。它会调用 SAM3，需要可用 CUDA。

## 点云脚本

### `mask_depth_to_pcd.py`

推荐的目标点云生成脚本。它使用 `bgr + depth + masks`，只导出目标物点云。

```bash
python mask_depth_to_pcd.py \
  --dataset-dir changan/洗涤器水壶加注管总成_1198 \
  --overwrite
```

### `bgr_depth_to_pcd.py`

全场景点云生成脚本，不使用 mask，适合检查 RGB-D 对齐和原始深度质量。

```bash
python bgr_depth_to_pcd.py \
  --dataset-dir dataset/test \
  --overwrite
```

### `visualize_pcd.py`

用 Open3D 查看或截图 PCD。

```bash
python visualize_pcd.py \
  changan/洗涤器水壶加注管总成_1198/pcd/20260520_153258__000000.pcd
```

## 内参

点云脚本按顺序读取：

1. 命令行 `--intrinsics-json`
2. `<dataset>/meta/intrinsic.json`
3. `<dataset>/intrinsic.json`
4. `intrinsic.json`
5. 脚本内置默认值

如果使用新的相机或原始分辨率数据，优先把对应内参放在数据集目录下，或显式传入 `--intrinsics-json`。
