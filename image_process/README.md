# RealSense 数据处理脚本说明

这个目录用于处理 RealSense 采集数据，主要流程包括：

1. 从 `.bag` 文件抽帧，生成 `bgr/` 和 `depth/`
2. 合并多个抽帧后的数据集
3. 以 `bgr/` 为准对齐 `depth/`
4. 用 `bgr + depth` 生成彩色点云 PCD
5. 用 SAM2/SAM3 风格流程生成目标 mask，并把 mask 映射到三维点云中高亮显示

常见目录结构：

```text
dataset_name/
  bgr/
    000000.png
  depth/
    000000.png
  mask/
    000000.png
  pcd/
    000000.pcd
    000000_filter.pcd
```

Windows 路径示例：`F:\dataset`

WSL 路径示例：`/mnt/f/dataset`

---

## `extract.py`

功能：

- 从 RealSense `.bag` 文件中抽帧
- 只生成 `bgr/` 和 `depth/`
- 不再直接生成 `pcd/`
- 支持把多个已抽帧数据集合并成一个 `merged_xxx` 数据集
- 支持抽帧后立即合并
- 支持对已有数据集重新编号

### 抽帧

```bash
python ~/workspace/realsense/extract.py \
  --bag-dir /mnt/f/data/20260521_ldgb \
  --output-root /mnt/f/dataset/20260521_ldgb \
  --save-interval 10
```

参数说明：

- `--bag-dir`：输入 `.bag` 文件所在目录
- `--output-root`：抽帧输出目录
- `--save-interval`：保存间隔，例如 `10` 表示每 10 帧保存一次
- `--overwrite`：允许覆盖已有输出目录

输出示例：

```text
/mnt/f/dataset/20260521_ldgb/
  bag_name/
    bgr/
    depth/
    meta/
```

### 合并数据集

```bash
python ~/workspace/realsense/extract.py \
  --merge \
  --source-root /mnt/f/dataset \
  --merge-output /mnt/f/dataset \
  --merge-name merged_ldgb \
  --apply
```

参数说明：

- `--merge`：进入合并模式
- `--source-root`：源数据集根目录
- `--merge-output`：合并结果输出位置
- `--merge-name`：合并后的文件夹名称
- `--apply`：真正执行复制；不加时只预览
- `--overwrite`：允许覆盖重名文件
- `--sources`：只合并指定的源文件夹名
- `--source-path`：指定任意源目录，可重复使用
- `--separator`：合并后文件名前缀与原文件名之间的分隔符，默认 `__`

合并规则：

- 默认合并 `bgr/` 和 `depth/`
- 已经存在的目标文件不会重复复制
- 不会因为源文件夹名已经出现过就跳过整个包，只要目标文件名不存在就会加入

### 抽帧后直接合并

```bash
python ~/workspace/realsense/extract.py \
  --bag-dir /mnt/f/data/20260521_ldgb \
  --output-root /mnt/f/dataset/20260521_ldgb \
  --save-interval 10 \
  --merge-after-extract \
  --merge-output /mnt/f/dataset \
  --merge-name merged_ldgb \
  --apply
```

### 重新编号

```bash
python ~/workspace/realsense/extract.py \
  --renumber \
  --dataset-dir /mnt/f/dataset/merged_ldgb \
  --start-index 0 \
  --digits 6
```

---

## `align_dataset.py`

功能：

- 递归检查数据集中的 `bgr/` 和 `depth/`
- 以 `bgr/` 为基准
- 删除 `depth/` 中多出来的文件
- 如果 `bgr/` 有但 `depth/` 没有，只报告，不删除 `bgr`

默认只对齐 `depth`。

### 删除 `depth` 多余文件

Windows：

```powershell
python D:\workspace\align_dataset.py F:\dataset --apply
```

WSL：

```bash
python ~/workspace/realsense/align_dataset.py /mnt/f/dataset --apply
```

常用参数：

- `root`：数据集根目录
- `--base`：基准文件夹，默认 `bgr`
- `--targets`：要对齐的目标文件夹，默认 `depth`
- `--patterns`：参与对齐的文件类型，默认 `*.png *.pcd`
- `--apply`：真正删除；不加时只预览

如果临时也要对齐 `pcd`：

```bash
python ~/workspace/realsense/align_dataset.py /mnt/f/dataset --targets depth pcd --apply
```

---

## `process_image.py`

功能：

- 输入 `bgr/` 和 `depth/`
- 用 SAM2 自动或手动提示分割图中明显目标
- 输出彩色 mask 图
- 把 mask 映射到三维点云中
- 输出 CloudCompare 可打开的彩色 PCD
- mask 对应区域在 PCD 中会被标红

当前输出：

```text
mask/<frame>.png
pcd/<frame>.pcd
```

如果开启深度滤波，PCD 输出为：

```text
pcd/<frame>_filter.pcd
```

### 自动分割并生成高亮点云

```bash
PYTHONPATH=/home/xue/workspace/sam2 HF_HUB_OFFLINE=1 \
python ~/workspace/realsense/process_image.py \
  --dataset-dir ~/workspace/realsense/sam3demo \
  --overwrite
```

默认目录结构：

```text
sam3demo/
  bgr/
  depth/
  mask/
  pcd/
```

### 指定某一帧

```bash
PYTHONPATH=/home/xue/workspace/sam2 HF_HUB_OFFLINE=1 \
python ~/workspace/realsense/process_image.py \
  --dataset-dir ~/workspace/realsense/sam3demo \
  --frame 20260521_112611__000000 \
  --overwrite
```

### 手动指定 box 或 point

```bash
PYTHONPATH=/home/xue/workspace/sam2 HF_HUB_OFFLINE=1 \
python ~/workspace/realsense/process_image.py \
  --dataset-dir ~/workspace/realsense/sam3demo \
  --box 760,350,990,510 \
  --point 875,425 \
  --overwrite
```

参数说明：

- `--box x1,y1,x2,y2`：手动指定目标框
- `--point x,y`：手动指定正样本点，可重复
- `--negative-point x,y`：手动指定负样本点，可重复
- 不加 `--box/--point` 时，脚本会自动检测明显前景物体，再交给 SAM2 分割

### 深度滤波

默认不滤波：

```bash
--depth-filter none
```

中值滤波：

```bash
--depth-filter median --median-ksize 5
```

双边滤波：

```bash
--depth-filter bilateral \
--bilateral-d 7 \
--bilateral-sigma-color 35 \
--bilateral-sigma-space 35
```

开启滤波后，输出文件名会带 `_filter`：

```text
pcd/<frame>_filter.pcd
```

其他常用参数：

- `--intrinsics-json`：指定相机内参文件
- `--depth-scale`：深度单位比例，默认 `1000.0`
- `--min-depth`：最小保留深度，单位米
- `--max-depth`：最大保留深度，单位米
- `--camera-frame`：保持相机坐标系 `x-right/y-down/z-forward`
- `--overwrite`：覆盖已有输出

---

## `bgr_depth_to_pcd.py`

功能：

- 不经过 SAM2
- 不生成 mask
- 直接把对齐的 `bgr/` 和 `depth/` 转成彩色点云 PCD
- 适合查看完整 RGB-D 场景点云

### 处理整个数据集

```bash
python ~/workspace/realsense/bgr_depth_to_pcd.py \
  --dataset-dir ~/workspace/realsense/sam3demo \
  --overwrite
```

输出：

```text
pcd/<frame>.pcd
```

### 只处理一帧

```bash
python ~/workspace/realsense/bgr_depth_to_pcd.py \
  --dataset-dir ~/workspace/realsense/sam3demo \
  --frame 20260521_112611__000000 \
  --overwrite
```

常用参数：

- `--dataset-dir`：包含 `bgr/` 和 `depth/` 的数据集目录
- `--bgr-dir`：单独指定 bgr 目录
- `--depth-dir`：单独指定 depth 目录
- `--output-dir`：指定 PCD 输出目录
- `--frame`：只处理指定帧；不加则处理全部同名帧
- `--intrinsics-json`：指定相机内参文件
- `--depth-scale`：深度单位比例，默认 `1000.0`
- `--min-depth`：最小保留深度，单位米
- `--max-depth`：最大保留深度，单位米
- `--camera-frame`：保持相机坐标系
- `--overwrite`：覆盖已有输出

---

## `intrinsic.json`

功能：

- 保存相机内参
- `process_image.py` 和 `bgr_depth_to_pcd.py` 会优先读取数据集里的内参
- 如果没有找到，会使用脚本目录下的 `intrinsic.json`
- 如果仍没有找到，会使用脚本内置默认内参

典型字段：

```json
{
  "width": 1280,
  "height": 720,
  "fx": 662.636,
  "fy": 662.636,
  "cx": 635.522,
  "cy": 348.738
}
```

---

## Windows 运行提示

如果在 Windows 原生 Python 中运行，路径使用 Windows 格式：

```powershell
python D:\workspace\realsense\extract.py --bag-dir F:\data --output-root F:\dataset --save-interval 10
```

如果通过 WSL 运行，路径使用 `/mnt/f/...`：

```powershell
wsl.exe -d Ubuntu-24.04 bash -lc "python ~/workspace/realsense/align_dataset.py /mnt/f/dataset --apply"
```

PowerShell 多行命令使用反引号：

```powershell
python D:\workspace\realsense\extract.py `
  --bag-dir F:\data `
  --output-root F:\dataset `
  --save-interval 10
```

Linux/WSL 多行命令使用反斜杠：

```bash
python ~/workspace/realsense/extract.py \
  --bag-dir /mnt/f/data \
  --output-root /mnt/f/dataset \
  --save-interval 10
```
