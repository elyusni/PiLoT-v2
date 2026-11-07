# PiLoT-v2

基于 DOM/DSM 地图的无人机图像定位。主程序为 `main.py`，运行脚本为 `run_feicuiwan.sh`。

## 1. 环境配置

环境：Linux x86_64、Python 3.9、PyTorch 2.8.0+cu128、TorchVision 0.23.0+cu128、
NumPy 1.26.4、segmentation-models-pytorch 0.3.4。运行主程序需要可用的 NVIDIA GPU。
随仓库提供的 `.so` 仅适用于 Python 3.9 / Linux x86_64。

创建环境示例：

```bash
conda create -n pilot-v2 -c conda-forge python=3.9 numpy=1.26.4 gdal=3.11.3 rasterio=1.4.3 pyproj=3.6.1 -y
conda activate pilot-v2

pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install numpy==1.26.4 segmentation-models-pytorch==0.3.4 omegaconf==2.3.0 opencv-python==4.10.0.84 scipy pyyaml tqdm matplotlib pykalman ipdb
```

检查 GPU：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## 2. 准备数据与路径

查询数据目录结构：

```text
<dataset_path>/
├── images/<name>/       # 查询图像，文件名主体为数字，例如 0.png、1.jpg
└── poses/<name>.txt     # 位姿，每行：image_name lon lat alt roll pitch yaw
```

运行前修改以下位置：

| 文件 | 需要配置的内容 |
| --- | --- |
| `configs/feicuiwan_m4t.yaml` | `default_confs.dataset_path`、`from_render_test.checkpoint` 和 `cam_query` 相机参数 |
| `main.py` | `ref_DOM_path`、`ref_DSM_path`、`ref_npy_path`；裁剪和定位两处的 DSM 路径需一致 |
| `main.py` | `output_folder`、`pre_path`、`depth_txt_folder`、`angle_txt_folder` |
| `run_feicuiwan.sh` | `target_names` 和 `read_pose_from_file()` 中的位姿文件路径；目标名称也需出现在 `names` 中 |


## 3. 运行

在仓库根目录执行：

```bash
conda activate pilot-v2
bash run_feicuiwan.sh
```

该脚本调用 `main.py`，使用 `configs/feicuiwan_m4t.yaml`，按 `target_names` 运行序列。
也可以直接运行单个序列：

```bash
python main.py --config configs/feicuiwan_m4t.yaml --name DJI_20250612182732_0001_V_test
```

估计位姿保存为 `<output_folder>/<name>.txt`，裁剪图像保存在 `<output_folder>/<name>/`。



# PiLoT-v2
