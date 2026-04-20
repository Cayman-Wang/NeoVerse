# NeoVerse：面向野外单目视频的 4D 世界模型增强

<strong>Computer Vision and Pattern Recognition（CVPR 2026 Highlight）</strong>

[Yuxue Yang](https://yuxueyang1204.github.io)<sup>1, 2</sup>，[Lue Fan](https://lue.fan)<sup>1 ✉️ †</sup>，[Ziqi Shi](https://renshengji.github.io)<sup>1</sup>，[Junran Peng](https://jrpeng.github.io)<sup>1</sup>，[Feng Wang](https://happynear.wang)<sup>2</sup>，[Zhaoxiang Zhang](https://zhaoxiangzhang.net)<sup>1 ✉️</sup>

<sup>1</sup>NLPR & MAIS, CASIA&emsp; <sup>2</sup>CreateAI

<sup>✉️</sup>通讯作者&emsp; <sup>†</sup>项目负责人

<a href='https://arxiv.org/abs/2601.00393'><img src='https://img.shields.io/badge/arXiv-2601.00393-b31b1b?logo=arxiv'></a> &nbsp;
<a href='https://neoverse-4d.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;
<a href='https://huggingface.co/Yuppie1204/NeoVerse'><img src='https://img.shields.io/badge/Hugging Face-Model-gold?logo=huggingface'></a> &nbsp;
<a href='https://www.modelscope.cn/models/Yuppie1204/NeoVerse'><img src='https://img.shields.io/badge/ModelScope-Model-blueviolet?logo=modelscope'></a> &nbsp;
<a href='https://www.bilibili.com/video/BV1ezvYBBEMi'><img src='https://img.shields.io/badge/BiliBili-Video-479fd1?logo=bilibili'></a> &nbsp;
<a href='https://youtu.be/1k8Ikf8zbZw'><img src='https://img.shields.io/badge/YouTube-Video-orange?logo=youtube'></a>

NeoVerse 是一个通用的 4D 世界模型，能够完成 4D 重建、新视角轨迹视频生成以及丰富的下游应用。

https://github.com/user-attachments/assets/4c957bd7-64e1-4a7e-9993-136740d911fe

**更多演示视频可在 [项目网站](https://neoverse-4d.github.io) 查看，以获得更好的观看体验。**

## 更新记录

- **[2026-04-14]** 发布训练脚本与可复现的演示 Notebook。
- **[2026-04-09]** NeoVerse 被选为 **highlight** 论文！
- **[2026-02-21]** NeoVerse 被 **CVPR 2026** 接收！
- **[2026-02-16]** 已在 [Hugging Face](https://huggingface.co/Yuppie1204/NeoVerse) 和 [ModelScope](https://www.modelscope.cn/models/Yuppie1204/NeoVerse) 发布推理脚本与模型检查点。
- **[2026-01-01]** 发布 arXiv 论文版本。

## 一句话概览

- **简单推理脚本**：只需一条 `python inference.py` 命令即可生成新轨迹视频
- **交互式 Gradio Demo**：提供重建、轨迹设计和生成的分步式 Web UI
- **支持多种重建器**：通过即插即用接口支持不同的 3D 重建器，例如 [Depth Anything 3](https://depth-anything-3.github.io/)
- **推理速度快**：在单张 A800 上，使用蒸馏 LoRA 加速后，整套推理流程可在 30 秒内完成
- **支持训练**：已发布控制分支训练代码，可在自有单目视频数据上进行微调
- **演示 Notebook**：`docs/` 中提供可复现的下游应用 Notebook

## 安装

### 第 1 步：安装依赖

我们在 CUDA 12.1 + PyTorch 2.3.1，以及 CUDA 12.8 + PyTorch 2.7.1 上验证过 NeoVerse。

```bash
git clone https://github.com/IamCreateAI/NeoVerse.git
cd NeoVerse
conda create -n neoverse python=3.10 -y
conda activate neoverse

# CUDA 12.1
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.3.1+cu121.html
pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git

# CUDA 12.8
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.7.1+cu128.html
pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git
```

### 第 2 步：下载模型检查点

```bash
hf download Yuppie1204/NeoVerse --local-dir models/NeoVerse
# 或使用 ModelScope
modelscope download --model Yuppie1204/NeoVerse --local_dir models/NeoVerse
```

期望的目录结构：
```
models/NeoVerse/
├── diffusion_pytorch_model-0000*-of-00006.safetensors
├── diffusion_pytorch_model.safetensors.index.json
├── models_t5_umt5-xxl-enc-bf16.pth
├── reconstructor.ckpt
├── Wan2.1_VAE.pth
├── google/
│   └── ...（tokenizer 文件）
└── loras/
    └── Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors
```

## 使用说明

NeoVerse 提供两种试用方式：**命令行推理脚本** 和 **交互式 Gradio Demo**。

### 推理脚本

推理脚本支持两种轨迹输入方式：

#### 预定义轨迹与可调参数

使用 `--trajectory` 可以从 13 种内置相机运动中选择一种，并通过 `--angle`、`--distance` 或 `--orbit_radius` 进行微调：

| 轨迹 | 说明 |
|-----------|-------------|
| `pan_left` / `pan_right` | 水平旋转相机（yaw） |
| `tilt_up` / `tilt_down` | 垂直旋转相机（pitch） |
| `move_left` / `move_right` | 水平平移相机 |
| `push_in` / `pull_out` | 沿前后方向平移相机 |
| `boom_up` / `boom_down` | 垂直方向平移相机 |
| `orbit_left` / `orbit_right` | 围绕场景中心做弧形运动 |
| `static` | 保持原始相机路径不变 |

```bash
# 向上倾斜
python inference.py \
    --input_path examples/videos/robot.mp4 \
    --trajectory tilt_up \
    --prompt "A two-arm robot assembles parts in front of a table." \
    --output_path outputs/tilt_up.mp4

# 向右移动 0.2 个单位
python inference.py \
    --input_path examples/videos/tree_and_building.mp4 \
    --trajectory move_right \
    --distance 0.2 \
    --output_path outputs/move_right.mp4

# 通过调整焦距实现 2 倍放大
python inference.py \
    --input_path examples/videos/animal.mp4 \
    --trajectory static \
    --zoom_ratio 2.0 \
    --output_path outputs/zoom_in.mp4
```

#### 来自 JSON 的自定义轨迹

如果需要更细粒度的关键帧控制，可以通过 `--trajectory_file` 提供轨迹 JSON 文件：

```bash
# 先向左环绕，再拉远
python inference.py \
    --input_path examples/videos/movie.mp4 \
    --trajectory_file examples/trajectories/orbit_left_pull_out.json \
    --alpha_threshold 0.95 \
    --output_path outputs/orbit_left_pull_out.mp4

# 自定义轨迹
python inference.py \
    --input_path examples/videos/driving.mp4 \
    --trajectory_file examples/trajectories/custom.json \
    --output_path outputs/custom_traj.mp4

# 在静态场景上使用自定义轨迹（单张图片输入）
python inference.py \
    --input_path examples/videos/jungle.png \
    --static_scene \
    --trajectory_file examples/trajectories/custom2.json \
    --output_path outputs/custom_traj2.mp4

# 稀疏关键帧位姿并进行插值
python inference.py \
    --input_path examples/videos/driving2.mp4 \
    --trajectory_file examples/trajectories/sparse_matrices.json \
    --output_path outputs/keyframe_interpolation.mp4
```

JSON 模式的格式说明见 [docs/trajectory_format.md](docs/trajectory_format.md)，坐标约定见 [docs/coordinate_system.md](docs/coordinate_system.md)。可直接使用的示例位于 `configs/trajectories/`。

如果只想检查轨迹文件是否合法，而不运行推理，可以执行：

```bash
python inference.py --trajectory_file my_trajectory.json --validate_only
```

#### 关键参数

| 参数 | 默认值 | 说明 |
|----------|---------|-------------|
| `--input_path` | — | 输入视频或图片路径 |
| `--trajectory` | — | 预定义轨迹类型（见上表） |
| `--trajectory_file` | — | 自定义轨迹 JSON 路径（与 `--trajectory` 二选一） |
| `--output_path` | `outputs/inference.mp4` | 输出视频文件路径 |
| `--prompt` | *(scene inpainting prompt)* | 生成时使用的文本提示词 |
| `--static_scene` | 关闭 | 启用静态场景模式（见下文） |
| `--traj_mode` | `relative` | 轨迹坐标模式（见下文） |
| `--alpha_threshold` | `1.0` | Alpha 掩码阈值（见下文） |
| `--reconstructor_path` | `models/NeoVerse/reconstructor.ckpt` | 重建器检查点路径 |
| `--num_frames` | `81` | 输出帧数 |
| `--height` / `--width` | `336` / `560` | 输出分辨率 |
| `--disable_lora` | 关闭 | 使用完整的 50 步推理，而不是 4 步蒸馏 LoRA |
| `--low_vram` | 关闭 | 启用低显存模式与模型卸载（见下文） |
| `--vis_rendering` | 关闭 | 额外保存目标轨迹的渲染可视化结果 |
| `--seed` | `42` | 随机种子 |

**场景类型**（`--static_scene`）—— 默认情况下，NeoVerse 将输入视为*一般场景*：会在整个时间范围内采样帧，以捕捉相机和物体运动。启用 `--static_scene` 后，所有帧共享同一时间戳，适合单张图像或相机完全静止的视频。

**模式**（`--traj_mode`）—— 在 `relative` 模式（默认）下，设计好的轨迹会与重建得到的输入相机姿态组合，因此运动是相对于原始视角的；在 `global` 模式下，轨迹矩阵会直接在世界坐标系中使用。

**Alpha 阈值**（`--alpha_threshold`）—— 从重建的 3D 场景渲染目标视角后，alpha 低于该阈值的像素会被遮罩，并由扩散模型重新绘制。默认值 `1.0` 会保留全部区域进行重绘。

**低显存模式**（`--low_vram`）—— 通过模型卸载减少 GPU 峰值显存占用。启用后，模型保留在 CPU 上，仅在需要时加载到 GPU（例如，重建时加载重建器，完成后卸载；去噪时加载扩散模型，完成后再卸载）。这会显著降低峰值 VRAM，但由于 CPU-GPU 数据传输，推理速度会变慢。默认模式下 GPU 上约分配 47 GB（`torch.cuda.memory_allocated`），峰值约 74 GB（`torch.cuda.max_memory_allocated`）；而 `--low_vram` 仅保持约 1 GB 分配，峰值降低到约 38 GB。

### 交互式 Demo（Gradio）

启动 Web UI：

```bash
python app.py

# 低显存模式
python app.py --low_vram
```

Demo 会引导你完成四个步骤：

1. **上传**：上传视频或图片组，并选择场景类型（General / Static）。
2. **重建**：点击 `Reconstruct` 构建 4D Gaussian Splat 场景。3D 视图会显示以 Gaussian-Splatting 为中心的点云，便于检查空间布局。
3. **设计轨迹**：选择相机运动类型并调整滑条，或上传轨迹 JSON。点击 `Render` 预览 RGB 与 mask 渲染结果。
4. **生成**：输入提示词并点击 `Generate`，生成最终视频。

### 其他重建器

NeoVerse 也支持 [Depth Anything 3](https://depth-anything-3.github.io/) 等其他重建器。它们预测的深度和相机参数可以转换为伪 Gaussian splats，从而接入 NeoVerse 的流水线。

下载 Depth Anything 3 的检查点：

```bash
# 从 Hugging Face 下载 model.safetensors
wget https://huggingface.co/depth-anything/DA3-GIANT-1.1/resolve/main/model.safetensors -O models/da3_giant_1.1.safetensors
```

然后通过 `--reconstructor_path` 传入：

```bash
# 使用 Depth Anything 3 进行命令行推理
python inference.py \
    --input_path examples/videos/driving.mp4 \
    --trajectory_file examples/trajectories/custom.json \
    --reconstructor_path models/da3_giant_1.1.safetensors \
    --output_path outputs/custom_traj_da3.mp4

# 使用 Depth Anything 3 启动 Gradio Demo
python app.py --reconstructor_path models/da3_giant_1.1.safetensors
```

## Demo Notebook（持续补充中）

我们在 [`docs/`](docs/) 中提供了可复现的 Jupyter Notebook，覆盖项目页展示的部分下游应用：

| Notebook | 说明 |
|----------|------|
| [`docs/single_view_to_multi_view.ipynb`](docs/single_view_to_multi_view.ipynb) | 通过连续相机平移，将单视角视频逐步扩展为多视角序列 |
| [`docs/video_editing.ipynb`](docs/video_editing.ipynb) | 在保持周围场景一致的前提下，按文本提示编辑视频中的掩码区域 |
| [`docs/camera_stabilization.ipynb`](docs/camera_stabilization.ipynb) | 结合 SLERP 与高斯滤波平滑恢复轨迹，实现手持视频稳像 |

**说明：** 如果你要做更强的反事实编辑，建议先用 [VACE](https://github.com/ali-vilab/VACE) 这类专业视频编辑模型在单视角视频上完成编辑，再按 [`docs/single_view_to_multi_view.ipynb`](docs/single_view_to_multi_view.ipynb) 的方式送入 NeoVerse 生成多视角或新视角结果。

## 训练

我们发布了 NeoVerse 控制分支的训练代码，可用于在自有单目视频数据上继续微调。

### 训练数据

我们在 `data/SpatialVID/` 中提供了来自 [SpatialVID](https://huggingface.co/datasets/FelixYuan/SpatialVID-HQ) 的 20 段示例片段作为最小可运行示例。训练时**只需要 RGB 视频帧和文本提示**，不需要深度图、外参和内参；三维结构与相机位姿由重建器在线估计，因此可以较容易迁移到通用野外单目视频数据。

### 基于 ZeRO-2 的多机训练

训练脚本基于 [Accelerate](https://huggingface.co/docs/accelerate) 与 DeepSpeed ZeRO Stage 2，示例命令如下：

```bash
accelerate launch \
    --use_deepspeed \
    --deepspeed_config_file training/configs/zero_stage2_config.json \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --num_machines $WORLD_SIZE \
    --num_processes $NUM_PROCESS \
    --machine_rank $NODE_RANK \
    --deepspeed_multinode_launcher standard \
    train.py training/configs/train.yaml
```

训练配置（数据路径、学习率、batch size 等）见 [`training/configs/train.yaml`](training/configs/train.yaml)。

## 模型结构

NeoVerse 主要由两部分组成：

1. **重建器**：从单目视频中恢复 3D 场景结构（Gaussian Splats + 相机位姿）。在当前发布版本中，我们提供了一个基于 [WorldMirror](https://3d-models.hunyuan.tencent.com/world/) 的重建器，并在 3D/4D 数据集上进行了微调。此外，NeoVerse 也可以兼容 [Depth Anything 3](https://depth-anything-3.github.io/) 等其他重建器，只需将其输出转换为伪 Gaussian splats 即可。
2. **视频扩散模型**：在重建场景的条件下生成高质量视频帧。这里使用的是 [WAN 2.1](https://github.com/Wan-Video/Wan2.1) 主干，并结合 [4 步蒸馏 LoRA](https://huggingface.co/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill-Lightx2v/tree/main/loras) 来提升推理速度。

技术细节请参考我们的 [论文](https://arxiv.org/abs/2601.00393)。

## 引用

如果你觉得这项工作有帮助，欢迎给仓库点 star，并考虑按以下格式引用。非常感谢！

```bibtex
@article{yang2026neoverse,
  title={NeoVerse: Enhancing 4D World Model with in-the-wild Monocular Videos},
  author={Yang, Yuxue and Fan, Lue and Shi, Ziqi and Peng, Junran and Wang, Feng and Zhang, Zhaoxiang},
  journal={arXiv preprint arXiv:2601.00393},
  year={2026}
}
```

## 致谢

感谢 [VGGT](https://vgg-t.github.io/)、[WorldMirror](https://3d-models.hunyuan.tencent.com/world/)、[Depth Anything 3](https://depth-anything-3.github.io/)、[Wan-Video](https://github.com/Wan-Video/Wan2.1)、[TrajectoryCrafter](https://trajectorycrafter.github.io/)、[ReCamMaster](https://jianhongbai.github.io/ReCamMaster/) 和 [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) 的开创性工作，以及它们对 3D 和视频生成社区的贡献。

## 联系方式

我们相信 NeoVerse 具有支持广泛应用的潜力，也期待社区基于它继续探索和扩展。如果你有任何问题、建议，或者想分享自己的结果，欢迎通过邮件 [yangyuxue2023@ia.ac.cn](mailto:yangyuxue2023@ia.ac.cn) 或微信（[Yuppie898988](Yuppie898988)）联系我们。也欢迎在 GitHub 上提交 issue，反馈 bug 或功能需求。
