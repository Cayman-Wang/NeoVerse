# NeoVerse 中文快速上手

这份文档是 NeoVerse README 的配套速查版，目标是让你在最短路径内跑起来。

## 你会得到什么

- 一条命令完成单目视频到新视角视频的推理
- 一个可交互的 Gradio 界面，适合逐步调试重建和轨迹
- 对不同重建器的兼容能力，比如 Depth Anything 3

## 最小流程

1. 安装依赖，建议使用 Python 3.10 的独立环境。
2. 下载模型检查点到 `models/NeoVerse/`。
3. 先用 `python inference.py --validate_only` 检查轨迹 JSON 是否有效。
4. 运行 `python inference.py --input_path ... --trajectory ...` 生成结果。
5. 如果你更想看过程，运行 `python app.py` 打开 Web UI。

## 推荐先试的命令

```bash
python inference.py \
  --input_path examples/videos/robot.mp4 \
  --trajectory tilt_up \
  --prompt "A two-arm robot assembles parts in front of a table." \
  --output_path outputs/tilt_up.mp4
```

如果显存紧张，可以改用：

```bash
python app.py --low_vram
```

## 容易踩坑的参数

- `--static_scene` 适合单张图片或完全静止的相机视频。
- `--traj_mode relative` 适合大多数场景；如果你已经在世界坐标里写好了轨迹，可以考虑 `global`。
- `--alpha_threshold` 越低，保留的已重建区域越多；越高，则更多区域会交给扩散模型重绘。
- `--low_vram` 会明显省显存，但速度更慢。

## 轨迹文件放哪儿

官方示例轨迹在 `configs/trajectories/`，JSON 结构说明见 `docs/trajectory_format.md`，坐标系约定见 `docs/coordinate_system.md`。

## 建议的验证顺序

如果你要在自己的环境里复现，建议按这个顺序确认：

1. 依赖是否安装成功。
2. 模型文件是否完整下载。
3. `--validate_only` 是否通过。
4. Demo 页面是否可以打开。
5. 再跑正式推理。