import argparse
import os
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
NEOVERSE_ROOT = SCRIPT_DIR.parent
if str(NEOVERSE_ROOT) not in sys.path:
    sys.path.insert(0, str(NEOVERSE_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="NeoVerse static/dynamic split export")
    parser.add_argument("--input_path", required=True, help="Input video/image path")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--split_mode", choices=["auto", "motion", "mask"], default="auto")
    parser.add_argument("--mask_dir", default=None, help="Mask directory with 000000.png ...")
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--width", type=int, default=560)
    parser.add_argument("--height", type=int, default=336)
    parser.add_argument("--resize_mode", choices=["center_crop", "resize"], default="center_crop")
    parser.add_argument("--static_scene", action="store_true")
    parser.add_argument("--model_path", default="models")
    parser.add_argument("--reconstructor_path", default="models/NeoVerse/reconstructor.ckpt")
    parser.add_argument("--low_vram", action="store_true")
    parser.add_argument("--alpha_threshold", type=float, default=0.05)
    parser.add_argument("--static_voxel_size", type=float, default=0.01,
                        help="voxel size in neoverse_scene_unit")
    parser.add_argument("--dynamic_voxel_size", type=float, default=0.01,
                        help="voxel size in neoverse_scene_unit")
    parser.add_argument("--dynamic_threshold", type=float, default=1e-4,
                        help="world velocity threshold for static candidate classification")
    parser.add_argument("--dynamic_threshold2", type=float, default=None,
                        help="second world velocity threshold for global motion tracking")
    parser.add_argument("--dynamic_denoise", action="store_true",
                        help="enable optional denoise for exported dynamic points")
    parser.add_argument("--dynamic_denoise_min_neighbors", type=int, default=1,
                        help="minimum occupied 26-neighborhood voxels to keep a dynamic point")
    parser.add_argument("--dynamic_denoise_min_cluster_size", type=int, default=12,
                        help="minimum 26-connected dynamic cluster size to keep")
    parser.add_argument("--dynamic_denoise_temporal_min_frames", type=int, default=1,
                        help="minimum track length (in frames) for temporal dynamic denoise")
    parser.add_argument("--dynamic_denoise_temporal_match_radius", type=float, default=0.06,
                        help="centroid match radius for adjacent-frame temporal dynamic denoise")
    parser.add_argument("--enable_global_motion_tracking", action="store_true",
                        help="verify static candidates by camera reprojection consistency")
    return parser.parse_args()


def main():
    args = parse_args()
    from diffsynth.pipelines.wan_video_neoverse import WanVideoNeoVersePipeline
    from diffsynth.utils.static_dynamic_split import SplitConfig, run_static_dynamic_split

    print(f"Loading NeoVerse reconstructor-only pipeline from {args.reconstructor_path}...")

    pipe = WanVideoNeoVersePipeline.from_reconstructor_only(
        reconstructor_path=args.reconstructor_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.bfloat16,
        enable_vram_management=args.low_vram,
    )

    config = SplitConfig(
        input_path=args.input_path,
        output_dir=args.output_dir,
        split_mode=args.split_mode,
        mask_dir=args.mask_dir,
        num_frames=args.num_frames,
        width=args.width,
        height=args.height,
        resize_mode=args.resize_mode,
        static_scene=args.static_scene,
        alpha_threshold=args.alpha_threshold,
        static_voxel_size=args.static_voxel_size,
        dynamic_voxel_size=args.dynamic_voxel_size,
        dynamic_threshold=args.dynamic_threshold,
        dynamic_threshold2=args.dynamic_threshold2,
        dynamic_denoise_enabled=args.dynamic_denoise,
        dynamic_denoise_min_neighbors=args.dynamic_denoise_min_neighbors,
        dynamic_denoise_min_cluster_size=args.dynamic_denoise_min_cluster_size,
        dynamic_denoise_temporal_min_frames=args.dynamic_denoise_temporal_min_frames,
        dynamic_denoise_temporal_match_radius=args.dynamic_denoise_temporal_match_radius,
        enable_global_motion_tracking=args.enable_global_motion_tracking,
    )

    result = run_static_dynamic_split(pipe=pipe, config=config)
    print("Split export completed")
    print(f"Output: {os.path.abspath(result['output_dir'])}")


if __name__ == "__main__":
    main()
