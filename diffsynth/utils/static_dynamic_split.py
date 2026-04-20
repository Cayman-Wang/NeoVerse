import csv
import contextlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F

from .auxiliary import center_crop, homo_matrix_inverse, load_video, pixel_to_world_coords
from ..auxiliary_models.worldmirror.models.models.rasterization import Gaussians


STATIC_COLOR = np.array([160, 160, 160], dtype=np.float32)
DYNAMIC_COLOR = np.array([255, 140, 0], dtype=np.float32)


@dataclass
class SplitConfig:
    input_path: str
    output_dir: str
    split_mode: str = "auto"
    mask_dir: Optional[str] = None
    num_frames: int = 81
    width: int = 560
    height: int = 336
    resize_mode: str = "center_crop"
    static_scene: bool = False
    alpha_threshold: float = 0.05
    static_voxel_size: float = 0.01
    dynamic_voxel_size: float = 0.01


@dataclass
class SplitOutputs:
    effective_mode: str
    sampled_num_frames: int
    static_points: np.ndarray
    dynamic_points_per_frame: List[np.ndarray]
    preview_path: str
    gaussian_export_mode: str
    fallback_reason: Optional[str]


def build_split_views(images: Sequence[Image.Image], device: str, static_scene: bool) -> dict:
    return _build_views(images=images, device=device, static_scene=static_scene)


def run_split_reconstruction(pipe, views: dict, use_motion: bool = True) -> dict:
    return _run_reconstruction(pipe=pipe, views=views, use_motion=use_motion)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _to_numpy_points(points: Sequence[np.ndarray]) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(points, axis=0).astype(np.float32)


def _voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if points.size == 0:
        return points.astype(np.float32)
    if voxel_size <= 0:
        return points.astype(np.float32)
    coords = np.floor(points / voxel_size).astype(np.int64)
    _, unique_idx = np.unique(coords, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)
    return points[unique_idx].astype(np.float32)


def _base_gaussian_mask(gaussian: Gaussians) -> torch.Tensor:
    mask = gaussian.opacities > 0.05
    if gaussian.confidences is not None:
        mask = mask & (gaussian.confidences > 0)
    return mask


def _materialize_splats_for_timestamp(splats: Sequence[Gaussians], timestamp: int) -> List[Gaussians]:
    result: List[Gaussians] = []
    for gs in splats:
        transitioned = gs.transition(timestamp, mask=_base_gaussian_mask(gs))
        if transitioned.means.shape[0] > 0:
            transitioned.timestamp = timestamp
            result.append(transitioned)
    return result


def _extract_means_from_splats(splats: Sequence[Gaussians]) -> np.ndarray:
    chunks = []
    for gs in splats:
        mask = _base_gaussian_mask(gs)
        if mask.any():
            chunks.append(gs.means[mask].detach().cpu().float().numpy())
    return _to_numpy_points(chunks)


def _build_views(images: Sequence[Image.Image], device: str, static_scene: bool) -> dict:
    S = len(images)
    views = {
        "img": torch.stack([F.to_tensor(image)[None] for image in images], dim=1).to(device),
        "is_target": torch.zeros((1, S), dtype=torch.bool, device=device),
    }
    if static_scene:
        views["is_static"] = torch.ones((1, S), dtype=torch.bool, device=device)
        views["timestamp"] = torch.zeros((1, S), dtype=torch.int64, device=device)
    else:
        views["is_static"] = torch.zeros((1, S), dtype=torch.bool, device=device)
        views["timestamp"] = torch.arange(0, S, dtype=torch.int64, device=device).unsqueeze(0)
    return views


def _run_reconstruction(pipe, views: dict, use_motion: bool) -> dict:
    if pipe.vram_management_enabled:
        pipe.reconstructor.to(pipe.device)
    if str(pipe.device).startswith("cuda"):
        autocast_ctx = torch.amp.autocast("cuda", dtype=pipe.torch_dtype)
    else:
        autocast_ctx = contextlib.nullcontext()
    with autocast_ctx:
        predictions = pipe.reconstructor(views, is_inference=True, use_motion=use_motion)
    if pipe.vram_management_enabled:
        pipe.reconstructor.cpu()
        torch.cuda.empty_cache()
    return predictions


def _process_mask_image(mask_image: Image.Image, resize_mode: str, resolution: Tuple[int, int]) -> np.ndarray:
    if resize_mode == "resize":
        mask_image = mask_image.resize(resolution, resample=Image.NEAREST)
    else:
        mask_image = center_crop(mask_image, resolution)
    arr = np.array(mask_image.convert("L"), dtype=np.uint8)
    return arr > 127


def _validate_mask_dir(mask_dir: str, sampled_num_frames: int) -> Tuple[bool, Optional[str], Optional[List[Path]]]:
    mask_root = Path(mask_dir)
    if not mask_root.exists() or not mask_root.is_dir():
        return False, f"mask_dir '{mask_dir}' does not exist or is not a directory", None

    expected = [mask_root / f"{i:06d}.png" for i in range(sampled_num_frames)]
    missing = [str(p.name) for p in expected if not p.exists()]
    all_png = sorted(mask_root.glob("*.png"))
    if missing:
        return False, f"missing masks: {', '.join(missing[:5])}", None
    if len(all_png) != sampled_num_frames:
        return False, (
            f"mask count mismatch: expected {sampled_num_frames}, found {len(all_png)}"
        ), None
    return True, None, expected


def _resolve_split_mode(requested_mode: str, mask_dir: Optional[str], sampled_num_frames: int) -> Tuple[str, Optional[List[Path]], Optional[str]]:
    if requested_mode == "motion":
        return "motion", None, None

    has_mask_dir = bool(mask_dir)
    if not has_mask_dir:
        if requested_mode == "mask":
            raise ValueError("split_mode=mask requires --mask_dir")
        return "motion", None, "mask_dir is not provided"

    valid, reason, ordered_masks = _validate_mask_dir(mask_dir, sampled_num_frames)
    if requested_mode == "mask":
        if not valid:
            raise ValueError(f"invalid mask_dir for mask mode: {reason}")
        return "mask", ordered_masks, None

    if valid:
        return "mask", ordered_masks, None
    return "motion", None, reason


def _project_points(points: np.ndarray, K: np.ndarray, world2cam: np.ndarray, width: int, height: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if points.shape[0] == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32)

    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    points_h = np.concatenate([points.astype(np.float32), ones], axis=1)
    cam = (world2cam @ points_h.T).T[:, :3]
    z = cam[:, 2]
    valid_z = z > 1e-6
    cam = cam[valid_z]
    z = z[valid_z]
    if cam.shape[0] == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = np.round((cam[:, 0] * fx / z) + cx).astype(np.int32)
    v = np.round((cam[:, 1] * fy / z) + cy).astype(np.int32)
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return u[inside], v[inside], z[inside]


def _overlay_points(frame: Image.Image, static_points: np.ndarray, dynamic_points: np.ndarray, K: np.ndarray, world2cam: np.ndarray) -> Image.Image:
    arr = np.array(frame.convert("RGB"), dtype=np.float32)
    h, w = arr.shape[:2]

    u_s, v_s, _ = _project_points(static_points, K, world2cam, w, h)
    if u_s.size > 0:
        arr[v_s, u_s] = arr[v_s, u_s] * 0.4 + STATIC_COLOR * 0.6

    u_d, v_d, _ = _project_points(dynamic_points, K, world2cam, w, h)
    if u_d.size > 0:
        arr[v_d, u_d] = arr[v_d, u_d] * 0.2 + DYNAMIC_COLOR * 0.8

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _save_points_index(index_path: Path, rows: List[dict]) -> None:
    _ensure_dir(index_path.parent)
    with open(index_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame_idx",
                "timestamp",
                "dynamic_points_rel",
                "dynamic_gaussians_rel",
                "num_dynamic_points",
                "split_mode_effective",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _save_numpy(path: Path, points: np.ndarray) -> None:
    _ensure_dir(path.parent)
    np.save(path, points.astype(np.float32))


def _save_torch(path: Path, obj) -> None:
    _ensure_dir(path.parent)
    torch.save(obj, path)


def _detach_cpu_tensor(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return value


def _clone_splats_to_cpu(splats: Sequence[Gaussians]) -> List[Gaussians]:
    cloned: List[Gaussians] = []
    for gs in splats:
        cloned.append(
            Gaussians(
                means=gs.means.detach().cpu(),
                harmonics=gs.harmonics.detach().cpu(),
                opacities=gs.opacities.detach().cpu(),
                scales=gs.scales.detach().cpu(),
                rotations=gs.rotations.detach().cpu(),
                confidences=gs.confidences.detach().cpu() if gs.confidences is not None else None,
                timestamp=int(gs.timestamp),
                life_span=_detach_cpu_tensor(getattr(gs, "life_span", 1.0)),
                life_span_gamma=float(getattr(gs, "life_span_gamma", 0.0)),
                forward_timestamp=getattr(gs, "forward_timestamp", None),
                forward_vel=_detach_cpu_tensor(getattr(gs, "forward_vel", None)),
                forward_scales=_detach_cpu_tensor(getattr(gs, "forward_scales", None)),
                forward_rotations=_detach_cpu_tensor(getattr(gs, "forward_rotations", None)),
                backward_timestamp=getattr(gs, "backward_timestamp", None),
                backward_vel=_detach_cpu_tensor(getattr(gs, "backward_vel", None)),
                backward_scales=_detach_cpu_tensor(getattr(gs, "backward_scales", None)),
                backward_rotations=_detach_cpu_tensor(getattr(gs, "backward_rotations", None)),
            )
        )
    return cloned


def _render_depth_alpha(pipe, splats: Sequence[Gaussians], K: torch.Tensor, cam2world: torch.Tensor, timestamp: int, width: int, height: int):
    world2cam = homo_matrix_inverse(cam2world[None])[0]
    rgb, depth, alpha = pipe.reconstructor.gs_renderer.rasterizer.forward(
        [list(splats)],
        [world2cam[None]],
        [K[None]],
        [torch.tensor([timestamp], device=world2cam.device, dtype=torch.int64)],
        sh_degree=0,
        width=width,
        height=height,
    )
    return rgb[0, 0], depth[0, 0, :, :, 0], alpha[0, 0, :, :, 0], world2cam


def _split_motion(
    splats: Sequence[Gaussians],
    frame_timestamps: List[int],
    static_voxel_size: float,
    dynamic_voxel_size: float,
) -> Tuple[np.ndarray, List[np.ndarray], List[dict], List[Gaussians], List[Gaussians]]:
    static_splats = [gs for gs in splats if gs.timestamp == -1]
    dynamic_splats = [gs for gs in splats if gs.timestamp >= 0]

    static_points = _voxel_downsample(_extract_means_from_splats(static_splats), voxel_size=static_voxel_size)

    dynamic_points_per_frame: List[np.ndarray] = []
    index_rows: List[dict] = []
    for frame_idx, timestamp in enumerate(frame_timestamps):
        transitioned = _materialize_splats_for_timestamp(dynamic_splats, timestamp)
        points = _voxel_downsample(_extract_means_from_splats(transitioned), voxel_size=dynamic_voxel_size)
        dynamic_points_per_frame.append(points)
        index_rows.append(
            {
                "frame_idx": frame_idx,
                "timestamp": timestamp,
                "dynamic_points_rel": f"points/dynamic_by_timestamp/{frame_idx:06d}.npy",
                "dynamic_gaussians_rel": "gaussians/dynamic.pt",
                "num_dynamic_points": int(points.shape[0]),
                "split_mode_effective": "motion",
            }
        )
    return static_points, dynamic_points_per_frame, index_rows, static_splats, dynamic_splats


def _filter_snapshot_by_mask_and_depth(
    gaussian: Gaussians,
    K: np.ndarray,
    world2cam: np.ndarray,
    depth_map: np.ndarray,
    mask_dynamic: np.ndarray,
) -> Optional[Gaussians]:
    means = gaussian.means.detach().cpu().float().numpy()
    if means.shape[0] == 0:
        return None

    h, w = depth_map.shape
    u, v, z = _project_points(means, K, world2cam, w, h)
    if u.size == 0:
        return None

    sampled_depth = depth_map[v, u]
    keep = mask_dynamic[v, u] & (sampled_depth > 0) & (np.abs(sampled_depth - z) <= (0.05 + 0.02 * sampled_depth))
    if not np.any(keep):
        return None

    full_keep = np.zeros(means.shape[0], dtype=bool)
    ones = np.ones((means.shape[0], 1), dtype=np.float32)
    pts_h = np.concatenate([means.astype(np.float32), ones], axis=1)
    cam = (world2cam @ pts_h.T).T[:, :3]
    valid_z = cam[:, 2] > 1e-6
    if not np.any(valid_z):
        return None

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    uu = np.round((cam[valid_z, 0] * fx / cam[valid_z, 2]) + cx).astype(np.int32)
    vv = np.round((cam[valid_z, 1] * fy / cam[valid_z, 2]) + cy).astype(np.int32)
    inside = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)

    valid_indices = np.where(valid_z)[0][inside]
    if valid_indices.size == 0:
        return None

    z_inside = cam[valid_z, 2][inside]
    sampled_depth_inside = depth_map[vv[inside], uu[inside]]
    keep_inside = mask_dynamic[vv[inside], uu[inside]] & (sampled_depth_inside > 0) & (
        np.abs(sampled_depth_inside - z_inside) <= (0.05 + 0.02 * sampled_depth_inside)
    )
    full_keep[valid_indices[keep_inside]] = True
    if not np.any(full_keep):
        return None

    keep_t = torch.from_numpy(full_keep).to(gaussian.means.device)
    return Gaussians(
        means=gaussian.means[keep_t],
        harmonics=gaussian.harmonics[keep_t],
        opacities=gaussian.opacities[keep_t],
        scales=gaussian.scales[keep_t],
        rotations=gaussian.rotations[keep_t],
        confidences=gaussian.confidences[keep_t] if gaussian.confidences is not None else None,
        timestamp=gaussian.timestamp,
    )


def _split_mask(
    pipe,
    splats: Sequence[Gaussians],
    images: Sequence[Image.Image],
    masks: Sequence[Path],
    frame_timestamps: List[int],
    intrinsics: torch.Tensor,
    cam2worlds: torch.Tensor,
    alpha_threshold: float,
    static_voxel_size: float,
    dynamic_voxel_size: float,
    resize_mode: str,
    width: int,
    height: int,
) -> Tuple[np.ndarray, List[np.ndarray], List[List[Gaussians]], List[dict], List[Gaussians]]:
    static_points_acc: List[np.ndarray] = []
    dynamic_points_per_frame: List[np.ndarray] = []
    dynamic_gaussians_per_frame: List[List[Gaussians]] = []
    index_rows: List[dict] = []

    for frame_idx, timestamp in enumerate(frame_timestamps):
        mask_bin = _process_mask_image(Image.open(masks[frame_idx]), resize_mode=resize_mode, resolution=(width, height))

        _, depth_map_t, alpha_map_t, world2cam_t = _render_depth_alpha(
            pipe=pipe,
            splats=splats,
            K=intrinsics[frame_idx],
            cam2world=cam2worlds[frame_idx],
            timestamp=timestamp,
            width=width,
            height=height,
        )
        depth_map = depth_map_t.detach().cpu().float().numpy()
        alpha_map = alpha_map_t.detach().cpu().float().numpy()
        valid = (alpha_map > alpha_threshold) & (depth_map > 0)

        ys, xs = np.where(valid)
        if ys.size > 0:
            depths = torch.from_numpy(depth_map[ys, xs]).to(intrinsics.device, dtype=intrinsics.dtype)
            pixel_x = torch.from_numpy(xs).to(intrinsics.device, dtype=intrinsics.dtype)
            pixel_y = torch.from_numpy(ys).to(intrinsics.device, dtype=intrinsics.dtype)
            world_points_t = pixel_to_world_coords(pixel_x, pixel_y, depths, intrinsics[frame_idx], world2cam_t)
            world_points = world_points_t.detach().cpu().float().numpy()

            dyn_select = mask_bin[ys, xs]
            dyn_points = world_points[dyn_select]
            st_points = world_points[~dyn_select]
        else:
            dyn_points = np.zeros((0, 3), dtype=np.float32)
            st_points = np.zeros((0, 3), dtype=np.float32)

        dyn_points = _voxel_downsample(dyn_points.astype(np.float32), voxel_size=dynamic_voxel_size)
        dynamic_points_per_frame.append(dyn_points)
        if st_points.shape[0] > 0:
            static_points_acc.append(st_points.astype(np.float32))

        transitioned = _materialize_splats_for_timestamp(splats, timestamp)
        k_np = intrinsics[frame_idx].detach().cpu().float().numpy()
        w2c_np = world2cam_t.detach().cpu().float().numpy()
        dyn_snapshots: List[Gaussians] = []
        for gs in transitioned:
            filtered = _filter_snapshot_by_mask_and_depth(gs, k_np, w2c_np, depth_map, mask_bin)
            if filtered is not None and filtered.means.shape[0] > 0:
                filtered.timestamp = timestamp
                dyn_snapshots.append(filtered)
        dynamic_gaussians_per_frame.append(dyn_snapshots)

        index_rows.append(
            {
                "frame_idx": frame_idx,
                "timestamp": timestamp,
                "dynamic_points_rel": f"points/dynamic_by_timestamp/{frame_idx:06d}.npy",
                "dynamic_gaussians_rel": f"gaussians/dynamic_by_timestamp/{frame_idx:06d}.pt",
                "num_dynamic_points": int(dyn_points.shape[0]),
                "split_mode_effective": "mask",
            }
        )

    static_points = _voxel_downsample(_to_numpy_points(static_points_acc), voxel_size=static_voxel_size)
    static_candidates = [gs for gs in splats if gs.timestamp == -1]
    if not static_candidates:
        static_candidates = list(splats)

    return static_points, dynamic_points_per_frame, dynamic_gaussians_per_frame, index_rows, static_candidates


def _render_preview(
    images: Sequence[Image.Image],
    static_points: np.ndarray,
    dynamic_points_per_frame: Sequence[np.ndarray],
    intrinsics: torch.Tensor,
    cam2worlds: torch.Tensor,
    output_preview_path: Path,
):
    preview_frames: List[Image.Image] = []
    for idx, image in enumerate(images):
        K = intrinsics[idx].detach().cpu().float().numpy()
        world2cam = homo_matrix_inverse(cam2worlds[idx][None])[0].detach().cpu().float().numpy()
        dynamic_points = dynamic_points_per_frame[idx] if idx < len(dynamic_points_per_frame) else np.zeros((0, 3), dtype=np.float32)
        preview_frames.append(_overlay_points(image, static_points, dynamic_points, K, world2cam))

    _ensure_dir(output_preview_path.parent)
    if len(preview_frames) == 1:
        preview_frames[0].save(output_preview_path.with_suffix(".png"))
        return str(output_preview_path.with_suffix(".png"))

    from .. import save_video

    save_video(preview_frames, str(output_preview_path.with_suffix(".mp4")), fps=16)
    return str(output_preview_path.with_suffix(".mp4"))


@torch.no_grad()
def run_static_dynamic_split(
    pipe,
    config: SplitConfig,
    images: Optional[Sequence[Image.Image]] = None,
    views: Optional[dict] = None,
    predictions: Optional[dict] = None,
) -> dict:
    output_root = Path(config.output_dir)
    points_root = output_root / "points"
    dyn_points_root = points_root / "dynamic_by_timestamp"
    gaussians_root = output_root / "gaussians"
    dyn_gaussian_root = gaussians_root / "dynamic_by_timestamp"

    _ensure_dir(points_root)
    _ensure_dir(dyn_points_root)
    _ensure_dir(gaussians_root)

    if images is None:
        images = load_video(
            config.input_path,
            config.num_frames,
            resolution=(config.width, config.height),
            resize_mode=config.resize_mode,
            static_scene=config.static_scene,
        )
    sampled_num_frames = len(images)
    if sampled_num_frames == 0:
        raise ValueError("No sampled frames found from input")

    if predictions is None:
        if views is None:
            views = _build_views(images, device=pipe.device, static_scene=config.static_scene)
        predictions = _run_reconstruction(pipe, views, use_motion=True)

    splats = predictions["splats"][0]
    intrinsics = predictions["rendered_intrinsics"][0]
    cam2worlds = predictions["rendered_extrinsics"][0]
    rendered_timestamps = predictions["rendered_timestamps"][0]
    frame_timestamps = [int(t.item()) for t in rendered_timestamps]

    effective_mode, ordered_masks, fallback_reason = _resolve_split_mode(
        requested_mode=config.split_mode,
        mask_dir=config.mask_dir,
        sampled_num_frames=sampled_num_frames,
    )

    if effective_mode == "motion":
        static_points, dynamic_points_per_frame, index_rows, static_splats, dynamic_splats = _split_motion(
            splats=splats,
            frame_timestamps=frame_timestamps,
            static_voxel_size=config.static_voxel_size,
            dynamic_voxel_size=config.dynamic_voxel_size,
        )
        _save_torch(gaussians_root / "static.pt", static_splats)
        _save_torch(gaussians_root / "dynamic.pt", dynamic_splats)
        gaussian_export_mode = "motion_raw_dynamic"
    else:
        (
            static_points,
            dynamic_points_per_frame,
            dynamic_gaussians_per_frame,
            index_rows,
            static_candidates,
        ) = _split_mask(
            pipe=pipe,
            splats=splats,
            images=images,
            masks=ordered_masks or [],
            frame_timestamps=frame_timestamps,
            intrinsics=intrinsics,
            cam2worlds=cam2worlds,
            alpha_threshold=config.alpha_threshold,
            static_voxel_size=config.static_voxel_size,
            dynamic_voxel_size=config.dynamic_voxel_size,
            resize_mode=config.resize_mode,
            width=config.width,
            height=config.height,
        )
        _save_torch(gaussians_root / "static.pt", static_candidates)
        _ensure_dir(dyn_gaussian_root)
        for frame_idx, dynamic_snapshot in enumerate(dynamic_gaussians_per_frame):
            _save_torch(dyn_gaussian_root / f"{frame_idx:06d}.pt", dynamic_snapshot)
        gaussian_export_mode = "mask_snapshot_audit"

    full_bundle = {
        "splats": _clone_splats_to_cpu(splats),
        "rendered_intrinsics": predictions["rendered_intrinsics"].detach().cpu()
        if isinstance(predictions.get("rendered_intrinsics"), torch.Tensor)
        else predictions.get("rendered_intrinsics"),
        "rendered_extrinsics": predictions["rendered_extrinsics"].detach().cpu()
        if isinstance(predictions.get("rendered_extrinsics"), torch.Tensor)
        else predictions.get("rendered_extrinsics"),
        "rendered_timestamps": predictions["rendered_timestamps"].detach().cpu()
        if isinstance(predictions.get("rendered_timestamps"), torch.Tensor)
        else predictions.get("rendered_timestamps"),
        "input_path": config.input_path,
        "sampled_num_frames": sampled_num_frames,
        "static_scene": config.static_scene,
        "width": config.width,
        "height": config.height,
        "resize_mode": config.resize_mode,
        "use_motion": True,
        "split_mode_requested": config.split_mode,
        "split_mode_effective": effective_mode,
    }
    _save_torch(gaussians_root / "full.pt", full_bundle)

    _save_numpy(points_root / "static_world.npy", static_points)
    for frame_idx, points in enumerate(dynamic_points_per_frame):
        _save_numpy(dyn_points_root / f"{frame_idx:06d}.npy", points)

    _save_points_index(points_root / "index.csv", index_rows)

    preview_path = _render_preview(
        images=images,
        static_points=static_points,
        dynamic_points_per_frame=dynamic_points_per_frame,
        intrinsics=intrinsics,
        cam2worlds=cam2worlds,
        output_preview_path=output_root / "preview" / "overlay",
    )

    nonempty_dynamic_frames = sum(int(points.shape[0] > 0) for points in dynamic_points_per_frame)
    meta = {
        "input_path": config.input_path,
        "sampled_num_frames": sampled_num_frames,
        "num_output_timestamps": len(frame_timestamps),
        "split_mode_requested": config.split_mode,
        "split_mode_effective": effective_mode,
        "has_mask_dir": bool(config.mask_dir),
        "use_motion": True,
        "static_point_count": int(static_points.shape[0]),
        "nonempty_dynamic_frames": int(nonempty_dynamic_frames),
        "gaussian_export_mode": gaussian_export_mode,
        "coordinate_unit": "neoverse_scene_unit",
        "preview_path": os.path.relpath(preview_path, start=str(output_root)),
    }
    if fallback_reason is not None:
        meta["split_mode_fallback_reason"] = fallback_reason

    with open(output_root / "split_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return {
        "meta": meta,
        "output_dir": str(output_root),
    }
