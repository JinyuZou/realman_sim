#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 HDF5 数据集（episode_*.hdf5）转成 LeRobot 格式并 push 到 HuggingFace Hub
用法：
    python hdf5_to_lerobot.py \
        --in-dir outputs/joycon_hdf5/task1 \
        --repo-id Jinyu220/realman_hdf5_task1 \
        --fps 8 \
        --push
输出结构：
    <root>/<repo-id>/
        data/chunk-000/episode_000000.parquet
        videos/chunk-000/episode_000000.mp4
        meta/chunk-000/episode_000000_meta_info.json
"""

import os, re, argparse, json
from pathlib import Path
import numpy as np
import h5py
import cv2
import torch

# LeRobot 兼容导入
try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
except Exception:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ---------- 参数 ----------
def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, help="HDF5 所在目录，含 episode_*.hdf5")
    ap.add_argument("--repo-id", required=True, help="HuggingFace repo id")
    ap.add_argument("--root", type=str, default="~/datasets/lerobot", help="LeRobot 本地缓存根目录")
    ap.add_argument("--fps", type=int, default=8, help="视频帧率")
    ap.add_argument("--push",default=True, action="store_true", help="转换完成后 push 到 HuggingFace Hub")
    ap.add_argument("--task", type=str, default="hang_task", help="task name saved in each frame (required by some lerobot versions)")

    return ap.parse_args()

# ---------- 工具 ----------
def _sorted_episode_paths(hdf_dir: Path):
    paths = list(hdf_dir.glob("episode_*.hdf5"))
    def key(p: Path):
        m = re.search(r"episode_(\d+)\.hdf5$", p.name)
        return int(m.group(1)) if m else 10**18
    return sorted(paths, key=key)

# ---------- 主逻辑 ----------
def main():
    args = parse()
    in_dir  = Path(args.in_dir).expanduser().resolve()
    root = Path(args.root).expanduser().resolve() / args.repo_id
    #root.parent.mkdir(parents=True, exist_ok=True)   # ✅ 只保证父目录存在

    hdf_paths = _sorted_episode_paths(in_dir)
    if not hdf_paths:
        raise FileNotFoundError(f"No episode_*.hdf5 found in {in_dir}")

    # 用第一个文件探测形状
    with h5py.File(hdf_paths[0], "r") as f:
        n_frames, H, W, C = f["observations/images/cam_high"].shape
        action_dim = f["action"].shape[1]
        qpos_dim   = f["qpos"].shape[1]
        qvel_dim   = f["qvel"].shape[1]
        effort_dim = f["observations/effort"].shape[1]

    # 构造 LeRobot features
    features = {
        "observation.images.cam_high":  {"dtype": "video", "shape": (H, W, 3), "names": ["height", "width", "channel"]},
        "observation.images.cam_left":  {"dtype": "video", "shape": (H, W, 3), "names": ["height", "width", "channel"]},
        "observation.images.cam_low":   {"dtype": "video", "shape": (H, W, 3), "names": ["height", "width", "channel"]},
        "observation.images.cam_right": {"dtype": "video", "shape": (H, W, 3), "names": ["height", "width", "channel"]},
        "observation.state":            {"dtype": "float32", "shape": (qpos_dim + qvel_dim + effort_dim,), "names": None},
        "action":                       {"dtype": "float32", "shape": (action_dim,), "names": None},
        #"task": {"dtype": "string", "shape": (), "names": None},
    }

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=str(root),
        fps=args.fps,
        features=features,
        image_writer_threads=8,
        image_writer_processes=4,
    )

    # 遍历每个 episode
    for ep_idx, hdf_path in enumerate(hdf_paths):
        print(f"[{ep_idx+1}/{len(hdf_paths)}] Converting {hdf_path.name}")
        with h5py.File(hdf_path, "r") as f:
            n = f["action"].shape[0]
            # 预先把所有图像读进内存（快）
            imgs = {
                "cam_high":  f["observations/images/cam_high"][:],   # (T,H,W,3)
                "cam_left":  f["observations/images/cam_left"][:],
                "cam_low":   f["observations/images/cam_low"][:],
                "cam_right": f["observations/images/cam_right"][:],
            }
            qpos   = f["qpos"][:]
            qvel   = f["qvel"][:]
            effort = f["observations/effort"][:]
            action = f["action"][:]

            for t in range(n):
                frame = {}
                # 1) 图像
                for cam in imgs:
                    frame[f"observation.images.{cam}"] = torch.from_numpy(imgs[cam][t].copy())
                # 2) 状态 = qpos + qvel + effort
                state = np.concatenate([
                    qpos[t],
                    qvel[t],
                    effort[t] if effort.size > 0 else np.zeros((0,), np.float32)
                ]).astype(np.float32)
                frame["observation.state"] = torch.from_numpy(state)
                # 3) action
                frame["action"] = torch.from_numpy(action[t].astype(np.float32))
                if args.task is not None:
                    frame["task"] = args.task
                dataset.add_frame(frame)

        dataset.save_episode()
        print(f"  └─ saved & encoded -> {root}/videos/chunk-000/episode_{ep_idx:06d}.mp4")

    # 可选：推送到 Hub
    if args.push:
        print("Pushing to HuggingFace Hub...")
        dataset.push_to_hub(private=False)
        print("Done!")

if __name__ == "__main__":
    main()
