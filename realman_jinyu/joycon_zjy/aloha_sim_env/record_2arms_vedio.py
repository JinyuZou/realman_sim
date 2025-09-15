#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Teleop recorder -> LeRobot v2 dataset
- B键：开始/结束一段episode；开始前自动对齐初始位姿
- 显示4路相机拼图窗口(q退出)
- 输出结构(示例):
  <root>/<task>/
    data/chunk-000/episode_000000.parquet
    data/chunk-000/episode_000001.parquet
    videos/chunk-000/observation.images.wrist_cam_left/episode_000000.mp4
    ...
    meta/info.json
    meta/episodes.jsonl
"""

import os, io, json, time, math, argparse, glob, re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import cv2
import pandas as pd

import gymnasium as gym
import gym_av_aloha
from gym_av_aloha.env.sim_env import AVAlohaEnv
from gym_av_aloha.kinematics.grad_ik import GradIK, GradIKConfig

# 你自己的模块
from TeleopEnv2arms import TeleopEnv2Arms
from xbox_controller import DualJoyConController

# ---------------- 基本配置 ----------------
FPS = 8
CAMERAS = {
    "wrist_cam_left":  [480, 640],
    "wrist_cam_right": [480, 640],
    "overhead_cam":    [480, 640],
    "worms_eye_cam":   [480, 640],
}
# pygame/Xbox 常见映射：A=0, B=1, X=2, Y=3
B_INDEX = 1

INITIAL_FIXED = None   # 固定初始位姿（留空则用启动时姿态）

POS_TOL_M = 1e-2       # 回初始：位置容差 1 cm
ROT_TOL_DEG = 5.0      # 回初始：角度容差 5°
MAX_RETURN_SECS = 2.0  # 回初始：最长 2 秒

# LeRobot v2：固定第一块分片
CHUNK = "chunk-000"

# ---------------- 工具函数 ----------------
def tile_four(imgs: dict):
    """把4路画面拼成2x2用于OpenCV预览；不写盘。"""
    keys = ["wrist_cam_left", "wrist_cam_right", "overhead_cam", "worms_eye_cam"]
    any_img = None
    for v in imgs.values():
        if v is not None:
            any_img = v
            break
    if any_img is None:
        return None
    H, W, C = any_img.shape
    black = np.zeros((H, W, 3), dtype=any_img.dtype)
    tiles = [imgs.get(k, None) for k in keys]
    tiles = [t if t is not None else black for t in tiles]
    top = np.concatenate(tiles[0:2], axis=1)
    bottom = np.concatenate(tiles[2:4], axis=1)
    return np.concatenate([top, bottom], axis=0)

def _pose_from_pos_quat(pos_xyz, quat_wxyz):
    qw, qx, qy, qz = quat_wxyz
    R = np.array([
        [1-2*(qy*qy+qz*qz),   2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),   1-2*(qx*qx+qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),   2*(qy*qz + qx*qw), 1-2*(qx*qx+qy*qy)]
    ], dtype=float)
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3]  = np.array(pos_xyz, dtype=float)
    return T

def _pose_from_env(env):
    lT = env.env.left_arm.get_eef_pose()
    rT = env.env.right_arm.get_eef_pose()
    return lT.copy(), rT.copy()

def _pose_reached(curr_T, target_T, pos_tol=POS_TOL_M, rot_tol_deg=ROT_TOL_DEG):
    dp = np.linalg.norm(curr_T[:3, 3] - target_T[:3, 3])
    R_err = curr_T[:3, :3].T @ target_T[:3, :3]
    cos_theta = (np.trace(R_err) - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.degrees(np.arccos(cos_theta))
    return (dp <= pos_tol) and (theta <= rot_tol_deg)

def _ensure_uint8_rgb(img: np.ndarray) -> np.ndarray:
    """确保帧是uint8 RGB(H,W,3)"""
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    return img

def _scan_next_episode_id(dataset_root: Path) -> int:
    """在 data/chunk-000/ 中扫描现有 episode_XXXXXX.parquet 得到下一个ID"""
    data_dir = dataset_root / "data" / CHUNK
    data_dir.mkdir(parents=True, exist_ok=True)
    ids = []
    for p in data_dir.glob("episode_*.parquet"):
        m = re.match(r"episode_(\d{6})\.parquet", p.name)
        if m:
            ids.append(int(m.group(1)))
    return (max(ids) + 1) if ids else 0

# ---------------- LeRobot v2 写盘器 ----------------
class LeRobotV2Writer:
    """
    把一个episode写成：
      - data/chunk-000/episode_xxxxxx.parquet       (逐步数值，表格)
      - videos/chunk-000/observation.images.<cam>/episode_xxxxxx.mp4
    并维护：
      - meta/info.json
      - meta/episodes.jsonl  (每行一个episode的元信息)
    """
    def __init__(self, root: Path, fps: float, cameras: Dict[str, List[int]], fourcc: str = "mp4v", state_dim: int = 0):
        self.root = Path(root)
        self.fps = float(fps)
        self.cameras = cameras
        self.state_dim = int(state_dim)  # <--- 新增
        self.fourcc = cv2.VideoWriter_fourcc(*fourcc)

        # 目录
        (self.root / "data" / CHUNK).mkdir(parents=True, exist_ok=True)
        for cam in cameras.keys():
            (self.root / "videos" / CHUNK / f"observation.images.{cam}").mkdir(parents=True, exist_ok=True)
        (self.root / "meta").mkdir(parents=True, exist_ok=True)

        # meta路径
        self.info_path = self.root / "meta" / "info.json"
        self.episodes_path = self.root / "meta" / "episodes.jsonl"

        self._init_meta()

        # 当前episode资源
        self._writers: Dict[str, cv2.VideoWriter] = {}
        self._sizes: Dict[str, Tuple[int,int]] = {}
        self._ep_name: str = ""

    def _init_meta(self):
        default_info = {
            "version": "2.0",
            "fps": self.fps,
            "chunks": [CHUNK],
            "modalities": {
                "action": {"dtype":"float32", "shape":[-1, 14]},
                "observation.state": {"dtype": "float32", "shape": [-1, int(self.state_dim)]},
                "timestamp": {"dtype":"float64", "shape":[-1]},
                "is_first": {"dtype":"bool", "shape":[-1]},
                "is_last": {"dtype":"bool", "shape":[-1]},
                "is_terminal": {"dtype":"bool", "shape":[-1]},
                "observation.images": {
                    "cameras": list(self.cameras.keys()),
                    "storage": "videos/mp4-per-episode-per-camera"
                }
            }
        }
        if not self.info_path.exists():
            with open(self.info_path, "w") as f:
                json.dump(default_info, f, indent=2)
        if not self.episodes_path.exists():
            self.episodes_path.touch()

    # ---------- episode 开启/关闭 ----------
    def start_episode(self, ep_name: str):
        self._ep_name = ep_name
        self._writers.clear()
        self._sizes.clear()

    def _open_video(self, cam: str, frame_rgb: np.ndarray):
        h, w = frame_rgb.shape[:2]
        path = self.root / "videos" / CHUNK / f"observation.images.{cam}" / f"{self._ep_name}.mp4"
        wr = cv2.VideoWriter(str(path), self.fourcc, self.fps, (w, h))
        if not wr.isOpened():
            raise RuntimeError(f"无法创建视频文件: {path}")
        self._writers[cam] = wr
        self._sizes[cam] = (w, h)

    def write_frame(self, images_dict: Dict[str, np.ndarray], resize_to_cfg=True):
        """
        一次写一帧（多路相机）。frame必须是RGB uint8。
        如果分辨率不一致，统一resize到CAMERAS配置。
        """
        for cam, frm in images_dict.items():
            if frm is None:
                continue
            frm = _ensure_uint8_rgb(frm)
            if resize_to_cfg:
                H, W = self.cameras[cam]
                if frm.shape[0] != H or frm.shape[1] != W:
                    frm = cv2.resize(frm, (W, H), interpolation=cv2.INTER_AREA)
            if cam not in self._writers:
                self._open_video(cam, frm)
            w, h = self._sizes[cam]
            if (frm.shape[1] != w) or (frm.shape[0] != h):
                frm = cv2.resize(frm, (w, h), interpolation=cv2.INTER_AREA)
            # OpenCV 需要BGR
            self._writers[cam].write(frm[:, :, ::-1])

    def end_episode(self, df_rows: List[dict], length: int, split: str = "train"):
        # 1) 关视频
        for wr in self._writers.values():
            try:
                wr.release()
            except Exception:
                pass
        self._writers.clear()
        self._sizes.clear()

        # 2) parquet（每行一个step）
        parquet_path = self.root / "data" / CHUNK / f"{self._ep_name}.parquet"
        pd.DataFrame(df_rows).to_parquet(parquet_path, index=False)

        # 3) 追加 episodes.jsonl
        with open(self.episodes_path, "a") as f:
            rec = {
                "episode": self._ep_name,
                "chunk": CHUNK,
                "length": int(length),
                "split": split
            }
            f.write(json.dumps(rec) + "\n")

# ---------------- 主程序 ----------------
def main():
    ap = argparse.ArgumentParser(description="Teleop recorder (LeRobot v2 format).")
    ap.add_argument("--env-name",   type=str, default="cube-transfer")
    ap.add_argument("--fps",        type=float, default=FPS)
    ap.add_argument("--out",        type=str, default="outputs")
    ap.add_argument("--task",       type=str, default="task1")
    ap.add_argument("--fourcc",     type=str, default="mp4v", help="cv2 FourCC (mp4v/avc1/xvid)")
    ap.add_argument("--no_resize",  action="store_true", help="不把相机帧统一resize到CAMERAS配置")
    args = ap.parse_args()

    env = TeleopEnv2Arms(env_name=args.env_name, fps=args.fps, cameras=CAMERAS)
    joy = DualJoyConController(env)
    obs, info = env.reset()

    # 初始位姿
    if INITIAL_FIXED is not None:
        initial_L = _pose_from_pos_quat(INITIAL_FIXED["left"]["pos"],
                                        INITIAL_FIXED["left"]["quat_wxyz"])
        initial_R = _pose_from_pos_quat(INITIAL_FIXED["right"]["pos"],
                                        INITIAL_FIXED["right"]["quat_wxyz"])
    else:
        initial_L, initial_R = _pose_from_env(env)

    # 数据集根路径
    dataset_root = Path(args.out) / args.task
    dataset_root.mkdir(parents=True, exist_ok=True)
    print(f"[Recorder] 数据集将保存到: {dataset_root.resolve()}")
    print("操作：按 B 开始 → 先回初始位姿 → 自动开始录制；再按 B 结束并保存；窗口按 q 退出。")

    # 运行状态
    recording = False
    returning = False
    return_start_t = 0.0
    prev_B = False

    # —— 缓存（每次开始录制时重置）——
    df_rows: List[dict] = []
    step_idx = 0

    # Writer
    writer = LeRobotV2Writer(root=dataset_root, fps=args.fps, cameras=CAMERAS, fourcc=args.fourcc,state_dim=14)

    # OpenCV窗口
    win_name = "Teleop Recorder (4 cams)"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            loop_t0 = time.time()

            # 手柄
            running_flag, left_pose, left_grip, right_pose, right_grip, _ = joy.poll()

            # B键上升沿：开始/结束
            btn_B = joy.joy.get_button(B_INDEX)
            if btn_B and not prev_B:
                if recording:
                    # ---- 结束：补最后一行的标记并落盘 ----
                    if len(df_rows) > 0:
                        df_rows[-1]["is_last"] = True
                        df_rows[-1]["is_terminal"] = True
                    writer.end_episode(df_rows, length=len(df_rows), split="train")
                    print(f"[Recorder] ■ 结束并保存 -> {df_rows[0]['episode'] if df_rows else 'EPISODE'}，共 {len(df_rows)} 帧")
                    recording = False
                    df_rows.clear()
                    step_idx = 0
                else:
                    # ---- 准备开始：先回初始位姿 ----
                    returning = True
                    return_start_t = loop_t0
                    print("[Recorder] 对齐到初始位姿中……")
            prev_B = btn_B

            # 一步控制
            if returning:
                obs, _, _, _, step_info = env.step(
                    left_pose=initial_L, left_gripper=1.0,
                    right_pose=initial_R, right_gripper=1.0
                )
                l_now, r_now = step_info["left_arm_pose"], step_info["right_arm_pose"]
                reached_L = _pose_reached(l_now, initial_L)
                reached_R = _pose_reached(r_now, initial_R)
                if (reached_L and reached_R) or ((loop_t0 - return_start_t) > MAX_RETURN_SECS):
                    # 对齐手柄零点
                    if hasattr(joy, "recenter_to"):
                        joy.recenter_to(initial_L, initial_R)
                    returning = False

                    # —— 开启新episode ——：计算新ID并start
                    next_id = _scan_next_episode_id(dataset_root)
                    ep_name = f"episode_{next_id:06d}"
                    writer.start_episode(ep_name)
                    df_rows.clear()
                    step_idx = 0
                    recording = True
                    print(f"[Recorder] ▶ 开始录制 -> {ep_name}")
            else:
                # 正常遥操作
                obs, _, _, _, step_info = env.step(
                    left_pose=left_pose, left_gripper=left_grip,
                    right_pose=right_pose, right_gripper=right_grip
                )

            # 显示
            pix = obs.get("pixels", {})
            tiled = tile_four(pix) if pix else None
            if tiled is not None:
                cv2.imshow(win_name, tiled[:, :, ::-1])  # RGB->BGR
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[Recorder] q 被按下，退出。")
                    break

            # 录制：写一帧
            if recording:
                # 写视频帧（四路）
                images_for_write = {}
                for cam in CAMERAS.keys():
                    frm = pix.get(cam, None)
                    if frm is None:
                        continue
                    images_for_write[cam] = _ensure_uint8_rgb(frm)
                writer.write_frame(images_dict=images_for_write, resize_to_cfg=(not args.no_resize))

                # 准备状态（agent_pos可无）
                agent_pos = obs.get("agent_pos", None)
                state_vec = np.zeros((0,), dtype=np.float32) if agent_pos is None else agent_pos.astype(np.float32)

                # 一行数据（注意：DataFrame将把ndarray列存为Arrow数组）
                row = {
                    "episode": ep_name,                         # 便于检索（可选列）
                    "timestamp": float(time.time()),            # float64
                    "is_first": (step_idx == 0),                # bool
                    "is_last": False,                           # 结束时再置True
                    "is_terminal": False,                       # 结束时再置True
                    "action": step_info["action"].astype(np.float32),     # float32[14]
                    "observation.state": state_vec.astype(np.float32),                        # float32[state_dim] 或空(0,)
                }
                df_rows.append(row)
                step_idx += 1

            # 控制频率
            elapsed = time.time() - loop_t0
            target_dt = 1.0 / float(args.fps)
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)

    except KeyboardInterrupt:
        print("\n[Recorder] 收到 Ctrl+C，安全退出。")
    finally:
        try:
            # 如果正在录制，尽量落盘收尾
            if recording and len(df_rows) > 0:
                df_rows[-1]["is_last"] = True
                df_rows[-1]["is_terminal"] = True
                writer.end_episode(df_rows, length=len(df_rows), split="train")
        except Exception as e:
            print(f"[Recorder] 结束时保存失败：{e}")
        try:
            joy.close()
        except Exception:
            pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
