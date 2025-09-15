#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, time, argparse
from pathlib import Path
import numpy as np
import imageio.v3 as iio
import cv2
import h5py

import gymnasium as gym
import realman_jinyu

from xbox_controller import DualJoyConController
from TeleopEnv2arms import TeleopEnv2Arms

# ---------------- 基本配置 ----------------
FPS = 8
CAMERAS = {
    "top_cam":         [480, 640],   # -> cam_high
    "button_cam":      [480, 640],   # -> cam_low
    "wrist_cam_left":  [480, 640],   # -> cam_left
    "wrist_cam_right": [480, 640],   # -> cam_right
}
B_INDEX = 1  # 手柄B键

INITIAL_FIXED = None
POS_TOL_M = 1e-2
ROT_TOL_DEG = 5.0
MAX_RETURN_SECS = 2.0

# ---------- 小工具 ----------
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

def next_episode_path(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    idx = 0
    while True:
        p = root / f"episode_{idx:04d}.hdf5"
        if not p.exists():
            return p
        idx += 1

# ---------- 取 MuJoCo 低层 ----------
def _get_physics(env_like):
    for cand in [getattr(env_like, "env", None),
                 getattr(getattr(env_like, "env", None), "env", None),
                 env_like]:
        if cand is None: continue
        physics = getattr(cand, "physics", None)
        if physics is not None:
            return physics
    return None

def _read_qpos_qvel_effort(env_like):
    physics = _get_physics(env_like)
    if physics is None:  # 兜底
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    qpos = np.asarray(physics.data.qpos, dtype=np.float32).copy()
    qvel = np.asarray(physics.data.qvel, dtype=np.float32).copy()
    # effort（关节力/力矩），dm_control 里常放在 actuator_force 或 qfrc_actuator
    effort = None
    if hasattr(physics.data, "actuator_force"):
        effort = np.asarray(physics.data.actuator_force, dtype=np.float32).copy()
    elif hasattr(physics.data, "qfrc_actuator"):
        effort = np.asarray(physics.data.qfrc_actuator, dtype=np.float32).copy()
    else:
        effort = np.zeros((0,), dtype=np.float32)
    return qpos, qvel, effort

# ---------- HDF5 追加写 ----------
class H5Writer:
    def __init__(self, path: Path, fps: float, img_hw=(480, 640)):
        self.f = h5py.File(str(path), "w")
        self.f.attrs["fps"] = float(fps)
        self.step = 0
        H, W = img_hw
        # 先探一次尺寸（qpos/qvel/effort 会在首帧用实际尺寸建）
        # images
        g_obs = self.f.create_group("observations")
        g_img = g_obs.create_group("images")
        self.ds_cam = {
            "cam_high":  g_img.create_dataset("cam_high",  (0, H, W, 3), maxshape=(None, H, W, 3),
                                              dtype="uint8", chunks=(1, H, W, 3), compression="gzip"),
            "cam_left":  g_img.create_dataset("cam_left",  (0, H, W, 3), maxshape=(None, H, W, 3),
                                              dtype="uint8", chunks=(1, H, W, 3), compression="gzip"),
            "cam_low":   g_img.create_dataset("cam_low",   (0, H, W, 3), maxshape=(None, H, W, 3),
                                              dtype="uint8", chunks=(1, H, W, 3), compression="gzip"),
            "cam_right": g_img.create_dataset("cam_right", (0, H, W, 3), maxshape=(None, H, W, 3),
                                              dtype="uint8", chunks=(1, H, W, 3), compression="gzip"),
        }
        # 先占位，首帧再按真实维度重建
        self.ds_action = self.f.create_dataset("action", (0, 14), maxshape=(None, 14),
                                               dtype="float32", chunks=True, compression="gzip")
        self.ds_qpos = None
        self.ds_qvel = None
        self.ds_effort = None

    def _ensure_dyn(self, qpos, qvel, effort):
        if self.ds_qpos is None:
            self.ds_qpos = self.f.create_dataset("qpos", (0, qpos.shape[0]),
                                                 maxshape=(None, qpos.shape[0]),
                                                 dtype="float32", chunks=True, compression="gzip")
        if self.ds_qvel is None:
            self.ds_qvel = self.f.create_dataset("qvel", (0, qvel.shape[0]),
                                                 maxshape=(None, qvel.shape[0]),
                                                 dtype="float32", chunks=True, compression="gzip")
        if self.ds_effort is None:
            g_obs = self.f["observations"]
            dim = effort.shape[0] if effort is not None and effort.size > 0 else 0
            self.ds_effort = g_obs.create_dataset("effort", (0, dim),
                                                  maxshape=(None, dim),
                                                  dtype="float32", chunks=True, compression="gzip")

    def append(self, imgs_dict, action, qpos, qvel, effort):
        # 首帧建好动态维度
        self._ensure_dyn(qpos, qvel, effort if effort is not None else np.zeros((0,), np.float32))

        t = self.step
        # 1) images
        mapping = {
            "cam_high":  imgs_dict.get("top_cam", None),
            "cam_left":  imgs_dict.get("wrist_cam_left", None),
            "cam_low":   imgs_dict.get("button_cam", None),
            "cam_right": imgs_dict.get("wrist_cam_right", None),
        }
        for k, img in mapping.items():
            if img is None:
                continue
            # 扩展长度并写入
            ds = self.ds_cam[k]
            ds.resize((t+1,)+ds.shape[1:])
            ds[t, ...] = img

        # 2) action
        self.ds_action.resize((t+1, 14))
        self.ds_action[t, :] = action.astype(np.float32)

        # 3) qpos / qvel / effort
        self.ds_qpos.resize((t+1, self.ds_qpos.shape[1]))
        self.ds_qpos[t, :] = qpos.astype(np.float32)

        self.ds_qvel.resize((t+1, self.ds_qvel.shape[1]))
        self.ds_qvel[t, :] = qvel.astype(np.float32)

        if self.ds_effort.shape[1] == 0:
            # 没有努力度信息也保持长度一致
            self.ds_effort.resize((t+1, 0))
        else:
            self.ds_effort.resize((t+1, self.ds_effort.shape[1]))
            self.ds_effort[t, :] = (effort if effort is not None else np.zeros((self.ds_effort.shape[1],), np.float32)).astype(np.float32)

        self.step += 1

    def close(self):
        if self.f:
            self.f.flush()
            self.f.close()
            self.f = None

# ---------- 环境 / 控制 ----------
def _create_env(args):
    env = TeleopEnv2Arms(env_name=args.env_name, fps=args.fps, cameras=CAMERAS)
    obs, info = env.reset(seed=42, options={"randomize_light": False})
    return env, obs, info

def _destroy_env(env):
    try:
        if hasattr(env, "env"):
            env.env.close()
        else:
            env.close()
    except Exception:
        pass

def _open_viewer(env):
    try:
        import mujoco
        import mujoco.viewer as mjv
        physics = _get_physics(env)
        if physics is None: return None, None
        model = getattr(physics.model, "ptr", None) or getattr(physics.model, "_model", None)
        data  = getattr(physics.data,  "ptr", None) or getattr(physics.data,  "_data",  None)
        viewer = mjv.launch_passive(model, data)
        print("[Viewer] 原生窗口已启动。")
        return viewer, mjv
    except Exception as e:
        print(f"[Viewer] 启动失败：{e}")
        return None, None

def _close_viewer(viewer):
    try:
        if viewer is not None and hasattr(viewer, "close"):
            viewer.close()
    except Exception:
        pass

def _rebuild_everything(args, *, need_viewer):
    env, obs, _ = _create_env(args)
    controller = DualJoyConController(env)
    viewer = mjv = None
    if need_viewer:
        viewer, mjv = _open_viewer(env)
    return env, controller, viewer, mjv, obs

# ---------- 主程序 ----------
def main():
    ap = argparse.ArgumentParser(description="Teleop recorder -> HDF5")
    ap.add_argument("--env-name", type=str, default="hook-package-v1")
    ap.add_argument("--fps", type=float, default=FPS)
    ap.add_argument("--out", type=str, default="outputs/joycon_hdf5")
    ap.add_argument("--task", type=str, default="task1")
    ap.add_argument("--viewer", action="store_true")
    args = ap.parse_args()

    win_name = "Teleop Recorder (4 cams)"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)

    env, joy, viewer, mjv, obs = _rebuild_everything(args, need_viewer=args.viewer)

    def get_initial_poses(current_env):
        if INITIAL_FIXED is not None:
            initial_L = _pose_from_pos_quat(INITIAL_FIXED["left"]["pos"],  INITIAL_FIXED["left"]["quat_wxyz"])
            initial_R = _pose_from_pos_quat(INITIAL_FIXED["right"]["pos"], INITIAL_FIXED["right"]["quat_wxyz"])
        else:
            initial_L, initial_R = _pose_from_env(current_env)
        return initial_L, initial_R

    root = Path(os.path.join(args.out, args.task))
    print(f"[Recorder] 将保存到: {root.resolve()}")
    print("操作：按 B 开始/结束录制；窗口按 q 退出。")

    recording = False
    returning = False
    return_start_t = 0.0
    prev_B = False

    h5 = None
    step_idx = 0

    try:
        while True:
            loop_t0 = time.time()

            if viewer is not None and hasattr(viewer, "is_running") and not viewer.is_running():
                print("[Viewer] 关闭，继续无窗口。")
                viewer = None; mjv = None

            # 读手柄
            running_flag, left_pose, left_grip, right_pose, right_grip, _ = joy.poll()

            # B 键：开始/结束
            btn_B = joy.joy.get_button(B_INDEX)
            if btn_B and not prev_B:
                if recording:
                    # 结束
                    if h5 is not None: h5.close()
                    print(f"[Recorder] ■ 结束并保存，共 {step_idx} 帧")
                    recording = False
                    h5 = None
                else:
                    # 重建环境开始新集
                    print("[Recorder] 重建环境以开始新一集……")
                    _close_viewer(viewer); viewer = None; mjv = None
                    try: joy.close()
                    except: pass
                    _destroy_env(env)

                    env, joy, viewer, mjv, obs = _rebuild_everything(args, need_viewer=args.viewer)

                    returning = True
                    return_start_t = loop_t0
                    initial_L, initial_R = get_initial_poses(env)
                    print("[Recorder] 环境已重建，回到初始位姿中……")
                prev_B = btn_B
            else:
                prev_B = btn_B

            # 控一步
            if returning:
                obs, _, _, _, step_info = env.step(
                    left_pose=initial_L, left_gripper=1.0,
                    right_pose=initial_R, right_gripper=1.0
                )
                l_now, r_now = step_info["left_arm_pose"], step_info["right_arm_pose"]
                if (_pose_reached(l_now, initial_L) and _pose_reached(r_now, initial_R)) \
                   or ((time.time() - return_start_t) > MAX_RETURN_SECS):
                    joy.recenter_to(initial_L, initial_R)
                    returning = False
                    # 创建新 episode hdf5
                    ep_path = next_episode_path(root)
                    first_img = next((v for v in (obs.get("pixels", {}) or {}).values() if v is not None), None)
                    H, W = (first_img.shape[0], first_img.shape[1]) if first_img is not None else (CAMERAS["top_cam"][0], CAMERAS["top_cam"][1])
                    h5 = H5Writer(ep_path, fps=args.fps, img_hw=(H, W))
                    step_idx = 0
                    print(f"[Recorder] ▶ 开始录制 -> {ep_path.name}")
            else:
                obs, _, _, _, step_info = env.step(
                    left_pose=left_pose, left_gripper=left_grip,
                    right_pose=right_pose, right_gripper=right_grip
                )

            # 预览
            pix = obs.get("pixels", {}) if isinstance(obs, dict) else {}
            keys = ["top_cam", "button_cam", "wrist_cam_left", "wrist_cam_right"]
            imgs = [pix.get(k, None) for k in keys]
            any_img = next((im for im in imgs if im is not None), None)
            if any_img is not None:
                H, W, _ = any_img.shape
                black = np.zeros((H, W, 3), dtype=any_img.dtype)
                imgs = [im if im is not None else black for im in imgs]
                top = np.concatenate(imgs[:2], axis=1)
                bottom = np.concatenate(imgs[2:], axis=1)
                quad = np.concatenate([top, bottom], axis=0)
                cv2.imshow(win_name, quad[:, :, ::-1])
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[Recorder] 按 q，退出。")
                    break

            # 录制到 HDF5
            if recording or (h5 is not None and not returning):
                # action（从 step_info 里取 14 维；若你的 env 不同，可自行拼接）
                action = step_info["action"].astype(np.float32)

                # 低层状态
                qpos, qvel, effort = _read_qpos_qvel_effort(env)

                h5.append(
                    imgs_dict=pix,
                    action=action,
                    qpos=qpos,
                    qvel=qvel,
                    effort=effort
                )
                step_idx += 1
                recording = True  # 一旦写入就标记在录制

            # 控制频率
            elapsed = time.time() - loop_t0
            sleep_t = (1.0/args.fps) - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        print("\n[Recorder] Ctrl+C，退出。")
    finally:
        try:
            if h5 is not None: h5.close()
        except: pass
        try:
            if joy is not None: joy.close()
        except: pass
        cv2.destroyAllWindows()
        _close_viewer(viewer)
        _destroy_env(env)

if __name__ == "__main__":
    main()
