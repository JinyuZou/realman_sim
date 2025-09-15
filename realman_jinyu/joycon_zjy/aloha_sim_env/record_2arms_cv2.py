#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, io, json, time, math, argparse
from pathlib import Path
import numpy as np
import imageio.v3 as iio
import cv2

import gymnasium as gym
import realman_jinyu  


from xbox_controller import DualJoyConController
from TeleopEnv2arms import TeleopEnv2Arms

# ---------------- 基本配置 ----------------
FPS = 8
CAMERAS = {
    "top_cam":         [480, 640],
    "button_cam":      [480, 640],
    "wrist_cam_left":  [480, 640],
    "wrist_cam_right": [480, 640],
}
B_INDEX = 1  # 手柄B键

# 固定初始位姿（留空则用当前位姿）
INITIAL_FIXED = None

POS_TOL_M = 1e-2         # 位置容差 1 cm
ROT_TOL_DEG = 5.0        # 角度容差 5 度
MAX_RETURN_SECS = 2.0    # 回位最多 2s
# -----------------------------------------------------------

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

def next_episode_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    idx = 0
    while True:
        ep = root / f"episode_{idx:04d}"
        if not ep.exists():
            (ep / "frames").mkdir(parents=True, exist_ok=True)
            return ep
        idx += 1

def save_meta(ep_dir: Path, meta: dict):
    with open(ep_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

def save_npz(ep_dir: Path, arrays: dict):
    pack = {k: np.asarray(v) for k, v in arrays.items()}
    np.savez_compressed(ep_dir / "data.npz", **pack)

# ---------------- MuJoCo viewer 支持 ----------------
def _mj_model_data_from_env(env):
    """尽量通用地从 TeleopEnv2Arms -> RealmanEnv 里拿到 mjModel/mjData 指针。"""
    physics = None
    for cand in [getattr(env, "env", None),
                 getattr(getattr(env, "env", None), "env", None),
                 env]:
        if cand is None: 
            continue
        physics = getattr(cand, "physics", None)
        if physics is not None:
            break
    if physics is None:
        return None, None
    # dm_control Physics: model.ptr / data.ptr （新版也可能是 _model/_data）
    model = getattr(getattr(physics, "model", None), "ptr", None) \
         or getattr(getattr(physics, "model", None), "_model", None)
    data  = getattr(getattr(physics, "data",  None), "ptr", None) \
         or getattr(getattr(physics, "data",  None), "_data",  None)
    return model, data

def _init_mj_viewer(env):
    try:
        import mujoco
        import mujoco.viewer as mjv
    except Exception as e:
        print(f"[Viewer] 载入 mujoco.viewer 失败：{e}")
        return None, None
    model, data = _mj_model_data_from_env(env)
    if model is None or data is None:
        print("[Viewer] 没找到 mjModel/mjData；无法打开原生窗口。")
        return None, None
    try:
        viewer = mjv.launch_passive(model, data)  # 返回 Viewer 对象
        print("[Viewer] 原生窗口已启动（被动模式）。")
        return viewer, mjv
    except Exception as e:
        print(f"[Viewer] 启动失败：{e}")
        return None, None

# -----------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Teleop recorder with initial return (B start/stop, CV2 view, optional MuJoCo viewer).")
    ap.add_argument("--env-name", type=str, default="realman-aloha-v1")
    ap.add_argument("--fps", type=float, default=FPS)
    ap.add_argument("--out", type=str, default="outputs/joycon_dataset")
    ap.add_argument("--task", type=str, default="task1")
    ap.add_argument("--viewer", action="store_true", help="打开 MuJoCo 原生渲染窗口（mujoco.viewer）")
    args = ap.parse_args()

    # 1) 环境 + 控制器
    env = TeleopEnv2Arms(env_name=args.env_name, fps=args.fps, cameras=CAMERAS)
    joy = DualJoyConController(env)

    # reset
    obs, info = env.reset(seed=42, options={"randomize_light": False})

    # 2) 初始位姿
    if INITIAL_FIXED is not None:
        initial_L = _pose_from_pos_quat(INITIAL_FIXED["left"]["pos"],
                                        INITIAL_FIXED["left"]["quat_wxyz"])
        initial_R = _pose_from_pos_quat(INITIAL_FIXED["right"]["pos"],
                                        INITIAL_FIXED["right"]["quat_wxyz"])
    else:
        initial_L, initial_R = _pose_from_env(env)

    # 3) 保存路径与提示
    root = Path(os.path.join(args.out, args.task))
    print(f"[Recorder] 数据将保存到: {root.resolve()}")
    print("操作：按 B 开始 → 先回初始位姿 → 自动开始录制；再按 B 结束并保存；窗口按 q 退出。")

    # 4)（可选）启动 MuJoCo Viewer
    viewer = None
    mjv = None
    if args.viewer:
        viewer, mjv = _init_mj_viewer(env)

    # 缓冲
    recording = False
    returning = False
    return_start_t = 0.0
    prev_B = False

    frames_ts, frames_action = [], []
    frames_left_pose, frames_right_pose = [], []
    frames_left_grip, frames_right_grip = [], []
    frames_agent_pos = []
    step_idx = 0

    meta_base = {
        "env_name": args.env_name,
        "fps": args.fps,
        "cameras": CAMERAS,
        "schema": {
            "agent_pos": "float32 [T, state_dim]",
            "action": "float32 [T, 14]  (L6,Lg,R6,Rg)",
            "left_pose":  "float32 [T, 4,4]",
            "right_pose": "float32 [T, 4,4]",
            "left_gripper":  "float32 [T]",
            "right_gripper": "float32 [T]",
            "timestamp_s": "float64 [T]",
            "frames": "PNG images per step: *_ZL.png, *_ZR.png, *_WL.png, *_WR.png",
        }
    }

    # OpenCV 窗口
    win_name = "Teleop Recorder (4 cams)"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)

    try:
        dt = 1.0 / float(args.fps)
        while True:
            loop_t0 = time.time()

            # 如果 viewer 打开但被用户关闭，就退出或继续无窗口运行
            if viewer is not None and hasattr(viewer, "is_running") and not viewer.is_running():
                print("[Viewer] 窗口已关闭，继续无窗口运行。")
                viewer = None
                mjv = None

            # 读手柄
            running_flag, left_pose, left_grip, right_pose, right_grip, _ = joy.poll()

            # B 键：开始/结束
            btn_B = joy.joy.get_button(B_INDEX)
            if btn_B and not prev_B:
                if recording:
                    # ---- 结束并保存 ----
                    save_npz(ep_dir, {
                        "timestamp_s":  frames_ts,
                        "action":       frames_action,
                        "left_pose":    frames_left_pose,
                        "right_pose":   frames_right_pose,
                        "left_gripper": frames_left_grip,
                        "right_gripper":frames_right_grip,
                        "agent_pos":    frames_agent_pos,
                    })
                    meta = {**meta_base, "episode": ep_dir.name, "steps": len(frames_ts)}
                    save_meta(ep_dir, meta)
                    print(f"[Recorder] ■ 结束并保存 -> {ep_dir.name}，共 {len(frames_ts)} 帧")
                    recording = False
                else:
                    # ---- 准备开始：先回初始位姿 ----
                    returning = True
                    return_start_t = loop_t0
                    print("[Recorder] 对齐到初始位姿中……")
            prev_B = btn_B

            # ============ 控制一步 ============
            if returning:
                obs, _, _, _, step_info = env.step(
                    left_pose=initial_L, left_gripper=1.0,
                    right_pose=initial_R, right_gripper=1.0
                )
                l_now, r_now = step_info["left_arm_pose"], step_info["right_arm_pose"]
                if (_pose_reached(l_now, initial_L) and _pose_reached(r_now, initial_R)) \
                   or ((loop_t0 - return_start_t) > MAX_RETURN_SECS):
                    joy.recenter_to(initial_L, initial_R)
                    returning = False
                    ep_dir = next_episode_dir(root)
                    print(f"[Recorder] ▶ 开始录制 -> {ep_dir.name}")
                    frames_ts.clear(); frames_action.clear()
                    frames_left_pose.clear(); frames_right_pose.clear()
                    frames_left_grip.clear(); frames_right_grip.clear()
                    frames_agent_pos.clear()
                    step_idx = 0
                    save_meta(ep_dir, {**meta_base, "episode": ep_dir.name, "steps": 0})
                    recording = True
            else:
                obs, _, _, _, step_info = env.step(
                    left_pose=left_pose, left_gripper=left_grip,
                    right_pose=right_pose, right_gripper=right_grip
                )

            # ====== 原生 MuJoCo viewer 同步（如果打开了）======
            if viewer is not None:
                try:
                    viewer.sync()  # 将当前 mjData 渲染到窗口
                except Exception:
                    pass  # 防止偶发渲染异常中断采集

            # ===== OpenCV 四画面预览 =====
            pix = obs.get("pixels", {})
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
                cv2.imshow(win_name, quad[:, :, ::-1])  # RGB->BGR
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[Recorder] q 被按下，退出。")
                    break

            # 录制
            if recording:
                zl = pix.get("top_cam", None)
                zr = pix.get("button_cam", None)
                wl = pix.get("wrist_cam_left", None)
                wr = pix.get("wrist_cam_right", None)
                frame_base = ep_dir / "frames" / f"{step_idx:06d}"
                if zl is not None: iio.imwrite(str(frame_base)+"_ZL.png", zl)
                if zr is not None: iio.imwrite(str(frame_base)+"_ZR.png", zr)
                if wl is not None: iio.imwrite(str(frame_base)+"_WL.png", wl)
                if wr is not None: iio.imwrite(str(frame_base)+"_WR.png", wr)

                frames_ts.append(time.time())
                frames_action.append(step_info["action"].astype(np.float32))
                frames_left_pose.append(step_info["left_arm_pose"].astype(np.float32))
                frames_right_pose.append(step_info["right_arm_pose"].astype(np.float32))
                frames_left_grip.append(float(step_info["left_gripper"]))
                frames_right_grip.append(float(step_info["right_gripper"]))
                agent_pos = obs.get("agent_pos", None)
                frames_agent_pos.append(
                    np.zeros((0,), dtype=np.float32) if agent_pos is None else agent_pos.astype(np.float32)
                )
                step_idx += 1

            # 控制频率
            elapsed = time.time() - loop_t0
            sleep_t = (1.0/args.fps) - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        print("\n[Recorder] 收到 Ctrl+C，安全退出。")
    finally:
        joy.close()
        cv2.destroyAllWindows()
        # 关闭 MuJoCo viewer
        try:
            if viewer is not None and hasattr(viewer, "close"):
                viewer.close()
        except Exception:
            pass
        # 关闭底层 env
        try:
            env.env.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
