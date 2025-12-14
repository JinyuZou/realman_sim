#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, time, argparse
from pathlib import Path
import numpy as np
import cv2
import torch

import gymnasium as gym  # noqa: F401
import realman_jinyu     # noqa: F401

from xbox_controller import DualJoyConController
from TeleopEnv2arms import TeleopEnv2Arms

# Try both LeRobot import paths
try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
except Exception:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ---------------- Basic config ----------------
FPS = 8
CAMERAS = {
    "top_cam":         [480, 640],
    "button_cam":      [480, 640],
    "wrist_cam_left":  [480, 640],
    "wrist_cam_right": [480, 640],
}
B_INDEX = 1         # B button
DEBOUNCE_S = 0.30   # seconds
INITIAL_FIXED = None
POS_TOL_M = 1e-2
ROT_TOL_DEG = 5.0
MAX_RETURN_SECS = 2.0

# -----------------------------------------------------------
def _pose_from_pos_quat(pos_xyz, quat_wxyz):
    """wxyz -> 3x3 R, build 4x4 T"""
    qw, qx, qy, qz = quat_wxyz
    R = np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1-2*(qx*qx+qy*qy)],
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

# ---------------- Viewer helpers ----------------
def _mj_model_data_from_env(env):
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
        return None, None, None
    model = getattr(getattr(physics, "model", None), "ptr", None) \
         or getattr(getattr(physics, "model", None), "_model", None)
    data  = getattr(getattr(physics, "data",  None), "ptr", None) \
         or getattr(getattr(physics, "data",  None), "_data",  None)
    return physics, model, data

def _open_viewer(env):
    try:
        import mujoco.viewer as mjv
    except Exception as e:
        print(f"[Viewer] Failed to import mujoco.viewer: {e}")
        return None, None
    _, model, data = _mj_model_data_from_env(env)
    if model is None or data is None:
        print("[Viewer] No mjModel/mjData handle found; cannot open native viewer.")
        return None, None
    try:
        viewer = mjv.launch_passive(model, data)
        print("[Viewer] Native viewer launched (passive mode).")
        return viewer, mjv
    except Exception as e:
        print(f"[Viewer] Failed to launch native viewer: {e}")
        return None, None

def _close_viewer(viewer):
    try:
        if viewer is not None and hasattr(viewer, "close"):
            viewer.close()
    except Exception:
        pass

# ---------------- Env / Controller ----------------
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

def _rebuild_everything(args, *, need_viewer):
    env, obs, _ = _create_env(args)
    controller = DualJoyConController(env)
    viewer = mjv = None
    if need_viewer:
        viewer, mjv = _open_viewer(env)
    return env, controller, viewer, mjv, obs

# ---------------- LeRobot helpers ----------------
def _probe_shapes_for_features(env):
    """Step once with hold pose to read pixel shapes and dims."""
    left_pose, right_pose = _pose_from_env(env)
    obs, _, _, _, step_info = env.step(
        left_pose=left_pose, left_gripper=1.0,
        right_pose=right_pose, right_gripper=1.0
    )
    pixels = obs.get("pixels", {}) if isinstance(obs, dict) else {}
    cam_shapes = {k: v.shape for k, v in pixels.items() if v is not None}  # HWC
    agent_pos = obs.get("agent_pos", None)
    state_dim = int(agent_pos.shape[0]) if agent_pos is not None else 0
    action_dim = int(step_info["action"].shape[0])
    pose_dim = int(step_info["left_arm_pose"].size)  # 4x4 -> 16
    print("[Probe]", cam_shapes, state_dim, action_dim, pose_dim)
    return cam_shapes, state_dim, action_dim, pose_dim

def _build_features(cam_shapes, state_dim, action_dim, pose_dim):
    """Include 'task' as int32 (1,) to avoid string-mapping issues."""
    features = {}
    for cam, shape in cam_shapes.items():
        H, W, C = shape
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": (H, W, C),
            "names": ["height", "width", "channel"],
        }
    if state_dim > 0:
        features["observation.state"] = {"dtype": "float32", "shape": (state_dim,), "names": None}
    features["action"]         = {"dtype": "float32", "shape": (action_dim,), "names": None}
    features["left_arm_pose"]  = {"dtype": "float32", "shape": (pose_dim,),  "names": None}
    features["right_arm_pose"] = {"dtype": "float32", "shape": (pose_dim,),  "names": None}
    features["left_gripper"]   = {"dtype": "float32", "shape": (1,),         "names": None}
    features["right_gripper"]  = {"dtype": "float32", "shape": (1,),         "names": None}
    # <<< key change: int32 with shape (1,) >>>
    #features["task"]           = {"dtype": "int32",   "shape": (1,),         "names": None}
    return features

def wait_for_cameras_ready(env, video_keys, video_shapes, timeout_s=2.0, fps=FPS):
    """Warm-up cameras so first frames are valid."""
    hold_L, hold_R = _pose_from_env(env)
    t0 = time.time()
    period = 1.0 / max(1e-6, float(fps))
    while True:
        loop_t0 = time.time()
        obs, _, _, _, _ = env.step(
            left_pose=hold_L,  left_gripper=1.0,
            right_pose=hold_R, right_gripper=1.0
        )
        pix = obs.get("pixels", {}) if isinstance(obs, dict) else {}
        ok = True
        for vkey in video_keys:
            cam = vkey.split("observation.images.", 1)[1]
            H, W, C = video_shapes[vkey]
            img = pix.get(cam, None)
            if img is None or img.shape != (H, W, C) or img.dtype != np.uint8:
                ok = False
                break
        if ok or (time.time() - t0) > timeout_s:
            return
        sleep_t = period - (time.time() - loop_t0)
        if sleep_t > 0:
            time.sleep(sleep_t)

# -----------------------------------------------------------
def main():
    os.environ.setdefault("TMPDIR", str(Path("./data/lerobot_tmp").resolve()))
    Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)

    ap = argparse.ArgumentParser(description="Teleop recorder → LeRobotDataset (+ optional HF push)")
    ap.add_argument("--env-name", type=str, default="hook-package-v1")
    ap.add_argument("--fps", type=float, default=FPS)
    ap.add_argument("--viewer", action="store_true", help="Open the native MuJoCo viewer (mujoco.viewer)")
    ap.add_argument("--repo-id", type=str, default="Jinyu220/realman_teleop_dataset")
    ap.add_argument("--root", type=str, default="~/datasets/lerobot")
    ap.add_argument("--task", type=str, default="2", help="Task label (integer is recommended)")
    ap.add_argument("--push", dest="push", action="store_true", help="Push to the HuggingFace Hub after recording (requires login)")
    ap.add_argument("--no-push", dest="push", action="store_false")
    ap.set_defaults(push=True) 
    args = ap.parse_args()
    args.root = str(Path(args.root).expanduser().resolve())

    # OpenCV preview window
    win_name = "Teleop Recorder (4 cams)"
    GUI_OK = True
    try:
        cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
    except cv2.error as e:
        print("[Preview] OpenCV HighGUI not available; headless mode:", e)
        GUI_OK = False

    # First build (probe)
    env, joy, viewer, mjv, obs = _rebuild_everything(args, need_viewer=args.viewer)

    def get_initial_poses(current_env):
        if INITIAL_FIXED is not None:
            initial_L = _pose_from_pos_quat(INITIAL_FIXED["left"]["pos"],
                                            INITIAL_FIXED["left"]["quat_wxyz"])
            initial_R = _pose_from_pos_quat(INITIAL_FIXED["right"]["pos"],
                                            INITIAL_FIXED["right"]["quat_wxyz"])
        else:
            initial_L, initial_R = _pose_from_env(current_env)
        return initial_L, initial_R

    print(f"[Recorder] Will save as LeRobot dataset to: {args.root} / {args.repo_id}")

    # Probe → build features (explicitly include int32 task)
    cam_shapes, state_dim, action_dim, pose_dim = _probe_shapes_for_features(env)
    features_suggested = _build_features(cam_shapes, state_dim, action_dim, pose_dim)
    # 强制把 task 纳入 schema：int32、shape=(1,)
    

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=str(Path(args.root) / args.repo_id),
        fps=int(round(args.fps)),
        
        features=features_suggested,
        image_writer_threads=8,
        image_writer_processes=4,
    )

    # Effective schema (may reuse existing)
    ds_features = dataset.features
    print("[LeRobot] Features in use (effective):")
    for k, v in ds_features.items():
        print("  -", k, v["dtype"], v["shape"])

    # If an older dataset directory forces `task` to string, ask user to clean/restart
    if "task" in ds_features and ds_features["task"]["dtype"].lower() in ("string", "str", "utf8"):
        print("[LeRobot] ERROR: existing dataset schema defines task as string; "
              "please remove the old dataset directory or change --repo-id / --root to a new path.")
        raise SystemExit(2)

    video_keys = [k for k, spec in ds_features.items() if spec["dtype"] == "video"]
    video_shapes = {k: tuple(ds_features[k]["shape"]) for k in video_keys}
    print("[LeRobot] Video keys:", video_keys)

    # ---- Control state ----
    recording = False
    returning = False
    return_start_t = 0.0
    prev_B = False
    last_toggle_t = 0.0
    step_idx = 0
    episode_idx = 0

    try:
        while True:
            loop_t0 = time.time()

            if viewer is not None and hasattr(viewer, "is_running") and not viewer.is_running():
                print("[Viewer] Window closed; continuing in headless mode.")
                viewer = None
                mjv = None

            _, left_pose, left_grip, right_pose, right_grip, _ = joy.poll()

            # Toggle by B with debounce
            btn_B = joy.joy.get_button(B_INDEX)
            now = time.time()
            if btn_B and not prev_B and (now - last_toggle_t) > DEBOUNCE_S:
                last_toggle_t = now
                if recording:
                    if step_idx > 0:
                        dataset.save_episode()
                        print(f"[Recorder] ■ Finished and saved Episode {episode_idx} ({step_idx} frames)")
                    else:
                        print("[Recorder] Episode has 0 frames; skipping save.")
                    recording = False
                    episode_idx += 1
                else:
                    print("[Recorder] Rebuilding environment to start a new episode...")
                    _close_viewer(viewer); viewer = None; mjv = None
                    try: joy.close()
                    except Exception: pass
                    _destroy_env(env)

                    env, joy, viewer, mjv, obs = _rebuild_everything(args, need_viewer=args.viewer)

                    returning = True
                    return_start_t = loop_t0
                    initial_L, initial_R = get_initial_poses(env)
                    print(f"[Recorder] Environment rebuilt; aligning to initial pose... (Episode {episode_idx})")
                    step_idx = 0
            prev_B = btn_B

            # Control one step
            if returning:
                obs, _, _, _, step_info = env.step(
                    left_pose=initial_L, left_gripper=1.0,
                    right_pose=initial_R, right_gripper=1.0
                )
                l_now, r_now = step_info["left_arm_pose"], step_info["right_arm_pose"]
                if (_pose_reached(l_now, initial_L) and _pose_reached(r_now, initial_R)) \
                   or ((time.time() - return_start_t) > MAX_RETURN_SECS):
                    # Recenter controller & warm up cameras
                    joy.recenter_to(initial_L, initial_R)
                    wait_for_cameras_ready(env, video_keys, video_shapes, timeout_s=2.0, fps=args.fps)
                    recording = True
                    step_idx = 0
                    returning = False
                    print(f"[Recorder] ▶ Start recording Episode {episode_idx}")
            else:
                obs, _, _, _, step_info = env.step(
                    left_pose=left_pose, left_gripper=left_grip,
                    right_pose=right_pose, right_gripper=right_grip
                )

            # Preview (4 cams)
            if GUI_OK:
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
                    cv2.imshow(win_name, quad[:, :, ::-1])  # RGB->BGR
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("[Recorder] Key 'q' pressed; exiting.")
                        break

            # Write one frame
            if recording:
                f = {}

                # Optional proprio
                agent_pos = obs.get("agent_pos", None)
                if "observation.state" in ds_features and agent_pos is not None:
                    f["observation.state"] = torch.tensor(agent_pos.reshape(-1).astype(np.float32))

                # Actions / poses / grippers
                f["action"] = torch.tensor(step_info["action"].reshape(-1).astype(np.float32))
                f["left_arm_pose"]  = torch.tensor(step_info["left_arm_pose"].reshape(-1).astype(np.float32))
                f["right_arm_pose"] = torch.tensor(step_info["right_arm_pose"].reshape(-1).astype(np.float32))
                f["left_gripper"]   = torch.tensor([float(step_info["left_gripper"])], dtype=torch.float32)
                f["right_gripper"]  = torch.tensor([float(step_info["right_gripper"])], dtype=torch.float32)

                # Images: enforce write with padding/resizing if needed
                pix = obs.get("pixels", {}) if isinstance(obs, dict) else {}
                for vkey in video_keys:
                    cam = vkey.split("observation.images.", 1)[1]
                    H, W, C = video_shapes[vkey]
                    img = pix.get(cam, None)
                    if img is None:
                        img = np.zeros((H, W, C), dtype=np.uint8)
                    else:
                        if img.ndim == 2:
                            img = np.stack([img]*3, axis=-1)
                        if img.shape[:2] != (H, W):
                            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
                        if img.dtype != np.uint8:
                            img = img.astype(np.uint8, copy=False)
                    f[vkey] = torch.from_numpy(img.copy())

                # <<< always write task as int32 length-1 >>>
                try:
                    v = int(args.task)
                except Exception:
                    v = 0
                v = int(args.task)         # 比如命令行 --task 2
                f["task"] = int(args.task) 
               
                dataset.add_frame(f)
                step_idx += 1

            # FPS control
            elapsed = time.time() - loop_t0
            sleep_t = (1.0/args.fps) - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        print("\n[Recorder] Ctrl+C received; exiting safely.")
    finally:
        try:
            if recording and step_idx > 0:
                dataset.save_episode()
                print(f"[Recorder] ■ Force-finished and saved Episode {episode_idx} ({step_idx} frames)")
        except Exception as e:
            print(f"[Recorder] Error while saving the last episode: {e}")

        try:
            joy.close()
        except Exception:
            pass
        try:
            if GUI_OK:
                cv2.destroyAllWindows()
        except Exception:
            pass
        _close_viewer(viewer)
        _destroy_env(env)

        if args.push:
            print("pushing_zjy+++++++++++++++++++++++++++++++")
            try:
                dataset.push_to_hub(private=False)
                print("[LeRobot] Pushed to HuggingFace Hub.")
            except Exception as e:
                print(f"[LeRobot] push_to_hub failed: {e}")

if __name__ == "__main__":
    main()
