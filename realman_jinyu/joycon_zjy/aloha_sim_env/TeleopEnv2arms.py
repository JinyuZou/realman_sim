import sys, time, multiprocessing as mp
import numpy as np
import cv2
import gymnasium as gym

import pygame
from termcolor import colored
import sys


from realman_jinyu.kinematics.grad_ik import GradIK, GradIKConfig

from xbox_controller import DualJoyConController
FPS = 8
CAMERAS = {
    "wrist_cam_left":  [480, 640],
    "wrist_cam_right": [480, 640],
    "top_cam":    [480, 640],
    "button_cam":   [480, 640],
}

# ---------- 环境 ----------
class TeleopEnv2Arms1:
    def __init__(self, env_name="cube-transfer", fps=FPS, cameras=CAMERAS):
        self.cameras = cameras
        self.env = gym.make(
            f"realman_jinyu/{env_name}", fps=fps, cameras=cameras,
            render_camera="overhead_cam", enable_av=False
        ).unwrapped

        self.left_controller = GradIK(
            GradIKConfig(), self.env.physics,
            self.env.left_joints, self.env.left_eef_site)
        self.right_controller = GradIK(
            GradIKConfig(), self.env.physics,
            self.env.right_joints, self.env.right_eef_site)

        lT = self.env.left_arm.get_eef_pose()
        rT = self.env.right_arm.get_eef_pose()
        self.home_L = (lT[:3, :3].copy(), lT[:3, 3].copy())
        self.home_R = (rT[:3, :3].copy(), rT[:3, 3].copy())

    def _ik_step(self, left_pose, left_grip, right_pose, right_grip):
        lq = self.left_controller.run(
            self.env.left_arm.get_joint_positions(),
            left_pose[:3, 3], left_pose[:3, :3])
        rq = self.right_controller.run(
            self.env.right_arm.get_joint_positions(),
            right_pose[:3, 3], right_pose[:3, :3])
        action = np.zeros(14)
        action[:6], action[6]   = lq, left_grip
        action[7:13], action[13]= rq, right_grip
        return self.env.step(action)[0]

    def step_follow_or_home(self, running, left_pose, left_grip,
                            right_pose, right_grip):
        if running:
            return self._ik_step(left_pose, left_grip, right_pose, right_grip)
        else:  # 回到 home
            lT = np.eye(4)
            rT = np.eye(4)
            lT[:3, :3], lT[:3, 3] = self.home_L
            rT[:3, :3], rT[:3, 3] = self.home_R
            return self._ik_step(lT, 1.0, rT, 1.0)

    def reset(self, seed=None, options=None):
        obs, _ = self.env.reset(seed=seed, options=options)
        return obs,_

    def get_obs(self):
        return self.env.get_obs()

class TeleopEnv2Arms():
    """只控制左/右两臂（无中臂），输入为左右末端位姿 + 夹爪开合。"""
    def __init__(self, env_name="realman_jinyu/realman-aloha-v1", fps=FPS, cameras=CAMERAS):
        self.cameras = cameras
        # 关键：禁用中臂 AV
        self.env = gym.make(
            f"realman_jinyu/{env_name}", fps=fps, cameras=cameras,
            render_camera="top_cam"
        ).unwrapped

        self.left_controller = GradIK(
            config=GradIKConfig(),
            physics=self.env.physics,
            joints=self.env.left_joints,
            eef_site=self.env.left_eef_site,
        )
        self.right_controller = GradIK(
            config=GradIKConfig(),
            physics=self.env.physics,
            joints=self.env.right_joints,
            eef_site=self.env.right_eef_site,
        )

    def step(self, left_pose, left_gripper, right_pose, right_gripper):
        left_joints = self.left_controller.run(
            q=self.env.left_arm.get_joint_positions(),
            target_pos=left_pose[:3, 3],
            target_mat=left_pose[:3, :3],
        )
        right_joints = self.right_controller.run(
            q=self.env.right_arm.get_joint_positions(),
            target_pos=right_pose[:3, 3],
            target_mat=right_pose[:3, :3],
        )

        # 双臂动作向量：14 维
        action = np.zeros(14, dtype=np.float32)
        action[:6]   = left_joints
        action[6]    = float(left_gripper)
        action[7:13] = right_joints
        action[13]   = float(right_gripper)

        observation, reward, terminated, truncated, _info = self.env.step(action)

        info = {
            'left_arm_pose':  self.env.left_arm.get_eef_pose(),                # (4,4)
            'left_gripper':   float(self.env.left_arm.get_gripper_position()[0]),
            'right_arm_pose': self.env.right_arm.get_eef_pose(),               # (4,4)
            'right_gripper':  float(self.env.right_arm.get_gripper_position()[0]),
            'action': action,                                                 # (14,)
        }
        return observation, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        observation, _ = self.env.reset(seed=seed, options=options)
        info = {
            'left_arm_pose':  self.env.left_arm.get_eef_pose(),
            'left_gripper':   float(self.env.left_arm.get_gripper_position()[0]),
            'right_arm_pose': self.env.right_arm.get_eef_pose(),
            'right_gripper':  float(self.env.right_arm.get_gripper_position()[0]),
        }
        return observation, info
    
    def render_viewer(self):
        self.env.render_viewer()

    def get_obs(self):
        return self.env.get_obs()