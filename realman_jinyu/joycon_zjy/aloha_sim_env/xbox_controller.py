#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Xbox 手柄控制器（pygame 版）
返回：
    running, left_pose, left_grip, right_pose, right_grip, hat
其中 hat 统一为 (x,y) ∈ {-1,0,1}^2
"""
import pygame
import numpy as np
from realman_jinyu.env.sim_config import XML_DIR,SIM_DT

AXIS_DEAD = 0.15

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

def _make_T(R, p):
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


class DualJoyConController:
    def __init__(self, env,
                 fps=30,
                 xy_speed=0.30,
                 z_speed=0.25,
                 max_step=0.02,
                 workspace_lo=None,
                 workspace_hi=None):
        self.env = env
        self.dt = 1.0 / fps
        self.xy_speed = xy_speed
        self.z_speed = z_speed
        self.max_step = max_step
        self.ws_lo = workspace_lo
        self.ws_hi = workspace_hi

        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise RuntimeError("未检测到手柄")
        self.joy = pygame.joystick.Joystick(0)
        self.joy.init()

        # 记录初始位姿（home）
        lT = self.env.env.left_arm.get_eef_pose()
        rT = self.env.env.right_arm.get_eef_pose()
        self.RL_lock = lT[:3, :3].copy()
        self.RR_lock = rT[:3, :3].copy()
        self.pL = lT[:3, 3].astype(float).copy()
        self.pR = rT[:3, 3].astype(float).copy()

        self.left_grip = float(_clamp(self.env.env.left_arm.get_gripper_position()[0], 0, 1))
        self.right_grip = float(_clamp(self.env.env.right_arm.get_gripper_position()[0], 0, 1))

        self.running = True   # 外部可覆盖
        self.hat = (0, 0)     # 十字键状态

    def close(self):
        try:
            self.joy.quit()
        except Exception:
            pass
        pygame.quit()
    def recenter_to(self, T_left, T_right):
        """把手柄内部积分的‘零点’对齐到给定位姿（通常是 initial_L/R）"""
        self.RL_lock = T_left[:3, :3].copy()
        self.pL      = T_left[:3, 3].astype(float).copy()
        self.RR_lock = T_right[:3, :3].copy()
        self.pR      = T_right[:3, 3].astype(float).copy()

    def recenter_from_env(self):
        """把零点对齐到当前真实末端位姿（应急/调试用）"""
        lT = self.env.env.left_arm.get_eef_pose()
        rT = self.env.env.right_arm.get_eef_pose()
        self.recenter_to(lT, rT)

    def _axis(self, idx):
        v = float(self.joy.get_axis(idx))
        return 0.0 if abs(v) < AXIS_DEAD else v

    def _apply_ws(self, p):
        if self.ws_lo is not None:
            p = np.maximum(p, self.ws_lo)
        if self.ws_hi is not None:
            p = np.minimum(p, self.ws_hi)
        return p

    def poll(self):
        """
        每帧调用一次，返回：
        running, left_pose, left_grip, right_pose, right_grip, hat
        已把十字键增量算在内部，主循环无需再改 left_pose。
        """
        pygame.event.pump()

        # 十字键解析：优先 hat，否则用按钮 13/14/15/16
        try:
            self.hat = self.joy.get_hat(0)
        except pygame.error:
            up   = self.joy.get_button(13)
            down = self.joy.get_button(14)
            left = self.joy.get_button(15)
            right = self.joy.get_button(16)
            x =  1 if right else (-1 if left else 0)
            y =  1 if up   else (-1 if down else 0)
            self.hat = (x, y)

        # 摇杆
        lx =  self._axis(0)
        ly = -self._axis(1)
        rx =  self._axis(3)
        ry = -self._axis(4)

        # XY 位移
        dPL = np.array([lx, ly, 0]) * self.xy_speed * self.dt
        dPR = np.array([rx, ry, 0]) * self.xy_speed * self.dt
        dPL = np.clip(dPL, -self.max_step, self.max_step)
        dPR = np.clip(dPR, -self.max_step, self.max_step)

        # 左臂 Z：十字键 ↑/↓（按住连续移动）
        up   = self.joy.get_button(13)
        down = self.joy.get_button(14)
        zL = (1.0 if up else 0.0) - (1.0 if down else 0.0)
        dPL[2] = np.clip(zL * self.z_speed * self.dt, -self.max_step, self.max_step)

        # 右臂 Z：Y(3) / A(0)
        btn_Y = self.joy.get_button(3)
        btn_A = self.joy.get_button(0)
        zR = (1.0 if btn_Y else 0.0) - (1.0 if btn_A else 0.0)
        dPR[2] = np.clip(zR * self.z_speed * self.dt, -self.max_step, self.max_step)

        # 夹爪：LT(2) / RT(5) 轴
        lt = self.joy.get_axis(2)
        rt = self.joy.get_axis(5)
        self.left_grip  = 0.0 if lt > 0.5 else 1.0
        self.right_grip = 0.0 if rt > 0.5 else 1.0

        # 更新位置并限制工作空间
        self.pL = self._apply_ws(self.pL + dPL)
        self.pR = self._apply_ws(self.pR + dPR)

        left_pose  = _make_T(self.RL_lock, self.pL)
        right_pose = _make_T(self.RR_lock, self.pR)

        return self.running, left_pose, self.left_grip, right_pose, self.right_grip, self.hat