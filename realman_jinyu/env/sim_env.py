import os
import logging
import numpy as np
import mujoco.viewer
from dm_control import mjcf
import gymnasium as gym
from gymnasium import spaces

from realman_jinyu.env.sim_config import (
    XML_DIR, CAMERAS, RENDER_CAMERA,
    AV_STATE_DIM, STATE_DIM, AV_ACTION_DIM, ACTION_DIM,
    SIM_DT, SIM_PHYSICS_DT, SIM_PHYSICS_ENV_STEP_RATIO,
    LEFT_JOINT_NAMES, LEFT_GRIPPER_JOINT_NAMES,
    RIGHT_JOINT_NAMES, RIGHT_GRIPPER_JOINT_NAMES,
   
    LEFT_ACTUATOR_NAMES, LEFT_GRIPPER_ACTUATOR_NAME,
    RIGHT_ACTUATOR_NAMES, RIGHT_GRIPPER_ACTUATOR_NAME,
  
    LEFT_EEF_SITE, RIGHT_EEF_SITE,
    LIGHT_NAME, 
)
from realman_jinyu.env.robot import SimRobotArm


class RealmanEnv(gym.Env):

    metadata = {"render_modes": ["rgb_array"], "render_fps": 1/SIM_DT}
    XML = os.path.join(XML_DIR, 'scence_two_arms_tube.xml')
    LEFT_POSE = [0.0, 0.6353, 0.9712, -0.0127, 1.2112, -1.57]
    LEFT_GRIPPER_POSE = 0.1
    RIGHT_POSE = [0.0, 0.6353, 0.9712, -0.0127, 1.2112, -1.57]
    RIGHT_GRIPPER_POSE = 0.1

    PROMPTS = []

    def __init__(
        self,
        fps=25,
        cameras=CAMERAS,
        render_camera=RENDER_CAMERA,
        render_height=240,
        render_width=304,
        
    ):

        super().__init__()
        self.fps = fps
        self.cameras = cameras.copy()
        self.render_camera = render_camera
        self.render_height = render_height
        self.render_width = render_width
        
       

        self.max_reward = 0
        self.step_count = 0
        self.viewer = None
        self.metadata["render_fps"] = self.fps

        self.step_dt = 1/self.fps
        self.n_ctrl_steps = round(self.step_dt/SIM_DT)
        if self.n_ctrl_steps < 1:
            raise ValueError("FPS too high for simulation")

        self.mjcf_root = mjcf.from_path(self.XML)
        self.mjcf_root.option.timestep = SIM_PHYSICS_DT

        

        self.physics = mjcf.Physics.from_mjcf_model(self.mjcf_root)

        self.left_joints = [self.mjcf_root.find('joint', name) for name in LEFT_JOINT_NAMES]
        self.left_gripper_joints = [self.mjcf_root.find('joint', name) for name in LEFT_GRIPPER_JOINT_NAMES]
        self.right_joints = [self.mjcf_root.find('joint', name) for name in RIGHT_JOINT_NAMES]
        self.right_gripper_joints = [self.mjcf_root.find('joint', name) for name in RIGHT_GRIPPER_JOINT_NAMES]
        self.left_actuators = [self.mjcf_root.find('actuator', name) for name in LEFT_ACTUATOR_NAMES]
        self.left_gripper_actuator = self.mjcf_root.find('actuator', LEFT_GRIPPER_ACTUATOR_NAME)
        self.right_actuators = [self.mjcf_root.find('actuator', name) for name in RIGHT_ACTUATOR_NAMES]
        self.right_gripper_actuator = self.mjcf_root.find('actuator', RIGHT_GRIPPER_ACTUATOR_NAME)
        self.left_eef_site = self.mjcf_root.find('site', LEFT_EEF_SITE)
        self.right_eef_site = self.mjcf_root.find('site', RIGHT_EEF_SITE)



        self.left_arm = SimRobotArm(
            physics=self.physics,
            joints=self.left_joints,
            actuators=self.left_actuators,
            eef_site=self.left_eef_site,
            has_gripper=True,
            gripper_joints=self.left_gripper_joints,
            gripper_actuator=self.left_gripper_actuator,
        )

        self.right_arm = SimRobotArm(
            physics=self.physics,
            joints=self.right_joints,
            actuators=self.right_actuators,
            eef_site=self.right_eef_site,
            has_gripper=True,
            gripper_joints=self.right_gripper_joints,
            gripper_actuator=self.right_gripper_actuator,
        )

        self.observation_space_dict = {}

   

        self.observation_space_dict['agent_pos'] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(STATE_DIM,),
            dtype=np.float64,
        )

        # Define pixels observation if cameras are provided
        if len(self.cameras) > 0:
            self.observation_space_dict['pixels'] = spaces.Dict(
                {
                    camera: spaces.Box(
                        low=0,
                        high=255,
                        shape=(*dim, 3),
                        dtype=np.uint8,
                    )
                    for camera, dim in self.cameras.items()
                }
            )

        self.observation_space = spaces.Dict(self.observation_space_dict)



        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(ACTION_DIM,), dtype=np.float64)

        if len(self.PROMPTS) > 0:
            self.prompt = self.PROMPTS[0]
        else:
            self.prompt = None

    def get_obs(self) -> np.ndarray:
        obs = {}
        obs['agent_pos'] = np.concatenate([
                self.left_arm.get_joint_positions(),
                self.left_arm.get_gripper_position(),
                self.right_arm.get_joint_positions(),
                self.right_arm.get_gripper_position(),
        ])

        if len(self.cameras) > 0:
            obs['pixels'] = {
                camera: self.physics.render(
                    height=dim[0],
                    width=dim[1],
                    camera_id=camera
                )
                for camera, dim in self.cameras.items()
            }

        return obs
    
    def set_prompt(self, prompt: str):
        if self.PROMPTS:
            assert prompt in self.PROMPTS, f"Prompt must be one of {self.PROMPTS}"
        self.prompt = prompt

    def get_qpos(self):
        return self.physics.data.qpos.copy()

    def set_qpos(self, qpos):
        self.physics.data.qpos[:] = qpos
        self.physics.forward()

    def set_state(self, state, environment_state):
        self.left_arm.set_joint_positions(state[:6])
        self.left_arm.set_gripper_position(state[6])
        self.right_arm.set_joint_positions(state[7:13])
        self.right_arm.set_gripper_position(state[13])
       
        self.physics.forward()

    def get_ctrl(self):

        return np.concatenate([
            self.left_arm.get_joint_ctrl(),
            np.array([self.left_arm.get_gripper_ctrl()]),
            self.right_arm.get_joint_ctrl(),
            np.array([self.right_arm.get_gripper_ctrl()]),
        ])

    def set_ctrl(self, ctrl):
        self.left_arm.set_joint_ctrl(ctrl[0:6])
        self.left_arm.set_gripper_ctrl(ctrl[6])
        self.right_arm.set_joint_ctrl(ctrl[7:13])
        self.right_arm.set_gripper_ctrl(ctrl[13])

       

    def get_reward(self):
        return 0

    def render(self):
        return self.physics.render(
            height=self.render_height,
            width=self.render_width,
            camera_id=self.render_camera
        )

    def step(self, action: np.ndarray) -> tuple:
        ctrl = action
        prev_ctrl = self.get_ctrl()

        actions = np.linspace(prev_ctrl, ctrl, self.n_ctrl_steps+1)[1:]

        for i in range(self.n_ctrl_steps):
            self.set_ctrl(actions[i])
            self.physics.step(nstep=SIM_PHYSICS_ENV_STEP_RATIO)

        observation = self.get_obs()
        reward = self.get_reward()

        terminated = reward == self.max_reward
    
        self.step_count += 1
        truncated = False
        info = {"is_success": reward == self.max_reward}

        return observation, reward, terminated, truncated, info

    def randomize_light(self):
        x_range = [-0.3, 0.3]
        y_range = [-0.3, 0.3]
        z_range = [1.5, 1.5]
        ranges = np.vstack([x_range, y_range, z_range])
        self.physics.named.model.light_type[LIGHT_NAME] = mujoco.mjtLightType.mjLIGHT_SPOT
        self.physics.named.model.light_pos[LIGHT_NAME] = np.random.uniform(ranges[:, 0], ranges[:, 1])
        self.physics.named.model.light_ambient[LIGHT_NAME] = np.random.uniform(0, 0.2, size=3)
        self.physics.named.model.light_diffuse[LIGHT_NAME] = np.random.uniform(.5, 0.9, size=3)

    def reset_light(self):
        self.physics.named.model.light_type[LIGHT_NAME] = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
        self.physics.named.model.light_pos[LIGHT_NAME] = np.array([0, 1, 1.5])
        self.physics.named.model.light_ambient[LIGHT_NAME] = np.array([0.0, 0.0, 0.0])
        self.physics.named.model.light_dir[LIGHT_NAME] = np.array([0, 0, -1])
        self.physics.named.model.light_diffuse[LIGHT_NAME] = np.array([0.7, 0.7, 0.7])
        self.physics.named.model.light_ambient[LIGHT_NAME] = np.array([0.2, 0.2, 0.2])

    def reset(self, seed=None, options: dict = None) -> tuple:
        super().reset(seed=seed, options=options)

        self.step_count = 0

        # reset physics
        self.physics.reset()

        # random light
        if options and options.get('randomize_light', False):
            self.randomize_light()
        else:
            self.reset_light()

        if options:
            self.set_prompt(options.get('prompt', self.prompt))

        self.left_arm.set_joint_positions(self.LEFT_POSE)
        self.left_arm.set_gripper_position(self.LEFT_GRIPPER_POSE)
        self.right_arm.set_joint_positions(self.RIGHT_POSE)
        self.right_arm.set_gripper_position(self.RIGHT_GRIPPER_POSE)


        self.left_arm.set_joint_ctrl(self.LEFT_POSE)
        self.left_arm.set_gripper_ctrl(self.LEFT_GRIPPER_POSE)
        self.right_arm.set_joint_ctrl(self.RIGHT_POSE)
        self.right_arm.set_gripper_ctrl(self.RIGHT_GRIPPER_POSE)

        self.physics.forward()

        observation = self.get_obs()
        info = {"is_success": False}

        return observation, info

    def render_viewer(self) -> np.ndarray:
        if self.viewer is None:
            # launch viewer
            self.viewer = mujoco.viewer.launch_passive(
                self.physics.model.ptr,
                self.physics.data.ptr,
                show_left_ui=True,
                show_right_ui=True,
            )

        # render
        self.viewer.sync()

    def close(self) -> None:
        """
        Closes the viewer if it's open.
        """
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if hasattr(self, "physics"):
            del self.physics

def snapshot_line(env) -> str:
    """不改环境类，从外部读取左右臂/夹爪位置并拼成一行。"""
    u = env.unwrapped
    # 关节位置
    qL  = u.physics.bind(u.left_joints).qpos.copy()           if u.left_joints else np.array([])
    qR  = u.physics.bind(u.right_joints).qpos.copy()          if u.right_joints else np.array([])
    qLg = u.physics.bind(u.left_gripper_joints).qpos.copy()   if u.left_gripper_joints else np.array([])
    qRg = u.physics.bind(u.right_gripper_joints).qpos.copy()  if u.right_gripper_joints else np.array([])

    parts = []
    if qL.size:  parts.append("L "  + " ".join(f"{v: .3f}" for v in qL))
    if qLg.size: parts.append(f"Lgrip:{qLg[0]: .3f}")
    if qR.size:  parts.append("R "  + " ".join(f"{v: .3f}" for v in qR))
    if qRg.size: parts.append(f"Rgrip:{qRg[0]: .3f}")
    return " | ".join(parts)


import cv2  # 新增

def main():
    import gymnasium as gym
    import realman_jinyu
    import time

    # 显式打开 4 个相机
    env = gym.make(
        "realman_jinyu/realman-aloha-v1",
        cameras={
            "top_cam":    (240, 304),
            "wrist_cam_left":   (240, 304),
            "wrist_cam_right":  (240, 304),
            "button_cam":  (240, 304),
        },
        fps=8.33,
    )

    action = np.concatenate([
        env.unwrapped.LEFT_POSE,
        [env.unwrapped.LEFT_GRIPPER_POSE],
        env.unwrapped.RIGHT_POSE,
        [env.unwrapped.RIGHT_GRIPPER_POSE],
    ])

    obs, _ = env.reset(seed=42, options={"randomize_light": False})

    i = 0
    last_print = 0.0

    while True:
        step_start = time.time()

        obs, _, _, _, _ = env.step(action)
        env.unwrapped.render_viewer()

        # ---- 四画面拼接 ----
        frames = [
            obs["pixels"]["top_cam"],
            obs["pixels"]["wrist_cam_left"],
            obs["pixels"]["wrist_cam_right"],
            obs["pixels"]["button_cam"],
        ]
        # 上下两半
        top_row    = np.hstack([frames[0], frames[1]])
        bottom_row = np.hstack([frames[2], frames[3]])
        quad       = np.vstack([top_row, bottom_row])
        # OpenCV 需要 RGB→BGR
        quad_bgr = cv2.cvtColor(quad, cv2.COLOR_RGB2BGR)
        cv2.imshow("4-camera view", quad_bgr)
        # 按 q 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # --------------------

        now = time.time()
        if now - last_print >= 0.2:
            line = snapshot_line(env)
            print("\r" + line + " " * 6, end="", flush=True)
            last_print = now

        time_until_next = SIM_DT - (time.time() - step_start)
        time.sleep(max(0, time_until_next))

        if i % 10 == 0:
            obs, _ = env.reset(seed=42)
            i = 0
        i += 1

    cv2.destroyAllWindows()
    env.close()

if __name__ == '__main__':
    main()
