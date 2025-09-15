from realman_jinyu.env.sim_env import RealmanEnv
from realman_jinyu.env.sim_config import XML_DIR,SIM_DT
import numpy as np
import os
from gymnasium import spaces


class HookPackageEnv(RealmanEnv):
    XML = os.path.join(XML_DIR, 'task_hang_package.xml')
    LEFT_POSE = [0.0, 0.6353, 0.9512, -0.0127, 1.2112, -1.57]
    LEFT_GRIPPER_POSE = 1
    RIGHT_POSE = [0.0, 0.6353, 0.9512, -0.0127, 1.2112, -1.57]
    RIGHT_GRIPPER_POSE = 1
    ENV_STATE_DIM = 14
    PROMPTS = [
        "hook package"
    ]

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.max_reward = 4

        self.package_joint = self.mjcf_root.find('joint', 'package_joint')
        self.hook_joint = self.mjcf_root.find('joint', 'hook_joint')

       
        self.observation_space_dict['environment_state'] = spaces.Box(
            low=-np.inf,
            
            high=np.inf,
            dtype=np.float64
        )
        self.observation_space = spaces.Dict(self.observation_space_dict)

    def get_obs(self) -> np.ndarray:
        obs = super().get_obs()
        obs['environment_state'] = np.concatenate([
            self.physics.bind(self.package_joint).qpos,
            self.physics.bind(self.hook_joint).qpos,
        ])
        return obs
    
    def set_state(self, state, environment_state):
        super().set_state(state, environment_state)
        self.physics.bind(self.package_joint).qpos = environment_state[:7]
        self.physics.bind(self.hook_joint).qpos = environment_state[7:14]
        self.physics.forward()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        # reset physics
        x_range = [-0.1, 0.1]
        y_range = [.0, .0]
        z_range = [0.2, 0.3]
        ranges = np.vstack([x_range, y_range, z_range])
        hook_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
        hook_quat = np.array([1, 0, 0, 0])

        x_range = [-.1, 0.1]
        y_range = [0, 0.15]
        z_range = [0.0, 0.0]
        ranges = np.vstack([x_range, y_range, z_range])
        package_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
        package_quat = np.array([1, 0, 0, 0])

        self.physics.bind(self.hook_joint).qpos = np.concatenate([hook_position, hook_quat])
        self.physics.bind(self.package_joint).qpos = np.concatenate([package_position, package_quat])



        self.physics.forward()

        observation = self.get_obs()
        info = {"is_success": False}

        return observation, info

    def get_reward(self):

        touch_left_gripper = False
        touch_right_gripper = False
        package_touch_table = False
        package_touch_hook = False
        pin_touched = False

        # return whether peg touches the pin
        contact_pairs = []
        for i_contact in range(self.physics.data.ncon):
            id_geom_1 = self.physics.data.contact[i_contact].geom1
            id_geom_2 = self.physics.data.contact[i_contact].geom2
            geom1 = self.physics.model.id2name(id_geom_1, 'geom')
            geom2 = self.physics.model.id2name(id_geom_2, 'geom')
            contact_pairs.append((geom1, geom2))
            contact_pairs.append((geom2, geom1))

        for geom1, geom2 in contact_pairs:
            if "package" in geom1 and "right" in geom2:
                touch_right_gripper = True

            if "package" in geom1 and "left" in geom2:
                touch_left_gripper = True

            if geom1 == "table" and "package" in geom2:
                package_touch_table = True

            if geom1 == "hook" and "package" in geom2:
                package_touch_hook = True

            if geom1 == "pin-package" and geom2 == "pin-hook":
                pin_touched = True

        reward = 0
        if touch_left_gripper and touch_right_gripper:  # touch both
            reward = 1
        if touch_left_gripper and touch_right_gripper and (not package_touch_table):  # grasp both
            reward = 2
        if package_touch_hook and (not package_touch_table):
            reward = 3
        if pin_touched:
            reward = 4
        return reward


def main():
    import gymnasium as gym
    import realman_jinyu
    import time
   

    env = gym.make("realman_jinyu/hook-package-v1", cameras={
            "top_cam":    (240, 304),
            "wrist_cam_left":   (240, 304),
            "wrist_cam_right":  (240, 304),
            "button_cam":  (240, 304),
        },
        fps=8.33)

    action = np.concatenate([
        env.unwrapped.LEFT_POSE,
        [env.unwrapped.LEFT_GRIPPER_POSE],
        env.unwrapped.RIGHT_POSE,
        [env.unwrapped.RIGHT_GRIPPER_POSE],
     
    ])

    options_list = [
        {"randomize_light": True},
        {"distractors": True},
        {"adverse": True},
        {}
    ]

    observation, info = env.reset(seed=42, options=options_list[0])

    i = 0
    j = 0
    while True:
        step_start = time.time()

        # Take a step in the environment using the chosen action
        observation, reward, terminated, truncated, info = env.step(action)

        env.unwrapped.render_viewer()

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = SIM_DT - (time.time() - step_start)
        time.sleep(max(0, time_until_next_step))

        if i % 20 == 0:
            env.reset(seed=42, options=options_list[j % len(options_list)])
            j += 1

        i += 1


if __name__ == '__main__':
    main()