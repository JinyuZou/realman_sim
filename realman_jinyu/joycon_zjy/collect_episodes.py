#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, time, glob, pickle
import numpy as np
import torch
from tqdm import tqdm
from termcolor import colored

# your env
import gym_av_aloha
from gym_av_aloha.scripts.teleoperate import TeleopEnv
from gym_av_aloha.env.sim_config import SIM_DT

# dataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# controller
from joycon_control import DualJoyConController, R_to_T, CTRL_HZ

def prompt(env: TeleopEnv, message):
    while True:
        user_input = input(message).strip()
        if env.prompts and user_input not in env.prompts:
            print(colored(f"Invalid input. Choose from: {', '.join(env.prompts)}", 'red'))
            continue
        print(colored(f"You entered: '{user_input}'", 'green'))
        if input("Is this correct? (y/n): ").strip().lower() == 'y':
            return user_input

def run_episode(env: TeleopEnv, controller: DualJoyConController, task, episode_idx):
    ts, info = env.reset()
    print(colored("Hold Right A to start recording (release A to end).", "cyan"))

    # wait to start
    while True:
        running, *_ = controller.poll()
        env.render_viewer()
        if running:
            env.set_prompt(task)
            break
        time.sleep(SIM_DT)

    data, reward = [], []
    step = 0
    print(colored(f"Episode {episode_idx} started. Task: {task}", "green"))
    while True:
        t0 = time.time()
        running, left_pose, left_grip, right_pose, right_grip = controller.poll()
        if not running and step > 0:
            print(colored(f"Episode {episode_idx} ended by user.", "yellow"))
            break

        frame = {}
        frame['observation.state'] = ts['agent_pos'].copy()
        frame['observation.environment_state'] = ts['environment_state'].copy()
        frame['left_arm_pose'] = info['left_arm_pose'].copy()
        frame['right_arm_pose'] = info['right_arm_pose'].copy()
        frame['middle_arm_pose'] = info['middle_arm_pose'].copy()
        middle_pose = info['middle_arm_pose']

        ts, r, _, _, info = env.step(
            left_pose=left_pose, left_gripper=left_grip,
            right_pose=right_pose, right_gripper=right_grip,
            middle_pose=middle_pose
        )
        frame['action'] = info['action'].copy()

        data.append(frame); reward.append(r)
        env.render_viewer()

        time.sleep(max(0, SIM_DT - (time.time()-t0)))
        step += 1

    if max(reward) != env.max_reward:
        print(f"Episode {episode_idx} failed. Reward: {max(reward)}")
        return [], task, False
    return data, task, True

def confirm_episode(task, episode_idx):
    ans = input(colored(f"Episode {episode_idx} done. Save(s) / Redo(r) / Quit(q)? [s/r/q]: ", "cyan")).strip().lower()
    if ans == 'q':
        return False, True
    return (ans == 's'), False

def collect_data(args):
    num_episodes = args["num_episodes"]
    env_name     = args["env_name"]
    repo_id      = args["repo_id"]
    root         = args["root"]
    task         = args["task"]

    env = TeleopEnv(
        env_name=env_name,
        cameras={"zed_cam_left":[480,640], "zed_cam_right":[480,640]},
    )
    if env.prompts:
        if len(env.prompts) == 1:
            task = env.prompts[0]
            print(colored(f"Using default task: '{task}'", 'green'))
        assert task in env.prompts, f"Task '{task}' not in prompts: {env.prompts}"

    controller = DualJoyConController(env)

    traj_dir = os.path.join(root, 'trajectories', repo_id)
    os.makedirs(traj_dir, exist_ok=True)
    episode_idx = len(glob.glob(os.path.join(traj_dir, 'episode_*.pkl')))

    print(colored(f"Collecting {num_episodes} episodes for task '{task}' in '{env_name}'", "green"))
    try:
        while episode_idx < num_episodes:
            data, task, ok = run_episode(env, controller, task, episode_idx)
            if not ok:
                continue
            save, quit_all = confirm_episode(task, episode_idx)
            if quit_all:
                break
            if not save:
                continue
            save_path = os.path.join(traj_dir, f'episode_{episode_idx}.pkl')
            with open(save_path, 'wb') as f:
                pickle.dump({'task': task, 'data': data}, f)
            print(colored(f"Saved → {save_path}", "green"))
            episode_idx += 1
    finally:
        controller.close()

    # ===== build dataset =====
    print(colored("Building LeRobotDataset…", "cyan"))
    env = TeleopEnv(
        env_name=env_name,
        cameras={
            "zed_cam_left": [480, 640],
            "zed_cam_right": [480, 640],
            "wrist_cam_left": [480, 640],
            "wrist_cam_right": [480, 640],
            "overhead_cam": [480, 640],
            "worms_eye_cam": [480, 640],
        },
    )
    ts, info = env.reset()
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=os.path.join(root, repo_id),
        fps=round(1.0/SIM_DT),
        features={
            "observation.images.zed_cam_left":  {"dtype":"video","shape":(env.cameras['zed_cam_left'][0],  env.cameras['zed_cam_left'][1],  3),"names":["height","width","channel"]},
            "observation.images.zed_cam_right": {"dtype":"video","shape":(env.cameras['zed_cam_right'][0], env.cameras['zed_cam_right'][1], 3),"names":["height","width","channel"]},
            "observation.images.wrist_cam_left":  {"dtype":"video","shape":(env.cameras['wrist_cam_left'][0],  env.cameras['wrist_cam_left'][1],  3),"names":["height","width","channel"]},
            "observation.images.wrist_cam_right": {"dtype":"video","shape":(env.cameras['wrist_cam_right'][0], env.cameras['wrist_cam_right'][1], 3),"names":["height","width","channel"]},
            "observation.images.overhead_cam":    {"dtype":"video","shape":(env.cameras['overhead_cam'][0],    env.cameras['overhead_cam'][1],    3),"names":["height","width","channel"]},
            "observation.images.worms_eye_cam":   {"dtype":"video","shape":(env.cameras['worms_eye_cam'][0],   env.cameras['worms_eye_cam'][1],   3),"names":["height","width","channel"]},
            "observation.state":              {"dtype":"float32","shape":(21,),                                "names":None},
            "observation.environment_state":  {"dtype":"float32","shape":(ts['environment_state'].shape[0],),  "names":None},
            "action":                         {"dtype":"float32","shape":(21,),                                "names":None},
            "left_eye":  {"dtype":"float32","shape":(2,), "names":None},
            "right_eye": {"dtype":"float32","shape":(2,), "names":None},
            "left_arm_pose":  {"dtype":"float32","shape":(16,), "names":None},
            "right_arm_pose": {"dtype":"float32","shape":(16,), "names":None},
            "middle_arm_pose":{"dtype":"float32","shape":(16,), "names":None},
        },
        image_writer_threads=8,
        image_writer_processes=4,
    )

    while True:
        if dataset.num_episodes >= num_episodes:
            break
        ep = dataset.num_episodes
        pkl_path = os.path.join(traj_dir, f'episode_{ep}.pkl')
        with open(pkl_path, 'rb') as f:
            filedata = pickle.load(f)
            task = filedata['task']; data = filedata['data']

        for frame in tqdm(data, desc=f"Packing ep {ep}"):
            env.set_state(frame['observation.state'], frame['observation.environment_state'])
            ts = env.get_obs()
            f_out = { k: torch.tensor(np.array(v).reshape(-1).astype(np.float32)) for k,v in frame.items() }
            for cam in ts['pixels']:
                f_out[f'observation.images.{cam}'] = torch.from_numpy(ts['pixels'][cam].copy())
            f_out['left_eye']  = torch.tensor([0.0, 0.0], dtype=torch.float32)
            f_out['right_eye'] = torch.tensor([0.0, 0.0], dtype=torch.float32)
            f_out['task'] = task
            dataset.add_frame(f_out)
        dataset.save_episode()

    # push 可按需打开
    # dataset.push_to_hub(private=False)
    print(colored("Dataset built.", "green"))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Collect episodes with Dual Joy-Con.")
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--env_name", type=str, default="thread-needle-v1")
    parser.add_argument("--repo-id", type=str, default="yourname/aloha_joycon_demo")
    parser.add_argument("--root", type=str, default="outputs")
    parser.add_argument("--task", type=str, default="pick red cube")
    args = vars(parser.parse_args())
    collect_data(args)