import pathlib
import os
print("sim_config loaded")
# task parameters
XML_DIR = os.path.join(str(pathlib.Path(__file__).parent.resolve()), '../mjcf')

# CAMERAS = {
#     "zed_cam_left": [480, 640],
#     "zed_cam_right": [480, 640],
#     "wrist_cam_left": [480, 640],
#     "wrist_cam_right": [480, 640],
#     "overhead_cam": [480, 640],
#     "worms_eye_cam": [480, 640],
# }

CAMERAS = {
    "wrist_cam_left": [480, 640],
    "wrist_cam_right": [480, 640],
    "top_cam": [480, 640],
    "button_cam": [480, 640],
}

RENDER_CAMERA = "top_cam"

# physics parameters
SIM_PHYSICS_DT = 0.002
SIM_DT = 0.04
SIM_PHYSICS_ENV_STEP_RATIO = int(SIM_DT/SIM_PHYSICS_DT)
SIM_DT = SIM_PHYSICS_DT * SIM_PHYSICS_ENV_STEP_RATIO

# robot parameters
AV_STATE_DIM = 21
STATE_DIM = 14
AV_ACTION_DIM = 21
ACTION_DIM = 14

# LEFT_JOINT_NAMES = [
#     "left_waist",
#     "left_shoulder",
#     "left_elbow",
#     "left_forearm_roll",
#     "left_wrist_angle",
#     "left_wrist_rotate",
# ]

LEFT_JOINT_NAMES = [
    "joint1_left",
    "joint2_left",
    "joint3_left",
    "joint4_left",
    "joint5_left",
    "joint6_left",
    
]

# 右臂 7 个关节
RIGHT_JOINT_NAMES = [
    "joint1_right",
    "joint2_right",
    "joint3_right",
    "joint4_right",
    "joint5_right",
    "joint6_right",
    
]

LEFT_GRIPPER_JOINT_NAMES = ["Joint_finger2_left", "Joint_finger2_left"]

RIGHT_GRIPPER_JOINT_NAMES = ["Joint_finger1_right", "Joint_finger2_right"]

LEFT_ACTUATOR_NAMES = [
    "joint1_left_act",
    "joint2_left_act",
    "joint3_left_act",
    "joint4_left_act",
    "joint5_left_act",
    "joint6_left_act",
]
LEFT_GRIPPER_ACTUATOR_NAME = "act_grip_left"
RIGHT_ACTUATOR_NAMES = [
    "joint1_right_act",
    "joint2_right_act",
    "joint3_right_act",
    "joint4_right_act",
    "joint5_right_act",
    "joint6_right_act",
]
RIGHT_GRIPPER_ACTUATOR_NAME = "act_grip_right"

LEFT_EEF_SITE = "gripper_tip_left"
RIGHT_EEF_SITE = "gripper_tip_right"




LIGHT_NAME = "light"
