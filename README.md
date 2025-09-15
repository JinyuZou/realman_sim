# To install realman_jinyu
cd realman_jinyu

conda create --name realman python=3.10

conda activate realman

pip install -e .

# To record the episodes
### 1. connect to the xbox controller correctly
<img width="180" height="113" alt="image" src="https://github.com/user-attachments/assets/ac2ace4e-d7da-4be6-8ede-bb3e68768f88" />

### for the gripper
press LT to close the left gripper, press RT to close the right gripper
### there are currently two sim env : 1. put_cube 2.hook package (the original one is realman-aloha-v1)
### 2. save episodes as hdf5
cd /home/jinyu/realman_jinyu/realman_jinyu/joycon_zjy/aloha_sim_env/
#### 2.1 change the default len of each episode 
in record_hdf5_same_len.py, you can change the default len in "ap.add_argument("--frames", type=int, default=300, help="每次按B录制的帧数上限（默认300）")"
#### 2.2 change the sim Env 
in record_hdf5_same_len.py, you can change the default sim Env in ap.add_argument("--env-name", type=str, default="put-cube-v1") like ap.add_argument("--env-name", type=str, default="hook-package-v1") 

python record_hdf5_same_len.py
### 3. save episodes as lerobot dataset
this one is under development

python /home/jinyu/realman_jinyu/realman_jinyu/joycon_zjy/aloha_sim_env/collect_sim_cv2.py
