# To install realman_jinyu
cd realman_sim

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
in record_hdf5_same_len.py, you can change the default len in "ap.add_argument("--frames", type=int, default=300, help="max frames to record per B-press (default 300)")
#### 2.2 change the sim Env 
in record_hdf5_same_len.py, you can change the default sim Env in ap.add_argument("--env-name", type=str, default="put-cube-v1") like ap.add_argument("--env-name", type=str, default="hook-package-v1") 

python record_hdf5_same_len.py






### 3. Convert HDF5 → LeRobot v3.0 (File-based)
python /home/jinyu/realman_jinyu/realman_jinyu/joycon_zjy/aloha_sim_env/hdf5_to_lerobot3.py
| Command |
| ------- |
| ```bash<br>python hdf5_to_lerobot3.py \<br>  --in-dir  /path/to/episode_*.hdf5 \<br>  --root    ~/datasets/lerobot \<br>  --repo-id <user>/<dataset><br>``` |

| Parameter | Description |
|-----------|-------------|
| `in-dir`  | Local folder containing `episode_0000.hdf5` … |
| `root`    | LeRobot cache root; results are saved under `<root>/<repo-id>/` |
| `repo-id` | Hugging Face repository name; ready for `push_to_hub` |

| Output Structure (LeRobot v3.0 File-based) |
|--------------------------------------------|
| `data/chunk-000/file-000.parquet` &nbsp; multi-episode observations & actions<br>`videos/camera/chunk-000/file-000.mp4` &nbsp; consolidated video chunk<br>`meta/episodes/chunk-000/file-000.parquet` &nbsp; structured metadata |

| Note |
|------|
| • The official online visualizer currently supports **v2.1 episode-based** only; specify `version="v2.1"` in the script for immediate web visualization.<br>• Run `dataset.push_to_hub()` after conversion to publish to Hugging Face Hub. |
