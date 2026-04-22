# $\pi_{0.5}$ for Franka
In this project, I finetune $\pi_{0.5}$-DROID on my own Franka dataset, 
collected by the [lerobot_franka_isoteleop](https://github.com/Shenzhaolong1330/lerobot_franka_isoteleop.git) project. The dataset is in the format of Lerobot V3.0.

## Env Setup and Installation

### 1. Clone the repository:
```bash
git clone --recurse-submodules https://github.com/Shenzhaolong1330/openpi-franka.git

# Or if you already cloned the repo:
git submodule update --init --recursive
```

### 2. Add the dependencies you need:

You can add the dependencies you need in the `pyproject.toml` `[project] dependencies` section.
For me I need the following dependencies:
```toml
"pyrealsense2",
"zerorpc",
```

As for `lerobot`, you can add in the `[tool.uv.sources]` section.
The default version of lerobot installed by `openpi` is a very old version and the data format is not compatible with the latest version.
So we need to install a proper version of lerobot manually.
Here I chose the version `da5d2f3e9187fa4690e6667fe8b294cae49016d6` which is the version compatible with data collected by [lerobot_franka_isoteleop](https://github.com/Shenzhaolong1330/lerobot_franka_isoteleop.git).
Now as I have installed the proper version of lerobot, I can add local path of lerobot in the `[tool.uv.sources]` section.
```toml
lerobot = {path = "/path/to/lerobot"}
```

### 3. Env management

Create a conda environment for this project:
```bash
conda create -n openpi-franka python=3.11
conda activate openpi-franka
```
Install `uv` in the conda environment:
```bash
pip install uv
```
and install the dependencies in the conda environment:
```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

## Config Franka Policy

In `src/openpi/policies/franka_policy.py`, Class `FrankaInputs` and `FrankaOutputs` is defined.
The `FrankaInputs` is used to convert the inputs to the model to the expected format.
The `FrankaOutputs` is used to convert the outputs of the model to the expected format.

## Config Franka Data

In `src/openpi/training/config.py`, Class `LeRobotFrankaDataConfig` is defined.
This class is used to configure the Franka dataset in the format of Lerobot V3.0.
It will be used to process and convert the data to the expected format in the training and inference process.

## Config Franka Trainer

In `src/openpi/training/config.py`, TrainConfig for `pi05_droid_finetune_franka` is defined.
This part configures the model, dataset, and parameters for training, and other hyperparameters.
Remember to set your own datasets in the `repo_id` field, and `weight_loader` to your own checkpoint.

## `data_loader` Reorder `observation.state` and Compute the State Norm

In my own datasets, the `obs.state` is not only the concatenation of the joint angles and the gripper position, but also contains the velocity of the joints and other information.
To match the format of the pretrained models, we need to reorder the `observation.state` to be `[joint_position, gripper_position]`.

For computing the state norm, we need to set the `OBS_INDICES` environment variable to `1,2,3,4,5,6,7,9` (the indices of the joint positions and gripper position).
```bash
# OBS_INDICES is the indices of the joint positions and gripper position in the observation.state
# --config-name is the name of the config name in the config.py
OBS_INDICES=1,2,3,4,5,6,7,8 uv run scripts/compute_norm_stats.py --config-name pi05_droid_finetune_franka

# delta ee
OBS_INDICES=10,11,12,13,14,15,8 uv run scripts/compute_norm_stats.py --config-name pi05_droid_finetune_delta_ee_franka
```
The computed state norm is saved in the `norm_stats.json` file in the dataset directory.

## Finetune the Model

I collected 50 training trajectories for finetuning the model.

```bash
OBS_INDICES=1,2,3,4,5,6,7,8 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py pi05_droid_finetune_franka --exp_name=pick_and_place_robotiq_0211 --overwrite

# delta ee
OBS_INDICES=10,11,12,13,14,15,8 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 python scripts/train.py pi05_droid_finetune_delta_ee_franka --exp_name=pick_and_place_last_dance --overwrite
```

## Inference the Model
[inference_pi_with_franka.py](examples/franka/inference_pi_with_franka.py)

---

## Use Lerobot data tools

```bash
lerobot-edit-dataset \
    --repo_id milk_shake_merged \
    --operation.type merge \
    --operation.repo_ids "['milk_shake_3_19_step1_20260319_v06', 'milk_shake_3_19_step3_20260319_v06']"
```

## Use Tos
Get data from tos:
```bash
./tosutil cp -r -p 40 -j 50 -nfj 40 tos://c20250510/shenzhaolong/datasets/pick_all_objects_20260208/ workspace/data/robotiq
```

Send model to tos:
```bash
./tosutil cp -r -p 40 -j 50 -nfj 40 /vepfs-mlp2/c20250510/250303034/workspace/openpi-franka/checkpoints/pi05_droid_finetune_franka/pick_and_place_robotiq_0211/55000 tos://c20250510/shenzhaolong/datasets/model/pick_and_place_robotiq_0211/55000
```

Get model from tos:
```bash
tosutil cp -r -p 40 -j 50 -nfj 40 tos://c20250510/shenzhaolong/datasets/model/pick_and_place_robotiq_0211/55000 /home/deepcybo/.cache/openpi/openpi-assets/checkpoints/pi05_droid_finetune_franka/pick_and_place_robotiq
```
