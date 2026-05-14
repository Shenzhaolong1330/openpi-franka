# Dual Franka Pi0.5 训练与部署指南

## 文件结构

```
examples/dual_franka/
├── config/
│   └── cfg_franka_pi.yaml          # 推理配置文件
├── dual_franka_interface_client.py  # 双臂机器人 RPC 客户端
├── inference_pi_with_dual_franka.py # 本地推理主程序
├── remote_inference_franka.py       # 远程推理主程序
├── remote_infer_test.py             # 远程推理测试
├── benchmark_remote_infer_latency.py # 推理延迟基准测试
├── test.py                          # 策略配置测试
├── recorder.py                      # 视频录制工具
├── utils.py                         # 工具函数
└── README.md                        # 本文件

注：Policy 输入输出转换已移至 src/openpi/policies/dual_franka_policy.py
```

## 数据格式

### 数据集路径
`/vepfs-mlp2/c20250510/250303034/workspace/data/dual_franka/pick_up_and_seal_merged`

### 动作空间 (14D)

⚠️ **动作顺序为交错排列**（与数据集一致）：
```
[L_dx, R_dx, L_dy, R_dy, L_dz, R_dz, L_drx, R_drx, L_dry, R_dry, L_drz, R_drz, L_grip, R_grip]
```

即：
- `left_delta_ee_pose.x`, `right_delta_ee_pose.x`
- `left_delta_ee_pose.y`, `right_delta_ee_pose.y`
- `left_delta_ee_pose.z`, `right_delta_ee_pose.z`
- `left_delta_ee_pose.rx`, `right_delta_ee_pose.rx`
- `left_delta_ee_pose.ry`, `right_delta_ee_pose.ry`
- `left_delta_ee_pose.rz`, `right_delta_ee_pose.rz`
- `left_gripper_cmd_bin`, `right_gripper_cmd_bin`

> 🔴 **推理时注意**：模型输出的 action 是上述交错顺序，推理脚本会自动重排为分组顺序 `[L(6), R(6), L_grip, R_grip]` 后再执行。不要直接按 `action[:6]` / `action[6:12]` 解析！

### 状态空间 (30D 原始 → 28D 重排序)

原始数据集顺序 (30D, 1-based 索引):
- 1-14: 左右臂关节位置交错排列 (left_joint_1, right_joint_1, ..., left_joint_7, right_joint_7)
- 15-26: 左右臂 EE pose 交错排列 (left_ee_x, right_ee_x, ..., left_ee_rz, right_ee_rz)
- 27-28: 左右夹爪状态归一化
- 29-30: 左右夹爪命令 (excluded)

**重排序后送入模型的 28D 状态**:
```
[left_joint(7), right_joint(7), left_ee(6), right_ee(6), left_gripper_state(1), right_gripper_state(1)]
```

提取规则 (1-based → 0-based 对应):
- left_joint:  1,3,5,7,9,11,13  → 0-based: 0,2,4,6,8,10,12
- right_joint: 2,4,6,8,10,12,14 → 0-based: 1,3,5,7,9,11,13
- left_ee:     15,17,19,21,23,25 → 0-based: 14,16,18,20,22,24
- right_ee:    16,18,20,22,24,26 → 0-based: 15,17,19,21,23,25
- left_gripper:  27 → 0-based: 26
- right_gripper: 28 → 0-based: 27

这 28D 状态会被 `PadStatesAndActions` 填充到 `action_dim=32` 后送入模型。

### 图像
- `observation.images.left_wrist_image` - 左腕相机
- `observation.images.right_wrist_image` - 右腕相机
- `observation.images.head_image` - 头部/第三人称相机

## 配置说明

### 1. Policy 文件 (`src/openpi/policies/dual_franka_policy.py`)

- `DualFrankaInputs`: 将数据集格式转换为模型输入格式
  - **不再重排序 state**：OBS_INDICES 已在 data_loader.py 中完成 30D→28D 重排序，推理时机器人客户端直接提供 28D 正确顺序
  - 解析三个相机图像（head, left_wrist, right_wrist）
  - 将 state 直接传入模型（由 PadStatesAndActions 填充到 action_dim=32）

- `DualFrankaOutputs`: 将模型输出转换回 14D 动作空间

### 2. 数据配置 (`src/openpi/training/config.py`)

- `LeRobotDualFrankaDataConfig`: 数据配置类
  - 映射图像和状态键名
  - 配置 action_dim=14（数据集实际动作维度）
  - extra_delta_transform=False（动作已是 delta EE，无需额外转换）

- `pi05_droid_finetune_dual_franka`: 训练配置
  - 使用 pi05 模型
  - model.action_dim=32（匹配预训练 pi05_droid 模型，28D state 和 14D action 会被 PadStatesAndActions 填充到 32D）
  - action_horizon=20
  - LoRA 微调配置（gemma_2b_lora + gemma_300m_lora）
  - num_train_steps=100_000

### 3. 推理配置 (`examples/dual_franka/config/cfg_franka_pi.yaml`)

```yaml
model:
  name: pi05_droid_finetune_dual_franka
  checkpoint_dir: ...

cameras:
  left_wrist_cam_serial: "..."
  right_wrist_cam_serial: "..."
  exterior_cam_serial: "..."

robot:
  ip: 192.168.110.15
  port: 4242
  left_initial_pose: [...]
  right_initial_pose: [...]
  action_fps: 20
  action_horizon: 10

task:
  description: "pick up and seal the container."
```

## 使用步骤

### 1. 生成归一化参数

归一化参数（`norm_stats.json`）是训练和推理的**关键依赖**，它定义了 state 和 action 的均值/标准差，用于数据归一化。如果归一化参数与训练时不一致，模型输出将完全错误。

#### ⚠️ 重要：必须使用正确的归一化参数

归一化参数的 state 顺序必须与模型训练时看到的顺序一致（28D 重排序后）。有两种生成方式：

#### 方式 A：从数据集 stats.json 直接转换（推荐 ✅）

使用 `scripts/convert_norm_stats.py` 脚本，直接从数据集自带的 `meta/stats.json`（30D 原始顺序）通过 `OBS_INDICES` 重排生成正确的 `norm_stats.json`（28D 分组顺序）：

```bash
cd /vepfs-mlp2/c20250510/250303034/workspace/openpi-franka

python scripts/convert_norm_stats.py \
    --dataset-path /vepfs-mlp2/c20250510/250303034/workspace/data/dual_franka/pick_up_and_seal_merged
```

脚本会自动：
1. 读取 `meta/stats.json` 中的 30D 原始 state 统计信息
2. 按 `OBS_INDICES=1,3,5,7,9,11,13,2,4,6,8,10,12,14,15,17,19,21,23,25,16,18,20,22,24,26,27,28` 重排为 28D
3. 输出到 `<dataset-path>/norm_stats.json`
4. 自动验证重排序的正确性

**优点**：不依赖 π₀ 训练框架，不需要 GPU，秒级完成，且避免了 `DualFrankaInputs` 双重重排序的 bug。

#### 方式 B：使用 π₀ 框架计算

```bash
cd /vepfs-mlp2/c20250510/250303034/workspace/openpi-franka

OBS_INDICES=1,3,5,7,9,11,13,2,4,6,8,10,12,14,15,17,19,21,23,25,16,18,20,22,24,26,27,28 \
python scripts/compute_norm_stats.py --config-name pi05_droid_finetune_dual_franka
```

> ⚠️ **注意**：此方式需要 GPU，且依赖 `DualFrankaInputs` 的实现。如果 `DualFrankaInputs` 中存在重排序逻辑，会导致**双重重排序 bug**（OBS_INDICES 已重排一次，DualFrankaInputs 又重排一次），使归一化参数顺序错乱。当前版本已修复此 bug（DualFrankaInputs 不再重排序），但请务必确认。

#### 🔴 推理时必须加载正确的归一化参数

推理时，策略服务器会自动从 checkpoint 目录加载 `norm_stats.json`。**必须确保**：

1. **checkpoint 目录下存在 `norm_stats.json`**：训练完成后，归一化参数会自动保存到 checkpoint 目录。如果手动生成了新的归一化参数，需要将其复制到 checkpoint 目录：
   ```bash
   # 将生成的 norm_stats.json 复制到 checkpoint 目录
   cp /vepfs-mlp2/c20250510/250303034/workspace/data/dual_franka/pick_up_and_seal_merged/norm_stats.json \
      checkpoints/pi05_droid_finetune_dual_franka/<experiment_name>/<step>/
   ```

2. **远程推理时**，策略服务器启动命令中 `--policy.dir` 指向的目录必须包含正确的 `norm_stats.json`：
   ```bash
   uv run scripts/serve_policy.py policy:checkpoint \
       --policy.config=pi05_droid_finetune_dual_franka \
       --policy.dir=checkpoints/pi05_droid_finetune_dual_franka/<experiment_name>/<step>
   # ↑ 该目录下必须有 norm_stats.json
   ```

3. **如果归一化参数与训练时不一致**，模型输出将完全错误（动作幅度、方向、夹爪开合都会异常），表现为机器人行为完全不可控。

### 2. 训练模型

```bash
# 使用 LoRA 微调
OBS_INDICES=1,3,5,7,9,11,13,2,4,6,8,10,12,14,15,17,19,21,23,25,16,18,20,22,24,26,27,28 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 python scripts/train.py pi05_droid_finetune_dual_franka --exp_name=my_experiment --overwrite
```

### 3. 启动推理

```bash
# 1. 启动策略服务器
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_droid_finetune_dual_franka --policy.dir=checkpoints/pi05_droid_finetune_dual_franka/my_experiment/20000

# 2. 运行推理程序
cd examples/dual_franka
python inference_pi_with_dual_franka.py
```

## 关键修改总结

### 新增文件
1. `src/openpi/policies/dual_franka_policy.py` - Policy 转换
2. `examples/dual_franka/dual_franka_interface_client.py` - RPC 客户端

### 修改文件
1. `src/openpi/training/config.py`
   - 添加 `import openpi.policies.dual_franka_policy as dual_franka_policy`
   - 添加 `LeRobotDualFrankaDataConfig` 类
   - 添加 `pi05_droid_finetune_dual_franka` 训练配置

2. `examples/dual_franka/inference_pi_with_dual_franka.py`
   - 适配双臂机器人接口
   - 支持 14D 动作空间
   - 支持三个相机输入

3. `examples/dual_franka/config/cfg_franka_pi.yaml`
   - 更新为 dual franka 配置

## 注意事项

1. **相机序列号**: 需要更新配置文件中的相机序列号
2. **机器人 IP**: 确认机器人 IP 地址和端口
3. **动作缩放**: `ee_action_scale` 可能需要根据实际效果调整
4. **夹爪阈值**: `close_threshold` 控制夹爪开合判断
5. **初始位姿**: 左右臂的初始位姿需要根据实际机器人调整
6. **🔴 双重重排序 Bug 修复**：之前的 `DualFrankaInputs` 会对已被 OBS_INDICES 重排的 28D state 再次做奇偶索引提取，导致 state 顺序完全错乱（如 left_joint 变成了 `[L_j1, L_j3, L_j5, L_j7, R_j2, R_j4, R_j6]`）。当前版本已修复，`DualFrankaInputs` 直接使用 state 不再重排序。**如果之前训练过模型，必须使用修复后的代码和正确的 `norm_stats.json` 重新训练。**
7. **🔴 归一化参数一致性**：推理时加载的 `norm_stats.json` 必须与训练时完全一致。如果归一化参数的 state 顺序与模型训练时不匹配，机器人行为将完全不可控。推荐使用 `scripts/convert_norm_stats.py` 从数据集 `meta/stats.json` 生成，避免双重重排序问题。
