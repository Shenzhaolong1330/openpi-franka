#!/usr/bin/env python3
import os, time, uuid, yaml
from typing import Dict, Any, List

import cv2
import numpy as np
import pyrealsense2 as rs
from PIL import Image

from openpi.training import config as openpi_config
from openpi.policies import policy_config
from openpi_client import image_tools
from franka_interface_client import FrankaInterfaceClient

# ========= YAML 配置文件读取 =========
def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载 YAML 配置文件
    """
    print(f"[配置加载] 读取配置文件: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print(f"[配置加载] 配置读取完成")
    return config

# 生成默认配置文件
def generate_default_config(config_path: str):
    """
    生成默认配置文件
    """
    default_config = {
        "robot": {
            "ip": "192.168.1.104",
            "port": 4242,
            "damping": 0.067,
            "speed": 0.01,
            "action_rate_hz": 15
        },
        "model": {
            "checkpoint_path": "/home/deepcybo/.cache/openpi/openpi-assets/checkpoints/pi05_droid",
            "infer_times": 50,
            "chunk_steps": 8
        },
        "task": {
            "prompt": "take a tissue out and place it on the table"
        }
    }
    
    print(f"[配置生成] 创建默认配置文件: {config_path}")
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)
    print(f"[配置生成] 默认配置文件已创建")
    return default_config

print("[初始化] 开始系统组件初始化...")
start_time = time.time()

# ========= RealSense =========
def start_pipeline(serial: str):
    print(f"[相机初始化] 启动相机 {serial}...")
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(cfg)
    print(f"[相机初始化] 相机 {serial} 启动成功")
    return pipeline

# 初始化两台相机（wrist/front）
print("[相机初始化] 开始初始化 RealSense 相机...")
ctx = rs.context()
devices = list(ctx.query_devices())
if len(devices) < 2:
    raise RuntimeError(f"需要至少 2 台 RealSense D455，相机发现数={len(devices)}")
print(f"[相机初始化] 发现 {len(devices)} 台 RealSense 设备")
serials = [d.get_info(rs.camera_info.serial_number) for d in devices[:2]]
print(f"[相机初始化] 选择相机: {serials[0]} (wrist), {serials[1]} (front)")

# 初始化手腕相机
print("[相机初始化] 初始化手腕相机...")
pipeline_start_time = time.time()
pipe_wrist = start_pipeline(serials[0])
print(f"[相机初始化] 手腕相机初始化完成，耗时: {time.time() - pipeline_start_time:.2f}秒")

# 初始化前置相机
print("[相机初始化] 初始化前置相机...")
pipeline_start_time = time.time()
pipe_front = start_pipeline(serials[1])
print(f"[相机初始化] 前置相机初始化完成，耗时: {time.time() - pipeline_start_time:.2f}秒")

print("[相机初始化] 所有相机初始化成功")

def get_latest_rgb():
    f1 = pipe_wrist.wait_for_frames()
    f2 = pipe_front.wait_for_frames()
    c1, c2 = f1.get_color_frame(), f2.get_color_frame()
    if not c1 or not c2: return None, None
    img1 = np.asanyarray(c1.get_data())  # BGR
    img2 = np.asanyarray(c2.get_data())
    pil1 = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    pil2 = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    wrist_image = pil2
    front_image = pil1
    return front_image, wrist_image


# ========= pi05 policy =========
def load_model(checkpoint_path: str):
    """
    加载 pi05 模型
    """
    print("[模型加载] 开始加载模型...")
    model_start_time = time.time()
    print(f"[模型加载] 检查点路径: {checkpoint_path}")
    print("[模型加载] 获取配置...")
    cfg = openpi_config.get_config("pi05_droid")
    print("[模型加载] 创建训练策略...")
    policy = policy_config.create_trained_policy(cfg, checkpoint_path)
    model_load_time = time.time() - model_start_time
    print(f"[模型加载] 模型加载完成，耗时: {model_load_time:.2f}秒")
    return policy

# ========= Franka 客户端初始化 =========
def init_franka_client(ip: str, port: int):
    """
    初始化 Franka 客户端
    """
    print(f"[初始化] 连接 Franka 机器人 (IP: {ip}, Port: {port})...")
    franka_client = FrankaInterfaceClient(ip=ip, port=port)
    franka_client.gripper_initialize()
    franka_client.robot_start_joint_impedance_control()
    print("[初始化] Franka 机器人连接成功")
    return franka_client

# ========= Franka 控制函数 =========
def get_franka_state(franka_client: FrankaInterfaceClient) -> np.ndarray:
    """
    获取 Franka 机器人状态，返回 14 维状态向量
    [gripper_position, joint1-7_positions, joint1-7_velocities]
    """
    joint_positions = franka_client.robot_get_joint_positions()  # (7,)
    print(f"[状态获取] 当前关节位置: {joint_positions}")
    ee_pose = franka_client.robot_get_ee_pose()
    print(f"[状态获取] 当前末端执行器位姿: {ee_pose}")
    gripper_width = franka_client.gripper_get_state()["width"]
    print(f"[状态获取] 当前夹爪宽度: {gripper_width}")
    gripper_state = max(0.0, min(1.0, gripper_width/0.0801))
    print(f"[状态获取] 当前夹爪状态: {gripper_state}")
    # 构造 14 维状态向量
    state14 = np.concatenate([np.array([gripper_state]), joint_positions, ee_pose])

    print(f"[状态获取] 14维状态向量: {state14}")
    return state14

def exec_franka_chunk(franka_client: FrankaInterfaceClient, actions: np.ndarray, damping: float, action_rate_hz: float) -> Dict[str, Any]:
    """
    执行动作序列控制 Franka 机器人
    actions: float32 (n,8)，前7维是关节调整量，第8维是夹爪控制
    """
    try:
        # 获取当前关节位置
        current_joints = franka_client.robot_get_joint_positions()
        
        for i, action in enumerate(actions):
            # 计算目标关节位置（相对调整）
            target_joints = current_joints + action[:7] * damping
            
            # 执行关节位置控制
            print(f"[执行动作 {i+1}/{len(actions)}] 关节目标位置: {target_joints}")
            # franka_client.robot_move_to_joint_positions(
            #     positions=target_joints,
            #     time_to_go=1.0/action_rate_hz,  # 根据动作频率计算执行时间
            #     delta=False
            # )
            franka_client.robot_update_desired_joint_positions(target_joints)
            print(f"[执行动作 {i+1}/{len(actions)}] 夹爪目标宽度: {action[7]*0.0801}")
            # gripper_width = action[7]*0.0801
            gripper_command = 0 if action[7] < 0.7 else 1
            franka_client.gripper_goto(width=gripper_command, speed=0.1, force=10.0)
            # 更新当前关节位置
            current_joints = target_joints
            
            # 控制夹爪（每几步执行一次）
            # if i % 2 == 0:
            #     gripper_width = max(0.0, min(0.08, action[7]))  # 夹爪宽度范围：0-0.08m
            #     franka_client.gripper_goto(width=gripper_width, speed=0.1, force=10.0)
            
            # 等待下一个动作
            time.sleep(1.0/action_rate_hz)
        
        return {"ok": True, "message": "动作执行完成"}
    except Exception as e:
        print(f"[执行动作失败] {e}")
        return {"ok": False, "message": str(e)}

def build_pi0_example(front_img: Image.Image, wrist_img: Image.Image, state14: np.ndarray, prompt: str) -> Dict[str, Any]:
    return {
        "observation/exterior_image_1_left": image_tools.resize_with_pad(np.array(front_img), 224, 224),
        "observation/wrist_image_left":      image_tools.resize_with_pad(np.array(wrist_img), 224, 224),
        "observation/joint_position":        state14[1:8],  # Franka 关节位置
        "observation/gripper_position":      state14[0],  # 夹爪位置
        "prompt":                            prompt,
    }

def stream_closed_loop_chunks(
    franka_client: FrankaInterfaceClient,
    policy,
    prompt: str,
    infer_times: int,
    chunk_steps: int,
    action_rate_hz: float,
    damping: float
) -> Dict[str, Any]:
    """
    闭环控制流程：
      get_state → pi0 推理 → 执行动作序列 → 循环
    """
    task_id = str(uuid.uuid4())
    success = 0
    timeouts = 0
    rtts: List[float] = []
    t0 = time.perf_counter()
    inference_start_time = time.time()
    
    print(f"[推理任务] 开始新任务 (ID: {task_id})")
    print(f"[推理任务] 参数: infer_times={infer_times}, chunk_steps={chunk_steps}, step_rate_hz={action_rate_hz}")

    while success < infer_times:
        # 1) 抓图
        frame_time = time.time()
        front_img, wrist_img = get_latest_rgb()
        if front_img is None:
            continue
        print(f"[推理步骤 {success+1}/{infer_times}] 图像获取完成，耗时: {time.time() - frame_time:.2f}秒")

        # 2) 取状态
        state_time = time.time()
        try:
            state14 = get_franka_state(franka_client)
            print(f"[推理步骤 {success+1}/{infer_times}] 状态获取完成，耗时: {time.time() - state_time:.2f}秒")
        except Exception as e:
            timeouts += 1
            print(f"[推理步骤 {success+1}/{infer_times}] 状态获取失败: {e}")
            continue

        # 3) 推理 → 取前 n 步
        inference_time = time.time()
        example = build_pi0_example(front_img, wrist_img, state14, prompt)
        print(f"[推理步骤 {success+1}/{infer_times}] 准备推理输入...")
        act_chunk = policy.infer(example)["actions"]          # (H,8)
        inference_duration = time.time() - inference_time
        print(f"[推理步骤 {success+1}/{infer_times}] 模型推理完成，耗时: {inference_duration:.2f}秒")
        
        if not isinstance(act_chunk, np.ndarray):
            act_chunk = np.asarray(act_chunk)
        if act_chunk.ndim != 2 or act_chunk.shape[1] != 8:
            raise RuntimeError(f"pi0 返回非法形状: {act_chunk.shape}")
        n = min(int(chunk_steps), int(act_chunk.shape[0]))
        actions_to_send = act_chunk[:n].astype(np.float32, copy=False)  # (n,8)

        # 显示进度
        progress_percent = (success / infer_times) * 100 if infer_times > 0 else 0
        elapsed_time = time.time() - inference_start_time
        if success > 0:
            avg_time_per_step = elapsed_time / success
            remaining_time = avg_time_per_step * (infer_times - success)
            remaining_str = f", 预计剩余: {remaining_time:.1f}秒"
        else:
            remaining_str = ""
        
        print(f"[进度] 已完成 {success}/{infer_times} ({progress_percent:.1f}%){remaining_str}")
        
        print(f"[推理步骤 {success+1}/{infer_times}] 生成 {n} 步动作序列")
        for i in range(n):
            action_list_3f = [f"{x:.3f}" for x in actions_to_send[i].tolist()]
            print(f"[inference step {success+1}/{infer_times} chunk {i+1}/{n}] actions: {action_list_3f}")

        # 4) 执行动作序列
        exec_time = time.time()
        try:
            print(f"[推理步骤 {success+1}/{infer_times}] 执行动作序列...")
            rep = exec_franka_chunk(franka_client, actions_to_send, damping, action_rate_hz)
            exec_duration = time.time() - exec_time
            print(f"[推理步骤 {success+1}/{infer_times}] 动作执行完成，耗时: {exec_duration:.2f}秒")
            
            rtts.append((time.perf_counter() - t0) * 1000.0)
            if rep.get("ok", False):
                success += 1
                print(f"[stream] 动作执行成功 ({success}/{infer_times})")
            else:
                print(f"[stream] 动作执行失败: {rep}")
        except Exception as e:
            timeouts += 1
            print(f"[推理步骤 {success+1}/{infer_times}] 执行异常: {e}")
            continue

    # 显示最终完成信息
    total_time = time.time() - inference_start_time
    print(f"[任务完成] 完成 {infer_times} 次推理，总耗时: {total_time:.1f}秒，平均每步: {(total_time/infer_times):.2f}秒")
    print(f"[任务完成] 失败次数: {timeouts}")

    elapsed = time.perf_counter() - t0
    return {
        "task_id": task_id,
        "model_infer_times": infer_times,
        "chunk_steps": chunk_steps,
        "step_rate_hz": action_rate_hz,
        "timeouts": timeouts,
        "elapsed_s": elapsed,
        "avg_roundtrip_ms": (float(np.mean(rtts)) if rtts else None),
    }

def main(config_path: str = "config.yaml"):
    """
    主函数
    """
    # 加载配置文件，如果不存在则生成默认配置
    if not os.path.exists(config_path):
        config = generate_default_config(config_path)
    else:
        config = load_config(config_path)
    
    # 提取配置参数
    robot_config = config.get("robot", {})
    model_config = config.get("model", {})
    task_config = config.get("task", {})
    
    # 加载模型
    policy = load_model(model_config.get("checkpoint_path"))
    
    # 初始化 Franka 客户端
    franka_client = init_franka_client(robot_config.get("ip"), robot_config.get("port"))
    
    print(f"[初始化] 系统组件初始化完成，总耗时: {time.time() - start_time:.2f}秒")
    print(f"[系统就绪] 开始执行任务...")
    
    # 执行闭环控制任务
    stats = stream_closed_loop_chunks(
        franka_client=franka_client,
        policy=policy,
        prompt=task_config.get("prompt"),
        infer_times=model_config.get("infer_times"),
        chunk_steps=model_config.get("chunk_steps"),
        action_rate_hz=robot_config.get("action_rate_hz"),
        damping=robot_config.get("damping")
    )
    
    # 显示任务统计信息
    print("\n[任务统计]")
    print(f"任务ID: {stats['task_id']}")
    print(f"模型推理次数: {stats['model_infer_times']}")
    print(f"每次推理发送步数: {stats['chunk_steps']}")
    print(f"动作频率: {stats['step_rate_hz']} Hz")
    print(f"超时次数: {stats['timeouts']}")
    print(f"总耗时: {stats['elapsed_s']:.2f} 秒")
    if stats['avg_roundtrip_ms']:
        print(f"平均往返时间: {stats['avg_roundtrip_ms']:.2f} ms")

if __name__ == "__main__":
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Pi05 with Franka robot control")
    parser.add_argument(
        "--config", 
        type=str, 
        default="examples/franka/config.yaml", 
        help="YAML configuration file path"
    )
    args = parser.parse_args()
    
    # 执行主程序
    main(args.config)