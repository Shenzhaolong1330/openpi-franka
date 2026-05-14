#!/usr/bin/env python3
"""
Remote Inference with Local Execution for Dual Franka Robot.

This program:
1. Connects to a remote policy server for inference (via WebSocket)
2. Executes actions locally on the Dual Franka robot

Architecture:
┌─────────────────┐     WebSocket      ┌─────────────────┐
│  本地双臂机器人   │ ──────────────────►│  远程推理服务器  │
│  (执行动作)      │    观测/动作        │  (GPU 推理)     │
└─────────────────┘                    └─────────────────┘

Usage:
    python examples/dual_franka/remote_inference_franka.py --host 100.101.84.9 --port 8000

Server side (remote):
    uv run scripts/serve_policy.py --policy.config pi05_droid_finetune_dual_franka --policy.dir /path/to/checkpoint --port 8000
"""

import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")

import argparse
import yaml
import os
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any
from scipy.spatial.transform import Rotation as R

from utils import FpsCounter
from openpi_client import image_tools
from openpi_client import websocket_client_policy
from recorder import Recorder
from lerobot.cameras.configs import ColorMode, Cv2Rotation
from lerobot.cameras.realsense.camera_realsense import RealSenseCameraConfig
from lerobot.cameras import make_cameras_from_configs
from dual_franka_interface_client import DualFrankaRobotiqRpcClient

home = Path.home()


def rotvec_to_rotation_matrix(rotation_vector: np.ndarray) -> np.ndarray:
    return R.from_rotvec(rotation_vector).as_matrix()


def rotation_matrix_to_rotvec(rot_matrix: np.ndarray) -> np.ndarray:
    return R.from_matrix(rot_matrix).as_rotvec()


def apply_delta_rotation(current_rotvec: np.ndarray, delta_rotvec: np.ndarray) -> np.ndarray:
    """Apply delta rotation to current rotation using rotation matrices."""
    current_rot = rotvec_to_rotation_matrix(current_rotvec)
    delta_rot = rotvec_to_rotation_matrix(delta_rotvec)
    new_rot = delta_rot @ current_rot
    return rotation_matrix_to_rotvec(new_rot)


def update_latest_symlink(target: Path, link_name: Path):
    """Update symlink to point to the latest log."""
    if link_name.exists() or link_name.is_symlink():
        link_name.unlink()
    os.symlink(target, link_name)


class RemoteInferenceDualFranka:
    """Remote inference with local execution for Dual Franka robot."""
    
    def __init__(self, config_path: Path, server_host: str, server_port: int):
        # Load YAML config
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        # Remote server config
        self.server_host = server_host
        self.server_port = server_port
        
        # Camera config - dual franka uses 3 cameras
        cam = cfg["cameras"]
        self.left_wrist_cam_serial = cam["left_wrist_cam_serial"]
        self.right_wrist_cam_serial = cam["right_wrist_cam_serial"]
        self.exterior_cam_serial = cam["exterior_cam_serial"]
        self.cam_fps = cam.get("fps", 30)

        # Video config
        video = cfg["video"]
        self.video_fps = video.get("fps", 7)
        self.visualize = video["visualize"]

        # Robot config - dual franka has left and right arms
        robot = cfg["robot"]
        self.robot_ip = robot["ip"]
        self.robot_port = robot["port"]
        self.left_initial_pose = np.asarray(robot["left_initial_pose"], dtype=np.float32)
        self.right_initial_pose = np.asarray(robot["right_initial_pose"], dtype=np.float32)
        self.action_fps = robot["action_fps"]
        self.action_horizon = robot["action_horizon"]

        # Gripper config
        gripper = cfg["gripper"]
        self.close_threshold = gripper["close_threshold"]
        self.gripper_force = gripper["gripper_force"]
        self.gripper_speed = gripper["gripper_speed"]
        self.gripper_reverse = gripper["gripper_reverse"]

        # Action mode config - dual franka only supports delta_ee mode
        action_mode = cfg.get("action_mode", {})
        self.action_mode = action_mode.get("mode", "delta_ee")
        self.ee_action_scale = action_mode.get("ee_action_scale", 1.0)

        # Task config
        task = cfg["task"]
        self.task_description = task["description"]
        
        # Setup directories
        time_str = time.strftime('%Y%m%d-%H%M%S')
        time_path = time.strftime('%Y%m%d')
        base_dir = Path(__file__).parent
        log_dir = base_dir / "logs"
        video_dir = base_dir / "videos" / time_path

        (log_dir / "all_logs").mkdir(parents=True, exist_ok=True)
        video_dir.mkdir(parents=True, exist_ok=True)

        latest_path = log_dir / "latest.yaml"
        log_path = log_dir / "all_logs" / f"log_{time_str}.yaml"

        # 3 video paths for dual franka
        left_wrist_video = video_dir / f"{self.task_description.replace(' ', '_')}_left_wrist_{time_str}.mp4"
        right_wrist_video = video_dir / f"{self.task_description.replace(' ', '_')}_right_wrist_{time_str}.mp4"
        exterior_video = video_dir / f"{self.task_description.replace(' ', '_')}_exterior_{time_str}.mp4"

        self.recorder = Recorder(
            log_path=log_path, 
            video_path=[left_wrist_video, right_wrist_video, exterior_video], 
            display_fps=self.video_fps, 
            visualize=self.visualize
        )
        update_latest_symlink(log_path, latest_path)

        self.fps_action = FpsCounter(name="action")
        self.robot_client = None
        self.cameras = None
        self.policy_client = None

    # --------------------------- CONNECTIONS --------------------------- #
    def connect_robot(self):
        """Connect to Dual Franka robot."""
        try:
            logging.info("\n===== [ROBOT] Connecting to Dual Franka robot =====")
            self.robot_client = DualFrankaRobotiqRpcClient(ip=self.robot_ip, port=self.robot_port)
            self.robot_client.gripper_initialize()

            obs = self.robot_client.get_observation()
            left_arm = obs.get('left_arm', {})
            right_arm = obs.get('right_arm', {})

            left_joints = left_arm.get('joint_position', [])
            if left_joints and len(left_joints) == 7:
                logging.info(f"[ROBOT] Left arm joint positions: {[round(j, 4) for j in left_joints]}")

            left_tcp = left_arm.get('ee_pose', [])
            if left_tcp and len(left_tcp) == 6:
                logging.info(f"[ROBOT] Left arm TCP pose: {[round(p, 4) for p in left_tcp]}")

            right_joints = right_arm.get('joint_position', [])
            if right_joints and len(right_joints) == 7:
                logging.info(f"[ROBOT] Right arm joint positions: {[round(j, 4) for j in right_joints]}")

            right_tcp = right_arm.get('ee_pose', [])
            if right_tcp and len(right_tcp) == 6:
                logging.info(f"[ROBOT] Right arm TCP pose: {[round(p, 4) for p in right_tcp]}")

            logging.info("===== [ROBOT] Dual Franka initialized successfully =====\n")
        except Exception as e:
            logging.error(f"[ERROR] Failed to connect to Dual Franka robot: {e}")
            exit(1)

    def connect_cameras(self):
        """Initialize and connect RealSense cameras for dual franka (3 cameras)."""
        try:
            logging.info("\n===== [CAM] Initializing cameras =====")
            
            left_wrist_cfg = RealSenseCameraConfig(
                serial_number_or_name=self.left_wrist_cam_serial,
                fps=self.cam_fps,
                width=640,
                height=480,
                color_mode=ColorMode.RGB,
                use_depth=False,
                rotation=Cv2Rotation.NO_ROTATION,
            )
            right_wrist_cfg = RealSenseCameraConfig(
                serial_number_or_name=self.right_wrist_cam_serial,
                fps=self.cam_fps,
                width=640,
                height=480,
                color_mode=ColorMode.RGB,
                use_depth=False,
                rotation=Cv2Rotation.NO_ROTATION,
            )
            exterior_cfg = RealSenseCameraConfig(
                serial_number_or_name=self.exterior_cam_serial,
                fps=self.cam_fps,
                width=640,
                height=480,
                color_mode=ColorMode.RGB,
                use_depth=False,
                rotation=Cv2Rotation.NO_ROTATION,
            )

            camera_config = {
                "left_wrist_image": left_wrist_cfg,
                "right_wrist_image": right_wrist_cfg,
                "exterior_image": exterior_cfg,
            }
            self.cameras = make_cameras_from_configs(camera_config)

            for name, cam in self.cameras.items():
                cam.connect()
                logging.info(f"[CAM] {name} connected successfully.")
            logging.info("===== [CAM] Cameras initialized successfully =====\n")
        except Exception as e:
            logging.error(f"[ERROR] Failed to initialize cameras: {e}")
            self.cameras = None

    def connect_policy_server(self):
        """Connect to remote policy server via WebSocket."""
        logging.info(f"\n===== [REMOTE] Connecting to policy server at {self.server_host}:{self.server_port} =====")
        try:
            self.policy_client = websocket_client_policy.WebsocketClientPolicy(
                host=self.server_host,
                port=self.server_port
            )
            metadata = self.policy_client.get_server_metadata()
            logging.info(f"[REMOTE] Connected to policy server")
            logging.info(f"[REMOTE] Server metadata: {metadata}")
            logging.info("===== [REMOTE] Policy server connected successfully =====\n")
        except Exception as e:
            logging.error(f"[ERROR] Failed to connect to policy server: {e}")
            exit(1)

    # --------------------------- OBSERVATION --------------------------- #
    def _transfer_obs_dual_franka_state(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Transfer observation for dual franka mode.

        Build 28D state from robot observations in the same order as training:
        [left_joint(7), right_joint(7), left_ee(6), right_ee(6), left_gripper(1), right_gripper(1)]
        """
        left_joint = np.asarray(obs["left_joint_positions"], dtype=np.float32)
        right_joint = np.asarray(obs["right_joint_positions"], dtype=np.float32)
        left_ee = np.asarray(obs["left_ee_pose"], dtype=np.float32)
        right_ee = np.asarray(obs["right_ee_pose"], dtype=np.float32)
        left_gripper = np.asarray([obs["left_gripper_position"]], dtype=np.float32)
        right_gripper = np.asarray([obs["right_gripper_position"]], dtype=np.float32)

        state = np.concatenate([
            left_joint,      # 7D
            right_joint,     # 7D
            left_ee,         # 6D
            right_ee,        # 6D
            left_gripper,    # 1D
            right_gripper,   # 1D
        ])  # Total: 28D

        return {
            "observation/state": state,
            "observation/image": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(obs["exterior_image"], 224, 224)
            ),
            "observation/wrist_image": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(obs["left_wrist_image"], 224, 224)
            ),
            "observation/right_wrist_image": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(obs["right_wrist_image"], 224, 224)
            ),
            "prompt": obs["prompt"],
        }

    def get_obs_state(self) -> Dict[str, Any]:
        """Get current observation from dual franka robot and cameras."""
        obs = {}

        if self.robot_client:
            full_obs = self.robot_client.get_observation()
            left_arm = full_obs.get('left_arm', {})
            right_arm = full_obs.get('right_arm', {})

            obs["left_joint_positions"] = np.array(left_arm.get('joint_position', [0.0] * 7))
            obs["right_joint_positions"] = np.array(right_arm.get('joint_position', [0.0] * 7))
            obs["left_ee_pose"] = np.array(left_arm.get('ee_pose', [0.0] * 6))
            obs["right_ee_pose"] = np.array(right_arm.get('ee_pose', [0.0] * 6))

        if self.cameras:
            for name, cam in self.cameras.items():
                obs[name] = cam.read()

        if self.task_description:
            obs["prompt"] = self.task_description

        if self.robot_client:
            full_obs = self.robot_client.get_observation()
            left_arm = full_obs.get('left_arm', {})
            right_arm = full_obs.get('right_arm', {})

            left_gripper = left_arm.get('gripper', {})
            right_gripper = right_arm.get('gripper', {})

            left_width = left_gripper.get('width', 0.0) if isinstance(left_gripper, dict) else 0.0
            right_width = right_gripper.get('width', 0.0) if isinstance(right_gripper, dict) else 0.0

            left_gripper_state = max(0.0, min(1.0, left_width / 0.0801))
            right_gripper_state = max(0.0, min(1.0, right_width / 0.0801))

            obs["left_gripper_position"] = 0.0 if left_gripper_state < self.close_threshold else 1.0
            obs["right_gripper_position"] = 0.0 if right_gripper_state < self.close_threshold else 1.0

        return self._transfer_obs_dual_franka_state(obs)

    # --------------------------- ACTION EXECUTION --------------------------- #
    def execute_actions(self, actions: np.ndarray, block: bool = False):
        """Execute actions on the dual franka robot.

        Dual Franka action format (14D):
        - left_delta_ee (6D): dx, dy, dz, drx, dry, drz
        - right_delta_ee (6D): dx, dy, dz, drx, dry, drz
        - left_gripper (1D)
        - right_gripper (1D)
        """
        if self.robot_client is None:
            logging.error("[ERROR] Robot not connected")
            return
        
        action_dim = actions.shape[-1] if len(actions.shape) > 1 else len(actions)
        if action_dim != 14:
            logging.error(f"[ERROR] Expected 14D action for dual franka, got {action_dim}D")
            return

        self._execute_dual_delta_ee_actions(actions, block)

    def _execute_dual_delta_ee_actions(self, actions: np.ndarray, block: bool = False):
        """Execute delta end-effector actions for dual franka.

        The model outputs actions in the same interleaved order as the dataset:
        [L_dx, R_dx, L_dy, R_dy, L_dz, R_dz, L_drx, R_drx, L_dry, R_dry, L_drz, R_drz, L_grip, R_grip]
        We reorder to grouped order for execution:
        [L_dx, L_dy, L_dz, L_drx, L_dry, L_drz, R_dx, R_dy, R_dz, R_drx, R_dry, R_drz, L_grip, R_grip]
        """
        if block:
            logging.info("[STATE] Block mode not implemented for dual franka delta actions")
            return

        # Get current observation for both arms
        full_obs = self.robot_client.get_observation()
        left_arm = full_obs.get('left_arm', {})
        right_arm = full_obs.get('right_arm', {})

        left_ee_pose = np.array(left_arm.get('ee_pose', [0.0] * 6))
        right_ee_pose = np.array(right_arm.get('ee_pose', [0.0] * 6))

        left_pos = left_ee_pose[:3].copy()
        left_rotvec = left_ee_pose[3:6].copy()
        right_pos = right_ee_pose[:3].copy()
        right_rotvec = right_ee_pose[3:6].copy()

        for i, action in enumerate(actions[:self.action_horizon]):
            start_time = time.perf_counter()

            # Reorder from interleaved [L_dx, R_dx, L_dy, R_dy, ...] to grouped [L(6), R(6)]
            # Interleaved indices (0-based): L=0,2,4,6,8,10  R=1,3,5,7,9,11
            left_delta = np.array([
                action[0], action[2], action[4],   # L_dx, L_dy, L_dz
                action[6], action[8], action[10],  # L_drx, L_dry, L_drz
            ]) * self.ee_action_scale
            right_delta = np.array([
                action[1], action[3], action[5],   # R_dx, R_dy, R_dz
                action[7], action[9], action[11],  # R_drx, R_dry, R_drz
            ]) * self.ee_action_scale
            left_gripper_cmd = action[12]
            right_gripper_cmd = action[13]

            # Apply delta to left arm
            left_delta_pos = left_delta[:3]
            left_delta_rotvec = left_delta[3:6]
            left_target_pos = left_pos + left_delta_pos
            left_target_rotvec = apply_delta_rotation(left_rotvec, left_delta_rotvec)

            # Apply delta to right arm
            right_delta_pos = right_delta[:3]
            right_delta_rotvec = right_delta[3:6]
            right_target_pos = right_pos + right_delta_pos
            right_target_rotvec = apply_delta_rotation(right_rotvec, right_delta_rotvec)

            # Send dual arm command
            self.robot_client.dual_robot_move_to_ee_pose(
                left_delta=np.concatenate([left_target_pos, left_target_rotvec]),
                right_delta=np.concatenate([right_target_pos, right_target_rotvec]),
                delta=False,  # We're sending absolute poses
                wait=False,
            )

            # Update current poses
            full_obs = self.robot_client.get_observation()
            left_arm = full_obs.get('left_arm', {})
            right_arm = full_obs.get('right_arm', {})
            left_ee_pose = np.array(left_arm.get('ee_pose', [0.0] * 6))
            right_ee_pose = np.array(right_arm.get('ee_pose', [0.0] * 6))
            left_pos = left_ee_pose[:3].copy()
            left_rotvec = left_ee_pose[3:6].copy()
            right_pos = right_ee_pose[:3].copy()
            right_rotvec = right_ee_pose[3:6].copy()

            # Control grippers
            left_gripper_cmd_val = 0.0 if left_gripper_cmd < self.close_threshold else 1.0
            right_gripper_cmd_val = 0.0 if right_gripper_cmd < self.close_threshold else 1.0

            if self.gripper_reverse:
                left_gripper_cmd_val = 1.0 - left_gripper_cmd_val
                right_gripper_cmd_val = 1.0 - right_gripper_cmd_val

            self.robot_client.set_left_gripper(left_gripper_cmd_val)
            self.robot_client.set_right_gripper(right_gripper_cmd_val)

            elapsed = time.perf_counter() - start_time
            to_sleep = 1.0 / self.action_fps - elapsed
            if to_sleep > 0:
                time.sleep(to_sleep)
            self.fps_action.update()

    # --------------------------- MAIN PIPELINE --------------------------- #
    def run(self):
        """Main pipeline: connect everything and run inference loop."""
        logging.info("=" * 60)
        logging.info("Remote Inference with Local Execution - Dual Franka")
        logging.info("=" * 60)
        
        # Connect all components
        self.connect_robot()
        self.connect_cameras()
        self.connect_policy_server()
        
        # Move both arms to initial pose
        logging.info("[STATE] Moving arms to initial pose...")
        self.robot_client.go_home('both')
        self.robot_client.open_gripper('left_arm')
        self.robot_client.open_gripper('right_arm')
        
        # Warmup
        obs = self.get_obs_state()
        logging.info(f"[STATE] Observation keys: {obs.keys()}")
        logging.info("Warming up remote inference...")
        start = time.time()
        self.policy_client.infer(obs)
        logging.info(f"Warmup completed in {time.time() - start:.2f}s")
        
        # Main inference loop
        infer_time = 1
        logging.info("=" * 60)
        logging.info("Starting Inference Loop")
        logging.info("=" * 60)
        
        try:
            while True:
                loop_start = time.perf_counter()
                
                # Get observation
                obs = self.get_obs_state()
                
                # Remote inference
                infer_start = time.perf_counter()
                result = self.policy_client.infer(obs)
                infer_time_ms = (time.perf_counter() - infer_start) * 1000
                
                # Execute actions locally
                self.execute_actions(result["actions"])
                
                # Record
                self.recorder.submit_actions(result["actions"][:self.action_horizon], infer_time, obs["prompt"])
                self.recorder.submit_obs(obs)
                
                loop_time = time.perf_counter() - loop_start
                logging.info(f"[LOOP {infer_time}] Inference: {infer_time_ms:.1f}ms, Total: {loop_time*1000:.1f}ms, Rate: {1/loop_time:.1f}Hz")
                
                # Log server timing if available
                if "server_timing" in result:
                    server_timing = result["server_timing"]
                    logging.info(f"    Server infer: {server_timing.get('infer_ms', 0):.1f}ms")
                
                infer_time += 1
                
        except KeyboardInterrupt:
            logging.info("\n[INFO] KeyboardInterrupt detected. Stopping...")
            self.robot_client.go_home('both')
            self.robot_client.open_gripper('left_arm')
            self.robot_client.open_gripper('right_arm')
            
        except Exception as e:
            logging.error(f"[ERROR] Inference loop error: {e}")
            self.robot_client.go_home('both')
            self.robot_client.open_gripper('left_arm')
            self.robot_client.open_gripper('right_arm')
            raise

        # Cleanup
        try:
            ans = input("Save recorded videos? [Y/n]: ").strip().lower()
            if ans in ("", "y", "yes"):
                logging.info("[INFO] Saving videos...")
                self.recorder.save_video()
        except Exception as e:
            logging.error(f"[ERROR] Failed to save videos: {e}")


def main():
    parser = argparse.ArgumentParser(description="Remote inference with local execution for Dual Franka")
    parser.add_argument("--host", type=str, required=True, help="Remote policy server host")
    parser.add_argument("--port", type=int, default=8000, help="Remote policy server port")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    args = parser.parse_args()
    
    config_path = Path(args.config) if args.config else Path(__file__).parent / "config" / "cfg_franka_pi.yaml"
    
    inference = RemoteInferenceDualFranka(
        config_path=config_path,
        server_host=args.host,
        server_port=args.port
    )
    inference.run()


if __name__ == "__main__":
    main()
