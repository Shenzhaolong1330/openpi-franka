#!/usr/bin/env python3
"""
Remote Inference with Local Execution for Franka Robot.

This program:
1. Connects to a remote policy server for inference (via WebSocket)
2. Executes actions locally on the Franka robot

Architecture:
┌─────────────────┐     WebSocket      ┌─────────────────┐
│  本地机器人      │ ──────────────────►│  远程推理服务器  │
│  (执行动作)      │    观测/动作        │  (GPU 推理)     │
└─────────────────┘                    └─────────────────┘

Usage:
    python examples/franka/remote_inference_franka.py --host 100.101.84.9 --port 8000

Server side (remote):
    uv run scripts/serve_policy.py --policy.config pi05_droid_finetune_delta_ee_franka --policy.dir /path/to/checkpoint --port 8000
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
from franka_interface_client import FrankaInterfaceClient

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


class RemoteInferenceFranka:
    """Remote inference with local execution for Franka robot."""
    
    def __init__(self, config_path: Path, server_host: str, server_port: int):
        # Load YAML config
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        # Remote server config
        self.server_host = server_host
        self.server_port = server_port
        
        # Camera config
        cam = cfg["cameras"]
        self.wrist_cam_serial = cam["wrist_cam_serial"]
        self.exterior_cam_serial = cam["exterior_cam_serial"]
        self.cam_fps = cam.get("fps", 30)

        # Video config
        video = cfg["video"]
        self.video_fps = video.get("fps", 7)
        self.visualize = video["visualize"]

        # Robot config
        robot = cfg["robot"]
        self.robot_ip = robot["ip"]
        self.robot_port = robot["port"]
        self.initial_pose = np.asarray(robot["initial_pose"], dtype=np.float32)
        self.action_fps = robot["action_fps"]
        self.action_horizon = robot["action_horizon"]

        # Gripper config
        gripper = cfg["gripper"]
        self.close_threshold = gripper["close_threshold"]
        self.gripper_force = gripper["gripper_force"]
        self.gripper_speed = gripper["gripper_speed"]
        self.gripper_reverse = gripper["gripper_reverse"]

        # Action mode config
        action_mode = cfg.get("action_mode", {})
        self.action_mode = action_mode.get("mode", "joint")
        self.ee_action_scale = action_mode.get("ee_action_scale", 0.1)

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
        wrist_video = video_dir / f"{self.task_description.replace(' ', '_')}_wrist_{time_str}.mp4"
        exterior_video = video_dir / f"{self.task_description.replace(' ', '_')}_exterior_{time_str}.mp4"

        self.recorder = Recorder(
            log_path=log_path, 
            video_path=[wrist_video, exterior_video], 
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
        """Connect to Franka robot."""
        try:
            logging.info("\n===== [ROBOT] Connecting to Franka robot =====")
            self.robot_client = FrankaInterfaceClient(ip=self.robot_ip, port=self.robot_port)
            self.robot_client.gripper_initialize()

            joints = self.robot_client.robot_get_joint_positions().tolist()
            if joints and len(joints) == 7:
                logging.info(f"[ROBOT] Current joint positions: {[round(j, 4) for j in joints]}")

            tcp_pose = self.robot_client.robot_get_ee_pose().tolist()
            if tcp_pose and len(tcp_pose) == 6:
                logging.info(f"[ROBOT] Current TCP pose: {[round(p, 4) for p in tcp_pose]}")
                logging.info("===== [ROBOT] Franka initialized successfully =====\n")
        except Exception as e:
            logging.error(f"[ERROR] Failed to connect to Franka robot: {e}")
            exit(1)

    def connect_cameras(self):
        """Initialize and connect RealSense cameras."""
        try:
            logging.info("\n===== [CAM] Initializing cameras =====")
            
            wrist_cfg = RealSenseCameraConfig(
                serial_number_or_name=self.wrist_cam_serial,
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

            camera_config = {"wrist_image": wrist_cfg, "exterior_image": exterior_cfg}
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
    def _transfer_obs_delta_ee_state(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Transfer observation for delta_ee mode."""
        state = np.concatenate((
            np.asarray(obs["ee_pose"], dtype=np.float32),
            np.asarray([obs["gripper_position"]], dtype=np.float32),
        ))
        return {
            "observation/state": state,
            "observation/image": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(obs["exterior_image"], 224, 224)
            ),
            "observation/wrist_image": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(obs["wrist_image"], 224, 224)
            ),
            "prompt": obs["prompt"],
        }

    def _transfer_obs_joint_state(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Transfer observation for joint mode."""
        state = np.concatenate((
            np.asarray(obs["joint_positions"], dtype=np.float32),
            np.asarray([obs["gripper_position"]], dtype=np.float32),
        ))
        return {
            "observation/state": state,
            "observation/image": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(obs["exterior_image"], 224, 224)
            ),
            "observation/wrist_image": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(obs["wrist_image"], 224, 224)
            ),
            "prompt": obs["prompt"],
        }

    def get_obs_state(self) -> Dict[str, Any]:
        """Get current observation from robot and cameras."""
        obs = {}

        if self.robot_client:
            obs["joint_positions"] = self.robot_client.robot_get_joint_positions()
            obs["ee_pose"] = self.robot_client.robot_get_ee_pose()

        if self.cameras:
            for name, cam in self.cameras.items():
                obs[name] = cam.read()

        if self.task_description:
            obs["prompt"] = self.task_description

        if self.robot_client:
            gripper_width = self.robot_client.gripper_get_state()["width"]
            gripper_state = max(0.0, min(1.0, gripper_width / 0.0801))
            obs["gripper_position"] = 0.0 if gripper_state < self.close_threshold else 1.0

        if self.action_mode == "delta_ee":
            return self._transfer_obs_delta_ee_state(obs)
        else:
            return self._transfer_obs_joint_state(obs)

    # --------------------------- ACTION EXECUTION --------------------------- #
    def execute_actions(self, actions: np.ndarray, block: bool = False):
        """Execute actions on the robot."""
        if self.robot_client is None:
            logging.error("[ERROR] Robot not connected")
            return
        
        if self.action_mode == "delta_ee":
            self._execute_delta_ee_actions(actions, block)
        else:
            self._execute_joint_actions(actions, block)

    def _execute_joint_actions(self, actions: np.ndarray, block: bool = False):
        """Execute joint position actions."""
        if block:
            logging.info("[STATE] Moving robot to initial pose...")
            self.robot_client.robot_move_to_joint_positions(positions=actions[:7], time_to_go=1.0)
            self.robot_client.gripper_grasp(width=0.085, speed=self.gripper_speed, force=self.gripper_force)
            logging.info("[STATE] Robot reached initial pose.")
        else:
            for i, action in enumerate(actions[:self.action_horizon]):
                start_time = time.perf_counter()
                
                joint_positions = action[:7]
                self.robot_client.robot_update_desired_joint_positions(joint_positions)
                
                gripper_command = 0 if action[7] < self.close_threshold else 1
                if self.gripper_reverse:
                    gripper_command = 1 - gripper_command
                self.robot_client.gripper_goto(
                    width=gripper_command * 0.085, 
                    speed=self.gripper_speed, 
                    force=self.gripper_force
                )
                
                elapsed = time.perf_counter() - start_time
                to_sleep = 1.0 / self.action_fps - elapsed
                if to_sleep > 0:
                    time.sleep(to_sleep)
                self.fps_action.update()

    def _execute_delta_ee_actions(self, actions: np.ndarray, block: bool = False):
        """Execute delta end-effector actions."""
        if block:
            logging.info("[STATE] Moving robot to initial pose...")
            initial_pose = actions[:6]
            self.robot_client.robot_move_to_ee_pose(
                position=initial_pose[:3],
                orientation=initial_pose[3:6],
                time_to_go=1.0
            )
            self.robot_client.gripper_grasp(width=0.085, speed=self.gripper_speed, force=self.gripper_force)
            logging.info("[STATE] Robot reached initial pose.")
        else:
            current_ee_pose = self.robot_client.robot_get_ee_pose()
            current_pos = current_ee_pose[:3].copy()
            current_rotvec = current_ee_pose[3:6].copy()
            
            for i, action in enumerate(actions[:self.action_horizon]):
                start_time = time.perf_counter()

                delta_pos = action[:3] * self.ee_action_scale
                delta_rotvec = action[3:6] * self.ee_action_scale
                
                target_pos = current_pos + delta_pos
                target_rotvec = apply_delta_rotation(current_rotvec, delta_rotvec)
                target_pose = np.concatenate([target_pos, target_rotvec])
                
                self.robot_client.robot_update_desired_ee_pose(target_pose)
                
                current_ee_pose = self.robot_client.robot_get_ee_pose()
                current_pos = current_ee_pose[:3].copy()
                current_rotvec = current_ee_pose[3:6].copy()

                gripper_command = 0 if action[6] < self.close_threshold else 1
                if self.gripper_reverse:
                    gripper_command = 1 - gripper_command
                self.robot_client.gripper_goto(
                    width=gripper_command * 0.0801, 
                    speed=self.gripper_speed, 
                    force=self.gripper_force
                )
                
                elapsed = time.perf_counter() - start_time
                to_sleep = 1.0 / self.action_fps - elapsed
                if to_sleep > 0:
                    time.sleep(to_sleep)
                self.fps_action.update()

    # --------------------------- MAIN PIPELINE --------------------------- #
    def run(self):
        """Main pipeline: connect everything and run inference loop."""
        logging.info("=" * 60)
        logging.info("Remote Inference with Local Execution")
        logging.info("=" * 60)
        
        # Connect all components
        self.connect_robot()
        self.connect_cameras()
        self.connect_policy_server()
        
        # Move to initial pose
        self.robot_client.robot_move_to_joint_positions(positions=self.initial_pose, time_to_go=5.0)
        self.robot_client.gripper_goto(width=0.085, speed=self.gripper_speed, force=self.gripper_force)
        
        # Start control mode
        if self.action_mode == "delta_ee":
            logging.info("[STATE] Starting Cartesian impedance control...")
            self.robot_client.robot_start_cartesian_impedance_control()
        else:
            logging.info("[STATE] Starting joint impedance control...")
            self.robot_client.robot_start_joint_impedance_control()
        
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
            self.robot_client.robot_move_to_joint_positions(positions=self.initial_pose, time_to_go=5.0)
            self.robot_client.gripper_goto(width=0.085, speed=self.gripper_speed, force=self.gripper_force)
            
        except Exception as e:
            logging.error(f"[ERROR] Inference loop error: {e}")
            self.robot_client.robot_move_to_joint_positions(positions=self.initial_pose, time_to_go=5.0)
            self.robot_client.gripper_goto(width=0.085, speed=self.gripper_speed, force=self.gripper_force)
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
    parser = argparse.ArgumentParser(description="Remote inference with local execution for Franka")
    parser.add_argument("--host", type=str, required=True, help="Remote policy server host")
    parser.add_argument("--port", type=int, default=8000, help="Remote policy server port")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    args = parser.parse_args()
    
    config_path = Path(args.config) if args.config else Path(__file__).parent / "config" / "cfg_franka_pi.yaml"
    
    inference = RemoteInferenceFranka(
        config_path=config_path,
        server_host=args.host,
        server_port=args.port
    )
    inference.run()


if __name__ == "__main__":
    main()
