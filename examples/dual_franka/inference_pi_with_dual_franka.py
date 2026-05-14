import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
import yaml
import os
import sys
import time
import threading
import numpy as np
from pathlib import Path
from typing import Dict, Any
from scipy.spatial.transform import Rotation as R

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (_REPO_ROOT / "packages" / "openpi-client" / "src", _REPO_ROOT / "src"):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from utils import FpsCounter
from openpi_client import image_tools
from recorder import Recorder
from openpi.training import config as _config
from openpi.training import checkpoints as _checkpoints
from openpi.shared import normalize as _normalize
from openpi.policies import policy_config as _policy_config
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
    """Apply delta rotation to current rotation using rotation matrices.
    
    For delta EE actions, the delta rotation is defined in the 
    current end-effector frame (local frame), so we use:
        R_new = R_current @ R_delta
    
    Args:
        current_rotvec: Current rotation vector [rx, ry, rz] in radians
        delta_rotvec: Delta rotation vector [drx, dry, drz] in radians
    
    Returns:
        New rotation vector after applying delta rotation
    """
    # Convert current rotation to matrix
    current_rot = rotvec_to_rotation_matrix(current_rotvec)
    
    # Convert delta rotation to matrix (small rotation in local frame)
    delta_rot = rotvec_to_rotation_matrix(delta_rotvec)
    
    # Apply delta rotation in local frame: R_new = R_current @ R_delta
    new_rot = delta_rot @ current_rot
    
    # Convert back to Euler angles
    new_rotvec = rotation_matrix_to_rotvec(new_rot)
    
    return new_rotvec

def update_latest_symlink(target: Path, link_name: Path):
    """
    这个函数在机器人推理系统中用于维护一个"最新日志"的快捷方式，
    方便用户快速访问当前会话的日志信息。
    """
    if link_name.exists() or link_name.is_symlink():
        link_name.unlink()
    os.symlink(target, link_name)

class Inference:
    def __init__(self, config_path: Path):
        # Load YAML config
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        # Model config
        model = cfg["model"]
        self.model_config = _config.get_config(model["name"])
        self.checkpoint_dir = home / model["checkpoint_dir"]
        
        # Camera config
        cam = cfg["cameras"]
        self.left_wrist_cam_serial = cam["left_wrist_cam_serial"]
        self.right_wrist_cam_serial = cam["right_wrist_cam_serial"]
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
        self.action_mode = action_mode.get("mode", "delta_ee")  # dual franka uses "delta_ee"
        legacy_scale = action_mode.get("ee_action_scale", 1.0)
        self.ee_pos_action_scale = action_mode.get("ee_pos_action_scale", legacy_scale)
        self.ee_rot_action_scale = action_mode.get("ee_rot_action_scale", legacy_scale)
        self.max_pos_delta = action_mode.get("max_pos_delta", None)
        self.max_rot_delta = action_mode.get("max_rot_delta", None)

        # Task config
        task = cfg["task"]
        self.task_description = task["description"]

        # time stamps
        time_str = time.strftime('%Y%m%d-%H%M%S')
        time_path = time.strftime('%Y%m%d')

        # base dir
        base_dir = Path(__file__).parent
        log_dir = base_dir / "logs"
        video_dir = base_dir / "videos" / time_path

        # create dir
        (log_dir / "all_logs").mkdir(parents=True, exist_ok=True)
        video_dir.mkdir(parents=True, exist_ok=True)

        # log paths
        latest_path = log_dir / "latest.yaml"
        log_path = log_dir / "all_logs" / f"log_{time_str}.yaml"

        # video paths - dual franka has 3 cameras
        left_wrist_video = video_dir / f"{self.task_description.replace(' ', '_')}_left_wrist_{time_str}.mp4"
        right_wrist_video = video_dir / f"{self.task_description.replace(' ', '_')}_right_wrist_{time_str}.mp4"
        exterior_video = video_dir / f"{self.task_description.replace(' ', '_')}_exterior_{time_str}.mp4"

        # Recorder
        self.recorder = Recorder(log_path=log_path, video_path=[left_wrist_video, right_wrist_video, exterior_video], display_fps=self.video_fps, visualize=self.visualize)

        # create symlink to latest log
        update_latest_symlink(log_path, latest_path)

        # create FPS counters
        self.fps_action = FpsCounter(name="action")

        # Internal states
        self.robot_client = None
        self.cameras = None


    # --------------------------- ROBOT --------------------------- #
    def connect_robot(self):
        """Connect to Dual Franka robot and print current state."""
        try:
            logging.info("\n===== [ROBOT] Connecting to Dual Franka robot =====")
            self.robot_client = DualFrankaRobotiqRpcClient(ip=self.robot_ip, port=self.robot_port)
            self.robot_client.gripper_initialize()

            # Get observation for both arms
            obs = self.robot_client.get_observation()
            left_arm = obs.get('left_arm', {})
            right_arm = obs.get('right_arm', {})

            # Left arm state
            left_joints = left_arm.get('joint_position', [])
            if left_joints and len(left_joints) == 7:
                formatted = [round(j, 4) for j in left_joints]
                logging.info(f"[ROBOT] Left arm joint positions: {formatted}")

            left_tcp = left_arm.get('ee_pose', [])
            if left_tcp and len(left_tcp) == 6:
                formatted_pose = [round(p, 4) for p in left_tcp]
                logging.info(f"[ROBOT] Left arm TCP pose: {formatted_pose}")

            # Right arm state
            right_joints = right_arm.get('joint_position', [])
            if right_joints and len(right_joints) == 7:
                formatted = [round(j, 4) for j in right_joints]
                logging.info(f"[ROBOT] Right arm joint positions: {formatted}")

            right_tcp = right_arm.get('ee_pose', [])
            if right_tcp and len(right_tcp) == 6:
                formatted_pose = [round(p, 4) for p in right_tcp]
                logging.info(f"[ROBOT] Right arm TCP pose: {formatted_pose}")

            logging.info("===== [ROBOT] Dual Franka initialized successfully =====\n")

        except Exception as e:
            logging.error("===== [ERROR] Failed to connect to Dual Franka robot =====")
            logging.error(f"Exception: {e}\n")
            exit(1)
    # --------------------------- CAMERAS --------------------------- #
    def connect_cameras(self):
        """Initialize and connect RealSense cameras for dual franka."""
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
            logging.error("[ERROR] Failed to initialize cameras.")
            logging.error(f"Exception: {e}\n")
            self.cameras = None

    # --------------------------- OBS TRANSFER --------------------------- #
    def _transfer_obs_dual_franka_state(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Transfer raw observation state to Dual Franka policy format.

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

        dual_franka_obs = {
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

        return dual_franka_obs

    # --------------------------- OBS STATE --------------------------- #
    def get_obs_state(self) -> Dict[str, Any]:
        """Return current observation from dual franka robot."""
        obs = {}

        # Robot state - get both arms
        if self.robot_client:
            full_obs = self.robot_client.get_observation()
            left_arm = full_obs.get('left_arm', {})
            right_arm = full_obs.get('right_arm', {})

            # Joint positions (7D each)
            obs["left_joint_positions"] = np.array(left_arm.get('joint_position', [0.0] * 7))
            obs["right_joint_positions"] = np.array(right_arm.get('joint_position', [0.0] * 7))

            # EE poses (6D each)
            obs["left_ee_pose"] = np.array(left_arm.get('ee_pose', [0.0] * 6))
            obs["right_ee_pose"] = np.array(right_arm.get('ee_pose', [0.0] * 6))

        # Camera images
        if self.cameras:
            for name, cam in self.cameras.items():
                frame = cam.read()
                obs[name] = frame

        # Task description
        if self.task_description:
            obs["prompt"] = self.task_description

        # Gripper state for both arms
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
        """Execute the inferenced actions from the model for dual franka.

        Dual Franka action format (14D):
        - left_delta_ee (6D): dx, dy, dz, drx, dry, drz
        - right_delta_ee (6D): dx, dy, dz, drx, dry, drz
        - left_gripper (1D)
        - right_gripper (1D)
        """
        if self.robot_client is None:
            logging.error("[ERROR] Robot controller not connected. Cannot execute actions.")
            return

        # Check action dimensions
        action_dim = actions.shape[-1] if len(actions.shape) > 1 else len(actions)
        logging.info(f"[DEBUG] Action shape: {actions.shape}, action_dim: {action_dim}")

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

        for i, action in enumerate(actions[:self.action_horizon]):
            start_time = time.perf_counter()

            # Debug: print raw action values (before scaling)
            if i == 0:
                logging.info(f"[DEBUG] Raw action (first in horizon): {action}")
                logging.info(f"[DEBUG] Action pos range: [{action[:6].min():.4f}, {action[:6].max():.4f}]")
                logging.info(f"[DEBUG] Action rot range: [{action[6:12].min():.4f}, {action[6:12].max():.4f}]")

            # Reorder from interleaved [L_dx, R_dx, L_dy, R_dy, ...] to grouped [L(6), R(6)]
            # Interleaved indices (0-based): L=0,2,4,6,8,10  R=1,3,5,7,9,11
            left_delta = np.array([
                action[0], action[2], action[4],   # L_dx, L_dy, L_dz
                action[6], action[8], action[10],  # L_drx, L_dry, L_drz
            ], dtype=np.float32)
            right_delta = np.array([
                action[1], action[3], action[5],   # R_dx, R_dy, R_dz
                action[7], action[9], action[11],  # R_drx, R_dry, R_drz
            ], dtype=np.float32)
            left_delta[:3] *= self.ee_pos_action_scale
            right_delta[:3] *= self.ee_pos_action_scale
            left_delta[3:] *= self.ee_rot_action_scale
            right_delta[3:] *= self.ee_rot_action_scale
            if self.max_pos_delta is not None:
                left_delta[:3] = np.clip(left_delta[:3], -self.max_pos_delta, self.max_pos_delta)
                right_delta[:3] = np.clip(right_delta[:3], -self.max_pos_delta, self.max_pos_delta)
            if self.max_rot_delta is not None:
                left_delta[3:] = np.clip(left_delta[3:], -self.max_rot_delta, self.max_rot_delta)
                right_delta[3:] = np.clip(right_delta[3:], -self.max_rot_delta, self.max_rot_delta)
            left_gripper_cmd = action[12]
            right_gripper_cmd = action[13]

            # Debug: print scaled delta values
            if i == 0:
                logging.info(f"[DEBUG] Left delta (scaled): pos={left_delta[:3]}, rot={left_delta[3:]}")
                logging.info(f"[DEBUG] Right delta (scaled): pos={right_delta[:3]}, rot={right_delta[3:]}")

            # Send delta commands directly (RPC server only supports delta=True)
            self.robot_client.dual_robot_move_to_ee_pose(
                left_delta=left_delta,
                right_delta=right_delta,
                delta=True,  # Send as delta increments
                wait=False,
            )

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

    # --------------------------- PIPELINE --------------------------- #
    def run(self):
        """Main pipeline: connect robot, cameras, and print state."""
        logging.info("========== Starting Dual Franka Inference Pipeline ==========")
        self.connect_robot()
        self.connect_cameras()

        # Move both arms to initial pose
        logging.info("[STATE] Moving arms to initial pose...")
        # Note: dual franka client may need different method for going home
        self.robot_client.go_home('both')
        self.robot_client.open_gripper('left_arm')
        self.robot_client.open_gripper('right_arm')

        obs = self.get_obs_state()
        logging.info(f"[STATE] Observation state keys: {obs.keys()}")
        logging.info(f"[DEBUG] Observation state shape: {obs['observation/state'].shape}")
        logging.info(f"[DEBUG] Observation state values: {obs['observation/state']}")
        
        # Load norm_stats directly from checkpoint assets directory
        # The norm_stats.json is in: checkpoint_dir/assets/norm_stats.json
        norm_stats_path = self.checkpoint_dir / "assets"
        norm_stats = _normalize.load(norm_stats_path)
        logging.info(f"Loaded norm stats from {norm_stats_path}")
        
        # Debug: print norm stats
        logging.info(f"[DEBUG] State norm - mean: {norm_stats['state'].mean}")
        logging.info(f"[DEBUG] State norm - std: {norm_stats['state'].std}")
        logging.info(f"[DEBUG] Actions norm - mean: {norm_stats['actions'].mean}")
        logging.info(f"[DEBUG] Actions norm - std: {norm_stats['actions'].std}")
        
        policy = _policy_config.create_trained_policy(
            self.model_config, 
            self.checkpoint_dir,
            norm_stats=norm_stats
        )
        logging.info("Warming up the model")
        start = time.time()
        policy.infer(obs)
        logging.info(f"Model warmup completed, took {time.time() - start:.2f}s")
        infer_time = 1
        logging.info("========== Starting Inference Loop ==========")
        try:
            while True:
                start_time = time.perf_counter()
                obs = self.get_obs_state()
                result = policy.infer(obs)
                actions = result["actions"]
                
                # Debug: check if actions are normalized or unnormalized
                if infer_time == 1:
                    logging.info(f"[DEBUG] Raw model output (should be unnormalized): {actions[0]}")
                    # Check action statistics
                    logging.info(f"[DEBUG] Actions mean per dim: {actions.mean(axis=0)}")
                    logging.info(f"[DEBUG] Actions std per dim: {actions.std(axis=0)}")
                
                self.execute_actions(result["actions"])
                self.recorder.submit_actions(result["actions"][:self.action_horizon], infer_time, obs["prompt"])
                self.recorder.submit_obs(obs)
                end_time = time.perf_counter()
                logging.info(f"[STATE] Inference loop rate: {1 / (end_time - start_time):.1f} HZ")
                infer_time += 1
        except KeyboardInterrupt:
            logging.info("[INFO] KeyboardInterrupt detected. Saving recorded videos before exiting...")
            self.robot_client.go_home('both')
            self.robot_client.open_gripper('left_arm')
            self.robot_client.open_gripper('right_arm')

        except Exception as e:
            logging.error(f"[ERROR] Inference loop encountered an error: {e}")
            self.robot_client.go_home('both')
            self.robot_client.open_gripper('left_arm')
            self.robot_client.open_gripper('right_arm')

        try:
            ans = input("Save recorded videos before exiting? [Y/n]: ").strip().lower()
            if ans in ("", "y", "yes"):
                logging.info("[INFO] Saving recorded videos before exiting...")
                self.recorder.save_video()

        except Exception as e:
            logging.error(f"[ERROR] Failed to save videos: {e}")

# --------------------------- MAIN --------------------------- #
def main():
    config_path = Path(__file__).parent / "config" / "cfg_franka_pi.yaml"
    inference = Inference(config_path)
    inference.run()

# --------------------------- ENTRY POINT --------------------------- #
if __name__ == "__main__":
    main()
