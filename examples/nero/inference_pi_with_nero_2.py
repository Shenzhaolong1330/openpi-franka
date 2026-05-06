import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
import yaml
import os
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any
from scipy.spatial.transform import Rotation as R
from utils import FpsCounter
from openpi_client import image_tools
from recorder import Recorder 
from openpi.training import config as _config
from openpi.policies import policy_config as _policy_config
from lerobot.cameras.configs import ColorMode, Cv2Rotation
from lerobot.cameras.realsense.camera_realsense import RealSenseCameraConfig
from lerobot.cameras import make_cameras_from_configs

# Import the Nero client
from nero_interface_client import NeroDualArmClient

home = Path.home()

def rotvec_to_rotation_matrix(rotation_vector: np.ndarray) -> np.ndarray:
    return R.from_rotvec(rotation_vector).as_matrix()

def rotation_matrix_to_rotvec(rot_matrix: np.ndarray) -> np.ndarray:
    return R.from_matrix(rot_matrix).as_rotvec()

def apply_delta_rotation(current_rotvec: np.ndarray, delta_rotvec: np.ndarray) -> np.ndarray:
    """Apply delta rotation to current rotation using rotation matrices."""
    # Convert current rotation to matrix
    current_rot = rotvec_to_rotation_matrix(current_rotvec)
    # Convert delta rotation to matrix
    delta_rot = rotvec_to_rotation_matrix(delta_rotvec)
    # Apply delta rotation
    new_rot = delta_rot @ current_rot
    # Convert back to Euler angles
    return rotation_matrix_to_rotvec(new_rot)

def update_latest_symlink(target: Path, link_name: Path):
    """
    Update a symlink to point to the latest log file.
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
        chk_dir = str(model["checkpoint_dir"])
        self.checkpoint_dir = Path(chk_dir) if chk_dir.startswith("/") else home / chk_dir
        
        # Camera config (base and two wrists)
        cam = cfg.get("cameras", {})
        self.exterior_cam_serial = cam.get("exterior_cam_serial")
        self.left_wrist_cam_serial = cam.get("left_wrist_cam_serial")
        self.right_wrist_cam_serial = cam.get("right_wrist_cam_serial")
        self.cam_fps = cam.get("fps", 30)

        # Video config
        video = cfg.get("video", {"fps": 15, "visualize": True})
        self.video_fps = video.get("fps", 15)
        self.visualize = video.get("visualize", True)

        # Robot config
        robot = cfg.get("robot", {})
        self.robot_ip = robot.get("ip", "127.0.0.1")
        self.robot_port = robot.get("port", 4242)
        self.initial_left_joints = np.asarray(robot.get("initial_left_joints", np.zeros(7)), dtype=np.float32)
        self.initial_right_joints = np.asarray(robot.get("initial_right_joints", np.zeros(7)), dtype=np.float32)
        
        self.action_fps = cfg.get("run", {}).get("action_fps", robot.get("action_fps", 20))
        self.action_horizon = cfg.get("run", {}).get("action_horizon", robot.get("action_horizon", 10))

        # Gripper config
        gripper = cfg.get("gripper", {})
        self.close_threshold = gripper.get("close_threshold", 0.05)
        self.gripper_force = gripper.get("gripper_force", 10.0)
        self.gripper_speed = gripper.get("gripper_speed", 0.1)
        self.gripper_reverse = gripper.get("gripper_reverse", False)

        # Action mode config
        action_mode = cfg.get("action_mode", {})
        self.action_mode = action_mode.get("mode", "delta_ee")
        self.ee_action_scale = action_mode.get("ee_action_scale", 1.0)

        # Task config
        task = cfg.get("task", {})
        self.task_description = task.get("description", cfg.get("run", {}).get("task_description", "pick and place"))
        
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

        # video paths
        exterior_video = video_dir / f"{self.task_description.replace(' ', '_')}_exterior_{time_str}.mp4"
        left_wrist_video = video_dir / f"{self.task_description.replace(' ', '_')}_left_wrist_{time_str}.mp4"
        right_wrist_video = video_dir / f"{self.task_description.replace(' ', '_')}_right_wrist_{time_str}.mp4"

        # Recorder  
        self.recorder = Recorder(log_path=log_path, video_path=[exterior_video, left_wrist_video, right_wrist_video], display_fps=self.video_fps, visualize=self.visualize)
        
        # create symlink to latest log
        update_latest_symlink(log_path, latest_path)

        # create FPS counters
        self.fps_action = FpsCounter(name="action")

        # Internal states
        self.robot_client = None
        self.cameras = None

    # --------------------------- ROBOT --------------------------- #
    def connect_robot(self):
        """Connect to Nero dual-arm robot and print current state."""
        try:
            logging.info("\n===== [ROBOT] Connecting to Nero dual-arm robot =====")
            self.robot_client = NeroDualArmClient(ip=self.robot_ip, port=self.robot_port)

            # Left Arm State
            left_pose = self.robot_client.left_robot_get_ee_pose().tolist()
            l_grip = self.robot_client.left_gripper_get_state().get("width", 0.0)
            formatted_left = [round(x, 4) for x in left_pose]
            logging.info(f"[ROBOT] Current Left TCP pose: {formatted_left} | Gripper: {l_grip:.4f}")

            # Right Arm State
            right_pose = self.robot_client.right_robot_get_ee_pose().tolist()
            r_grip = self.robot_client.right_gripper_get_state().get("width", 0.0)
            formatted_right = [round(x, 4) for x in right_pose]
            logging.info(f"[ROBOT] Current Right TCP pose: {formatted_right} | Gripper: {r_grip:.4f}")

            logging.info("===== [ROBOT] Nero initialized successfully =====\n")

        except Exception as e:
            logging.error("===== [ERROR] Failed to connect to Nero robot =====")
            logging.error(f"Exception: {e}\n")
            exit(1)

    # --------------------------- CAMERAS --------------------------- #
    def connect_cameras(self):
        """Initialize and connect RealSense cameras."""
        try:
            logging.info("\n===== [CAM] Initializing cameras =====")
            
            camera_config = {}
            if self.exterior_cam_serial:
                camera_config["exterior_image"] = RealSenseCameraConfig(
                    serial_number_or_name=self.exterior_cam_serial,
                    fps=self.cam_fps, width=640, height=480, color_mode=ColorMode.RGB,
                    use_depth=False, rotation=Cv2Rotation.NO_ROTATION,
                )
            if self.left_wrist_cam_serial:
                camera_config["left_wrist_image"] = RealSenseCameraConfig(
                    serial_number_or_name=self.left_wrist_cam_serial,
                    fps=self.cam_fps, width=640, height=480, color_mode=ColorMode.RGB,
                    use_depth=False, rotation=Cv2Rotation.NO_ROTATION,
                )
            if self.right_wrist_cam_serial:
                camera_config["right_wrist_image"] = RealSenseCameraConfig(
                    serial_number_or_name=self.right_wrist_cam_serial,
                    fps=self.cam_fps, width=640, height=480, color_mode=ColorMode.RGB,
                    use_depth=False, rotation=Cv2Rotation.NO_ROTATION,
                )

            if camera_config:
                self.cameras = make_cameras_from_configs(camera_config)
                for name, cam in self.cameras.items():
                    cam.connect()
                    logging.info(f"[CAM] {name} connected successfully.")
                logging.info("===== [CAM] Cameras initialized successfully =====\n")
            else:
                logging.error("[ERROR] No cameras configured.")
                self.cameras = None

        except Exception as e:
            logging.error("[ERROR] Failed to initialize cameras.")
            logging.error(f"Exception: {e}\n")
            self.cameras = None

    # --------------------------- OBS TRANSFER --------------------------- #
    def _transfer_obs_state(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Transfer raw observation state to policy format."""

        # 14D State for Dual Arm
        state = np.concatenate((
            np.asarray(obs["left_ee_pose"], dtype=np.float32),
            np.asarray(obs["right_ee_pose"], dtype=np.float32),
            np.asarray([obs["left_gripper_position"]], dtype=np.float32),
            np.asarray([obs["right_gripper_position"]], dtype=np.float32),
        ))

        ext_img = obs.get("exterior_image", np.zeros((480, 640, 3), dtype=np.uint8))
        lw_img = obs.get("left_wrist_image", np.zeros((480, 640, 3), dtype=np.uint8))
        rw_img = obs.get("right_wrist_image", np.zeros((480, 640, 3), dtype=np.uint8))

        nero_obs = {
            "observation/state": state,
            "observation/image": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(ext_img, 224, 224)
            ),
            "observation/wrist_image": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(lw_img, 224, 224)
            ),
            "observation/right_wrist_image": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(rw_img, 224, 224)
            ),
            "prompt": obs.get("prompt", ""),
        }

        # Handle keys as recommended in Nero user memory constraints
        nero_obs["base_0_rgb"] = nero_obs["observation/image"]
        nero_obs["left_wrist_0_rgb"] = nero_obs["observation/wrist_image"]
        nero_obs["right_wrist_0_rgb"] = nero_obs["observation/right_wrist_image"]

        return nero_obs

    # --------------------------- OBS STATE --------------------------- #
    def get_obs_state(self) -> Dict[str, Any]:
        """Return current observation from robot."""
        obs = {}

        # Robot state
        if self.robot_client:
            obs["left_ee_pose"] = self.robot_client.left_robot_get_ee_pose()
            obs["right_ee_pose"] = self.robot_client.right_robot_get_ee_pose()
            
            l_grip_width = self.robot_client.left_gripper_get_state().get("width", 0.0)
            l_grip_state = max(0.0, min(1.0, l_grip_width/0.0801))
            obs["left_gripper_position"] = 0.0 if l_grip_state < self.close_threshold else 1.0
            
            r_grip_width = self.robot_client.right_gripper_get_state().get("width", 0.0)
            r_grip_state = max(0.0, min(1.0, r_grip_width/0.0801))
            obs["right_gripper_position"] = 0.0 if r_grip_state < self.close_threshold else 1.0
        else:
            obs["left_ee_pose"] = np.zeros(6, dtype=np.float32)
            obs["right_ee_pose"] = np.zeros(6, dtype=np.float32)
            obs["left_gripper_position"] = 0.0
            obs["right_gripper_position"] = 0.0

        # Camera images
        if self.cameras:
            for name, cam in self.cameras.items():
                obs[name] = cam.read()
        else:
            dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
            obs["exterior_image"] = dummy_img
            obs["left_wrist_image"] = dummy_img
            obs["right_wrist_image"] = dummy_img

        # Task description    
        if self.task_description:
            obs["prompt"] = self.task_description
        
        return self._transfer_obs_state(obs)

    # --------------------------- ACTION EXECUTION --------------------------- #
    def execute_actions(self, actions: np.ndarray, block: bool = False):
        """Execute the inferenced actions from the model."""
        if self.robot_client is None:
            logging.error("[ERROR] Robot controller not connected. Cannot execute actions.")
            return
        
        if self.action_mode == "delta_ee":
            self._execute_delta_ee_actions(actions, block)
        else:
            logging.error(f"[ERROR] Unsupported action mode {self.action_mode} for NERO.")

    def _execute_delta_ee_actions(self, actions: np.ndarray, block: bool = False):
        """Execute delta end-effector actions on dual arm.
        Action format (14D): [l_dx, l_dy, l_dz, l_drx, l_dry, l_drz,
                              r_dx, r_dy, r_dz, r_drx, r_dry, r_drz,
                              l_grip, r_grip]"""
        cur_left_ee = self.robot_client.left_robot_get_ee_pose()
        cur_right_ee = self.robot_client.right_robot_get_ee_pose()
        
        cur_l_pos, cur_l_rot = cur_left_ee[:3].copy(), cur_left_ee[3:6].copy()
        cur_r_pos, cur_r_rot = cur_right_ee[:3].copy(), cur_right_ee[3:6].copy()

        for i, action in enumerate(actions[:self.action_horizon]):
            start_time = time.perf_counter()

            # Left Arm Deltas
            d_l_pos, d_l_rot = action[0:3] * self.ee_action_scale, action[3:6] * self.ee_action_scale
            tgt_l_pose = np.concatenate([cur_l_pos + d_l_pos, apply_delta_rotation(cur_l_rot, d_l_rot)])

            # Right Arm Deltas
            d_r_pos, d_r_rot = action[6:9] * self.ee_action_scale, action[9:12] * self.ee_action_scale
            tgt_r_pose = np.concatenate([cur_r_pos + d_r_pos, apply_delta_rotation(cur_r_rot, d_r_rot)])

            # Move arms
            self.robot_client.servo_p("left_robot", tgt_l_pose, delta=False)
            self.robot_client.servo_p("right_robot", tgt_r_pose, delta=False)

            cur_left_ee = self.robot_client.left_robot_get_ee_pose()
            cur_right_ee = self.robot_client.right_robot_get_ee_pose()
            cur_l_pos, cur_l_rot = cur_left_ee[:3].copy(), cur_left_ee[3:6].copy()
            cur_r_pos, cur_r_rot = cur_right_ee[:3].copy(), cur_right_ee[3:6].copy()

            # Control grippers
            l_grip_cmd = 0 if action[12] < self.close_threshold else 1
            r_grip_cmd = 0 if action[13] < self.close_threshold else 1
            if self.gripper_reverse:
                l_grip_cmd, r_grip_cmd = 1 - l_grip_cmd, 1 - r_grip_cmd

            self.robot_client.left_gripper_goto(width=l_grip_cmd*0.0801, force=self.gripper_force)
            self.robot_client.right_gripper_goto(width=r_grip_cmd*0.0801, force=self.gripper_force)

            elapsed = time.perf_counter() - start_time
            to_sleep = 1.0 / self.action_fps - elapsed
            if to_sleep > 0:
                time.sleep(to_sleep)
            self.fps_action.update()

    # --------------------------- PIPELINE --------------------------- #
    def run(self):
        """Main pipeline: connect robot, cameras, load policy, execute loop."""
        logging.info("========== Starting Inference Pipeline ==========")
        self.connect_robot()
        self.connect_cameras()
        
        # Initial robot position
        if self.robot_client:
            if np.any(self.initial_left_joints):
                logging.info("[STATE] Moving left robot to initial pose...")
                self.robot_client.left_robot_move_to_joint_positions(self.initial_left_joints)
            if np.any(self.initial_right_joints):
                logging.info("[STATE] Moving right robot to initial pose...")
                self.robot_client.right_robot_move_to_joint_positions(self.initial_right_joints)

            self.robot_client.left_gripper_goto(width=0.0801, force=self.gripper_force)
            self.robot_client.right_gripper_goto(width=0.0801, force=self.gripper_force)
        
        obs = self.get_obs_state()
        logging.info(f"[STATE] Observation state mapped keys: {obs.keys()}")
        
        # Policy
        policy = _policy_config.create_trained_policy(self.model_config, self.checkpoint_dir)
        logging.info("Warming up the model...")
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
                self.execute_actions(result["actions"])
                self.recorder.submit_actions(result["actions"][:self.action_horizon], infer_time, obs.get("prompt", ""))
                self.recorder.submit_obs(obs)
                end_time = time.perf_counter()
                logging.info(f"[STATE] Inference loop rate: {1 / (end_time - start_time):.1f} HZ")
                infer_time += 1
        except KeyboardInterrupt:
            logging.info("\n[INFO] KeyboardInterrupt detected. Stopping.")
        except Exception as e:
            logging.error(f"[ERROR] Inference loop encountered an error: {e}")

        try:
            ans = input("Save recorded videos before exiting? [Y/n]: ").strip().lower()
            if ans in ("", "y", "yes"):
                logging.info("[INFO] Saving recorded videos before exiting...")
                self.recorder.save_video()
                
        except Exception as e:
            logging.error(f"[ERROR] Failed to save videos: {e}")

        finally:
            if self.robot_client:
                try:
                    self.robot_client.close()
                except AttributeError:
                    pass
            if self.cameras:
                for name, cam in self.cameras.items():
                    cam.disconnect()

# --------------------------- MAIN --------------------------- #
def main():
    config_path = Path(__file__).parent / "config" / "cfg_nero_pi.yaml"
    inference = Inference(config_path)
    inference.run()

# --------------------------- ENTRY POINT --------------------------- #
if __name__ == "__main__":
    main()
