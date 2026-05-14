import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_dual_franka_example() -> dict:
    """Creates a random input example for the dual franka policy.

    Dual Franka schema:
    - state: 28D [left_joint(7), right_joint(7), left_ee(6), right_ee(6), left_gripper(1), right_gripper(1)]
    - action: 14D interleaved [L_dx, R_dx, L_dy, R_dy, L_dz, R_dz, L_drx, R_drx, L_dry, R_dry, L_drz, R_drz, L_grip, R_grip]
    - cameras: head_image, left_wrist_image, right_wrist_image
    """
    return {
        "observation/state": np.random.rand(28),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/right_wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/left_wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class DualFrankaInputs(transforms.DataTransformFn):
    """Convert dual franka observations into the common model input format.

    This is the robot-facing interface for dual arm franka robots.

    Data format:
    - state: 28D, already reordered by OBS_INDICES in data_loader.py:
        [left_joint(7), right_joint(7), left_ee(6), right_ee(6), left_gripper_state(1), right_gripper_state(1)]
      The 28D state will be padded to action_dim by PadStatesAndActions.
    - image: head_image, left_wrist_image, right_wrist_image
    - action: 14D delta EE pose + gripper command

    NOTE: The state reordering from 30D interleaved to 28D grouped is handled by
    OBS_INDICES in data_loader.py during training, and by the robot client during
    inference. We do NOT reorder here to avoid double-reordering.
    """

    # Determines which model will be used.
    model_type: _model.ModelType

    # Source keys in the raw observation dict.
    state_key: str = "observation/state"
    base_image_key: str = "observation/image"
    left_wrist_image_key: str = "observation/wrist_image"
    right_wrist_image_key: str = "observation/right_wrist_image"
    prompt_key: str = "prompt"

    def __call__(self, data: dict) -> dict:
        # Parse images
        base_image = _parse_image(data[self.base_image_key])
        left_wrist_image = _parse_image(data[self.left_wrist_image_key])
        right_wrist_image = _parse_image(data[self.right_wrist_image_key])

        # Use state directly - it is already in the correct 28D order:
        # [left_joint(7), right_joint(7), left_ee(6), right_ee(6), left_gripper(1), right_gripper(1)]
        # During training: reordered by OBS_INDICES in data_loader.py
        # During inference: built in correct order by the robot client
        state = np.asarray(data[self.state_key])

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        # Actions are only present during training.
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        # Pass the prompt to the model.
        if self.prompt_key in data:
            prompt = data[self.prompt_key]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class DualFrankaOutputs(transforms.DataTransformFn):
    """Convert model outputs back to the dual franka action format.

    The model outputs actions in the same interleaved order as the dataset:
    [L_dx, R_dx, L_dy, R_dy, L_dz, R_dz, L_drx, R_drx, L_dry, R_dry, L_drz, R_drz, L_grip, R_grip]

    The inference scripts must reorder this to grouped order before execution:
    [L_dx, L_dy, L_dz, L_drx, L_dry, L_drz, R_dx, R_dy, R_dz, R_drx, R_dry, R_drz, L_grip, R_grip]
    """

    # Number of action dimensions to return to the robot.
    action_dim: int = 14

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
