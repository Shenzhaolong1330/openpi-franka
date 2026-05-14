import dataclasses

import jax

from openpi.models import model as _model
from openpi.policies import dual_franka_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

config = _config.get_config("pi05_droid_finetune_dual_franka")
checkpoint_dir = "/path/to/checkpoint"

# Create a trained policy.
policy = _policy_config.create_trained_policy(config, checkpoint_dir)

# Run inference on a dummy example. This example corresponds to observations produced by the Dual Franka runtime.
example = dual_franka_policy.make_dual_franka_example()
result = policy.infer(example)

# Delete the policy to free up memory.
del policy

print("Actions shape:", result["actions"].shape)