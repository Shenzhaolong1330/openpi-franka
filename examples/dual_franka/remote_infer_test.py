from openpi_client import websocket_client_policy
from openpi.policies import dual_franka_policy
import numpy as np

def make_dual_franka_example() -> dict:
    """Creates a random input example for the dual franka policy."""
    return {"observation/state": np.random.rand(28),    
            "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
            "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
            "observation/right_wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
            "prompt": "pick up all the objects into the basket and seal the basket"
            }

client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)
obs = dual_franka_policy.make_dual_franka_example()
response = client.infer(obs)
print("Response from server:")
print(response)