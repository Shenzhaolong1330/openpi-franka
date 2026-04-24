from openpi_client import websocket_client_policy
from openpi.policies import franka_policy
import numpy as np

def make_ur5e_example() -> dict:
    """Creates a random input example for the ur5e policy."""
    return {"observation/state": np.random.rand(7),    
            "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
            "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
            "prompt": "do something"
            }

client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)
obs = franka_policy.make_franka_delta_ee_example()
response = client.infer(obs)
print("Response from server:")
print(response)