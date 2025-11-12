# f1_client.py
import os
import base64
import numpy as np
import flwr as fl

from fl_client import FLClient  # your existing client implementation
from update_utils import state_dict_to_numpy, serialize_numpy_state

# Absolute project path
PROJECT_ROOT = "/content/drive/MyDrive/fed-hospital-project"
DATA_PATH = os.path.join(PROJECT_ROOT, "data")
KEYS_PATH = os.path.join(PROJECT_ROOT, "keys")

# Client-specific constants
CLIENT_ID = 1
PRIVKEY = os.path.join(KEYS_PATH, f"client{CLIENT_ID}_priv.pem")

# Load local data (client 1)
X_tr = np.load(os.path.join(DATA_PATH, f"client{CLIENT_ID}_X_train.npy"))
y_tr = np.load(os.path.join(DATA_PATH, f"client{CLIENT_ID}_y_train.npy"))
X_val = np.load(os.path.join(DATA_PATH, f"client{CLIENT_ID}_X_val.npy"))
y_val = np.load(os.path.join(DATA_PATH, f"client{CLIENT_ID}_y_val.npy"))

# Instantiate original client (expects FLClient signature)
base_client = FLClient(
    cid=CLIENT_ID,
    X_tr=X_tr, y_tr=y_tr,
    X_val=X_val, y_val=y_val,
    privkey_path=PRIVKEY
)

# Create a subclass that attaches the named serialized state blob after local training
class FLClientWithBlob(base_client.__class__):
    def fit(self, parameters, config):
        """
        Call parent's fit() and attach a serialized named-state blob into the returned metrics
        so that server_aggregator can perform key-wise aggregation.
        Expected parent return: (parameters, num_examples, metrics_dict)
        """
        # Call base implementation
        res = super().fit(parameters, config)
        # Normalize return into (parameters, num_examples, metrics)
        try:
            parameters_out, num_examples, metrics = res
        except Exception:
            # If return format differs, try to handle gracefully
            # fallback: treat entire res as parameters_out
            parameters_out = res
            num_examples = getattr(res, "num_examples", 0)
            metrics = {}

        # Try to attach named serialized blob
        try:
            # Convert state_dict -> numpy dict
            numpy_state = state_dict_to_numpy(self.model.state_dict())
            # Serialize with optional meta
            blob = serialize_numpy_state(numpy_state, meta={"cid": CLIENT_ID})
            blob_b64 = base64.b64encode(blob).decode()
            metrics = metrics or {}
            metrics["serialized_blob_b64"] = blob_b64
        except Exception as e:
            # Log but do not crash the training flow
            print(f"[f{CLIENT_ID}] Warning: couldn't attach serialized blob: {e}")

        return parameters_out, num_examples, metrics

# Instantiate the patched client class, copying necessary attributes from base_client
client = FLClientWithBlob(
    cid=base_client.cid,
    X_tr=base_client.X_tr, y_tr=base_client.y_tr,
    X_val=base_client.X_val, y_val=base_client.y_val,
    privkey_path=PRIVKEY
)

# If your FLClient base class builds model etc. in __init__, above instantiation should work.
# Start Flower client (blocking)
fl.client.start_numpy_client(server_address="localhost:9091", client=client)
