# src/update_utils.py
import io, pickle, numpy as np
from typing import Dict, List, Tuple

def state_dict_to_numpy(state_dict) -> Dict[str, np.ndarray]:
    """Convert PyTorch state_dict (tensors) to numpy arrays (for serialization)."""
    out = {}
    for k, v in state_dict.items():
        # if torch tensor -> convert; else assume ndarray already
        try:
            import torch
            if isinstance(v, torch.Tensor):
                out[k] = v.cpu().detach().numpy().copy()
            else:
                out[k] = np.asarray(v).copy()
        except Exception:
            out[k] = np.asarray(v).copy()
    return out

def numpy_to_state_dict(numpy_dict: Dict[str, np.ndarray]):
    """Return dict that can be passed to model.load_state_dict after converting to tensors if needed."""
    try:
        import torch
        return {k: torch.tensor(v) for k, v in numpy_dict.items()}
    except Exception:
        return numpy_dict

def serialize_numpy_state(numpy_state: Dict[str, np.ndarray], meta: dict=None) -> bytes:
    """Serialize named numpy state + optional meta to bytes (pickle)."""
    payload = {"meta": meta or {}, "state": {k: v for k, v in numpy_state.items()}}
    return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)

def deserialize_numpy_state(blob: bytes) -> Tuple[Dict[str, np.ndarray], dict]:
    """Deserialize bytes to (numpy_state_dict, meta)."""
    payload = pickle.loads(blob)
    return payload.get("state", {}), payload.get("meta", {})

def fedavg_aggregate_named(state_list: List[Dict[str, np.ndarray]], weights: List[float]=None) -> Dict[str, np.ndarray]:
    """Aggregate a list of named numpy state_dicts (all must have same keys and shapes).
    weights: list of client weights (e.g., n_examples). If None, equal weights.
    """
    n = len(state_list)
    if n == 0:
        raise ValueError("No states to aggregate")
    weights = weights or [1.0]*n
    # Validate keys and shapes
    keys = list(state_list[0].keys())
    for s in state_list:
        if set(s.keys()) != set(keys):
            raise ValueError("All state dicts must have same keys")
    # Use float64 accumulator for numeric stability
    agg = {}
    total_w = float(sum(weights))
    for k in keys:
        shape = state_list[0][k].shape
        acc = np.zeros(shape, dtype=np.float64)
        for s, w in zip(state_list, weights):
            arr = np.asarray(s[k], dtype=np.float64)
            if arr.shape != shape:
                raise ValueError(f"Shape mismatch for key {k}: {arr.shape} vs {shape}")
            acc += arr * (w/total_w)
        agg[k] = acc.astype(state_list[0][k].dtype)  # keep original dtype
    return agg
