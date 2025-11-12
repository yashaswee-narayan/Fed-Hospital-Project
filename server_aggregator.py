import sys
sys.path.append('./drive/MyDrive/fed-hospital-project/src/')
import os, pickle, numpy as np, requests, base64, traceback
import flwr as fl

# Import the robust helpers we discussed
from update_utils import (
    deserialize_numpy_state,
    serialize_numpy_state,
    fedavg_aggregate_named,
)

PROJECT_ROOT = "/content/drive/MyDrive/fed-hospital-project"


class AccountableFedAvg(fl.server.strategy.FedAvg):
    """FedAvg strategy extended with accountability and GM unmasking.
       Supports named-state dict aggregation (recommended) and falls back to
       default ndarray aggregation when named blobs are not provided.
    """

    def __init__(self, gm_url="http://localhost:5001/open", save_root=PROJECT_ROOT, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gm_url = gm_url
        self.save_root = save_root
        os.makedirs(os.path.join(save_root, "outputs", "evidence"), exist_ok=True)
        os.makedirs(os.path.join(save_root, "outputs", "global_weights"), exist_ok=True)
        self.last_weights = None  # either list (legacy) or dict (named)

    def _compute_named_delta_norm(self, state_dict, last_state_dict):
        """Compute Frobenius norm between two named state dicts (sum over all keys)."""
        total = 0.0
        for k, v in state_dict.items():
            if k in last_state_dict:
                a = np.asarray(v, dtype=np.float64)
                b = np.asarray(last_state_dict[k], dtype=np.float64)
                if a.shape != b.shape:
                    # if shapes different, use difference of sizes (penalize heavily)
                    total += max(a.size, b.size)
                else:
                    total += float(np.sum((a - b) ** 2))
            else:
                total += float(np.sum(np.asarray(state_dict[k], dtype=np.float64) ** 2))
        return float(np.sqrt(total))

    def aggregate_fit(self, rnd, results, failures):
        """
        Attempt to aggregate named serialized blobs if present (recommended).
        Fallback to the superclass FedAvg aggregation if not.
        """
        parsed = []
        # parse results
        for client_proxy, fit_res in results:
            cid = client_proxy.cid
            # try to get named blob from metrics (preferred)
            metrics = fit_res.metrics or {}
            blob_b64 = metrics.get("serialized_blob_b64") or metrics.get("params_blob_b64")
            # also keep ndarrays (legacy path)
            try:
                params_nd = fl.common.parameters_to_ndarrays(fit_res.parameters)
            except Exception:
                params_nd = None

            parsed.append({
                "cid": cid,
                "num_examples": fit_res.num_examples,
                "metrics": metrics,
                "blob_b64": blob_b64,
                "params_nd": params_nd,
                "client_proxy": client_proxy,
            })

        # Decide whether to use named-blob aggregation (if at least one client provided a blob)
        use_named = any(p["blob_b64"] for p in parsed)

        suspicious = []
        norms = []

        if use_named:
            # Collect entries that actually have blobs
            blob_entries = [p for p in parsed if p["blob_b64"]]
            if not blob_entries:
                print("[AccountableFedAvg] No valid serialized blobs found despite use_named=True; falling back.")
                use_named = False

        # If we have last_weights, it might be a dict (named) or list (legacy)
        # Compute delta norms for suspicious detection accordingly
        if self.last_weights is not None:
            if use_named and isinstance(self.last_weights, dict):
                # compute named delta norms
                for item in parsed:
                    if item["blob_b64"]:
                        try:
                            blob = base64.b64decode(item["blob_b64"])
                            state_dict, meta = deserialize_numpy_state(blob)
                            item["named_state"] = state_dict
                            dn = self._compute_named_delta_norm(state_dict, self.last_weights)
                            item["delta_norm"] = dn
                            norms.append(dn)
                        except Exception as e:
                            print(f"[AccountableFedAvg] Error computing named delta for cid {item['cid']}: {e}")
                            item["delta_norm"] = None
                    else:
                        item["delta_norm"] = None
                # compute threshold
                valid_norms = [v for v in norms if v is not None]
                if valid_norms:
                    mean, std = float(np.mean(valid_norms)), float(np.std(valid_norms))
                    threshold = mean + 3 * std
                    for item in parsed:
                        if item.get("delta_norm") is not None and item["delta_norm"] > threshold:
                            suspicious.append(item)
            elif not use_named and isinstance(self.last_weights, (list, np.ndarray)):
                # legacy path: compare ndarrays lists
                for item in parsed:
                    if item["params_nd"] is not None:
                        try:
                            dn = np.sqrt(sum([np.sum((a - b) ** 2) for a, b in zip(item["params_nd"], self.last_weights)]))
                            item["delta_norm"] = float(dn)
                            norms.append(dn)
                        except Exception as e:
                            print(f"[AccountableFedAvg] Error computing legacy delta for cid {item['cid']}: {e}")
                            item["delta_norm"] = None
                    else:
                        item["delta_norm"] = None
                valid_norms = [v for v in norms if v is not None]
                if valid_norms:
                    mean, std = float(np.mean(valid_norms)), float(np.std(valid_norms))
                    threshold = mean + 3 * std
                    for item in parsed:
                        if item.get("delta_norm") is not None and item["delta_norm"] > threshold:
                            suspicious.append(item)
            else:
                # last_weights shape/type doesn't match incoming format; skip delta checks
                for item in parsed:
                    item["delta_norm"] = None
        else:
            for item in parsed:
                item["delta_norm"] = None

        # Save evidence for suspicious items (as before)
        for item in suspicious:
            ev_path = os.path.join(self.save_root, "outputs", "evidence", f"round{rnd}_cid{item['cid']}.pkl")
            try:
                pickle.dump(item, open(ev_path, "wb"))
            except Exception as e:
                print("[AccountableFedAvg] Error saving evidence:", e)

            # Notify GM for unmask
            try:
                payload = {
                    "evidence": {
                        "serialized_blob_b64": item.get("blob_b64"),
                        "signature_b64": item.get("metrics", {}).get("signature_b64"),
                        "round": rnd
                    },
                    "requestor": "server",
                    "reason": "delta_norm_outlier"
                }
                r = requests.post(self.gm_url, json=payload, timeout=10)
                if r.ok:
                    resp = r.json()
                    print(f"[GM] Unmasked client id: {resp.get('client_id')}, verified={resp.get('verified')}")
                else:
                    print(f"[GM] Could not verify suspicious client {item['cid']}")
            except Exception as e:
                print("GM connection error:", e)

        # Now perform the actual aggregation
        try:
            if use_named:
                # Aggregate only those entries that provided named blobs
                blob_entries = [p for p in parsed if p["blob_b64"]]
                states = []
                weights = []
                meta_example = None
                for item in blob_entries:
                    try:
                        blob = base64.b64decode(item["blob_b64"])
                        state_dict, meta = deserialize_numpy_state(blob)
                        states.append(state_dict)
                        weights.append(item["num_examples"] or 1)
                        meta_example = meta_example or meta
                    except Exception as e:
                        print(f"[AccountableFedAvg] Failed to deserialize blob for cid {item['cid']}: {e}")
                        traceback.print_exc()

                if len(states) == 0:
                    # nothing to aggregate — fallback to superclass behavior
                    print("[AccountableFedAvg] No valid named states to aggregate; falling back to default aggregation.")
                    aggregated = super().aggregate_fit(rnd, results, failures)
                else:
                    # Ensure keys/shapes consistent inside fedavg_aggregate_named (it will raise if not)
                    agg_state = fedavg_aggregate_named(states, weights)
                    # Save aggregated named state for inspection
                    savep = os.path.join(self.save_root, "outputs", "global_weights", f"round_{rnd}.pkl")
                    try:
                        pickle.dump(agg_state, open(savep, "wb"))
                    except Exception as e:
                        print("[AccountableFedAvg] Save error:", e)

                    # convert agg_state (dict) into ndarrays list in deterministic order
                    keys_order = list(agg_state.keys())
                    nds = [agg_state[k] for k in keys_order]
                    parameters = fl.common.ndarrays_to_parameters(nds)

                    # store last_weights as named dict for next round delta checks
                    self.last_weights = agg_state

                    aggregated = (parameters, {"model_keys": keys_order, "meta": meta_example or {}})
            else:
                # Legacy behavior: use superclass FedAvg aggregation (list-of-ndarrays)
                aggregated = super().aggregate_fit(rnd, results, failures)
                # Try to save ndarrays to last_weights as list (for legacy delta detection)
                try:
                    if aggregated is not None and aggregated[0] is not None:
                        nds = fl.common.parameters_to_ndarrays(aggregated[0])
                        self.last_weights = nds
                        savep = os.path.join(self.save_root, "outputs", "global_weights", f"round_{rnd}.pkl")
                        pickle.dump(nds, open(savep, "wb"))
                except Exception as e:
                    print("[AccountableFedAvg] Save error (legacy):", e)

        except Exception as e:
            # If something unexpected happens, log and fallback to superclass
            print("[AccountableFedAvg] Aggregation error:", e)
            traceback.print_exc()
            try:
                aggregated = super().aggregate_fit(rnd, results, failures)
            except Exception as e2:
                print("[AccountableFedAvg] Super fallback failed:", e2)
                aggregated = None

        return aggregated
