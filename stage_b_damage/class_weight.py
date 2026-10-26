"""Loss and identity helpers for the paired class-weight experiment."""
import hashlib
import json

import torch

from stage_b_damage.dataset import LABEL_TO_ID


def class_weights(counts, mode):
    if set(counts) != set(LABEL_TO_ID) or any(counts[k] <= 0 for k in LABEL_TO_ID):
        raise ValueError("All four training classes must have positive counts")
    if mode == "equal":
        return torch.ones(4, dtype=torch.float32)
    if mode != "inverse_frequency":
        raise ValueError(f"Unknown class weight mode: {mode}")
    weights = torch.tensor([1 / counts[k] for k in LABEL_TO_ID], dtype=torch.float32)
    return weights / weights.mean()


def model_hash(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def epoch_order(size, seed, epoch):
    # Independent from workers/model/global RNG; one permutation, no resampling.
    return torch.randperm(size, generator=torch.Generator().manual_seed(seed + epoch)).tolist()


def order_hash(order):
    return hashlib.sha256(json.dumps(order).encode()).hexdigest()
