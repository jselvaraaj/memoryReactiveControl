from typing import Tuple, List, Dict, Any

from ray.train import Checkpoint


def get_best_checkpoint(checkpoints: List[Tuple["Checkpoint", Dict[str, Any]]], metric: str, mode: str) -> Checkpoint:
    op = max if mode == "max" else min

    path = metric.split('/')

    def extract_value_of_checkpoint(checkpoint_tup):
        nonlocal path
        checkpoint, checkpoint_config = checkpoint_tup
        val = checkpoint_config
        for node in path:
            val = val[node]
        return val

    best_checkpoint = op(checkpoints, key=extract_value_of_checkpoint)[0]

    return best_checkpoint
