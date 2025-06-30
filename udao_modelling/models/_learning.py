import attrs
from typing import List, Optional


@attrs.define
class LearningParams:
    init_lr: float
    min_lr: float
    weight_decay: float
    loss_weights: List[int]
    epochs: int
    batch_size: int


def check_loss_objective_consistency(
    loss_weights: Optional[List[int]], objectives: List[str]
) -> None:
    if loss_weights is None:
        return
    if len(loss_weights) != len(objectives):
        raise ValueError("Loss weights must have the same length as objectives")
