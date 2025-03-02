from typing import Any

import torch as th
from udao.model import BaseEmbedder
from udao.utils.interfaces import UdaoEmbedItemShape


class DummyEmbedder(BaseEmbedder):
    @classmethod
    def from_iterator_shape(
        cls, iterator_shape: UdaoEmbedItemShape, **kwargs: Any
    ) -> "DummyEmbedder":
        return cls(cls.Params(**kwargs))

    def forward(self, input: th.Tensor) -> th.Tensor:
        return input
