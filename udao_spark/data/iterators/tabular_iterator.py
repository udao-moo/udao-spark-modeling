from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch as th
import udao
from udao.data import TabularContainer
from udao.utils.interfaces import UdaoEmbedInput


@dataclass
class XFerInput(UdaoEmbedInput[th.Tensor]):
    """The embedding input is a dgl.DGLGraph"""

    def to(self, device: th.device) -> "XFerInput":
        return XFerInput(self.features.to(device), self.embedding_input.to(device))


class XFerTabularIterator(udao.data.TabularIterator):
    def __init__(
        self,
        keys: Sequence[str],
        embedding_features: TabularContainer,
        tabular_features: TabularContainer,
        objectives: TabularContainer,
    ):
        super().__init__(keys, tabular_features)
        self.embedding_features = embedding_features
        self.objectives = objectives

    def _getitem(self, idx: int) -> Tuple[XFerInput, th.Tensor]:
        key = self.keys[idx]
        embeddings = th.tensor(
            self.embedding_features.get(key), dtype=self.tensors_dtype
        )
        features = th.tensor(self.tabular_feature.get(key), dtype=self.tensors_dtype)
        objectives = th.tensor(self.objectives.get(key), dtype=self.tensors_dtype)
        return XFerInput(features, embeddings), objectives

    @staticmethod
    def collate(
        items: List[Tuple[XFerInput, th.Tensor]]
    ) -> Tuple[XFerInput, th.Tensor]:
        embeddings = th.vstack([item[0].embedding_input for item in items])
        features = th.vstack([item[0].features for item in items])
        objectives = th.vstack([item[1] for item in items])
        return XFerInput(features, embeddings), objectives
