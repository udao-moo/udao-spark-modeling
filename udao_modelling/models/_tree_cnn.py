import attrs

from typing import List


@attrs.define
class TreeCNNStructuredParams:
    output_size: int
    type_embedding_dim: int
    hist_embedding_dim: int
    bitmap_embedding_dim: int
    n_layers: int
    hidden_dim: int
    dropout: float
    op_groups: List[str]
    tcnn_hidden_dim: int
