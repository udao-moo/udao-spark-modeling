from typing import Optional

from attrs import define

from .params import GraphSkipMLPParams


@define
class GraphConvParams(GraphSkipMLPParams):
    # Embedder parameters
    output_size: int = 32
    gcn_n_layers: int = 2
    readout: str = "mean"
    type_embedding_dim: int = 8
    hist_embedding_dim: int = 32
    bitmap_embedding_dim: int = 32
    embedding_normalizer: Optional[str] = None
    # Regressor parameters
    n_layers: int = 2
    hidden_dim: int = 32
    dropout: float = 0.1
