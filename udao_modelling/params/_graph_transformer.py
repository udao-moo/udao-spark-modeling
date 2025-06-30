from typing import Optional

from attrs import define

from .params import GraphSkipMLPParams


@define
class GraphTransformerParams(GraphSkipMLPParams):
    # Embedder parameters
    attention_layer_name: str = ""
    output_size: int = 32
    pos_encoding_dim: int = 8
    gtn_n_layers: int = 2
    gtn_n_heads: int = 2
    readout: str = "mean"
    type_embedding_dim: int = 8
    hist_embedding_dim: int = 32
    bitmap_embedding_dim: int = 32
    embedding_normalizer: Optional[str] = None
    gtn_dropout: float = 0.0
    # Regressor parameters
    n_layers: int = 2
    hidden_dim: int = 32
    dropout: float = 0.1
