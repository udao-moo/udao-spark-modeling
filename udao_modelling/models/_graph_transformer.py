import attrs

from typing import Optional, List
from ._learning import LearningParams


@attrs.define
class GraphTransformerStructuredParams:
    output_size: int
    pos_encoding_dim: int
    gtn_n_layers: int
    gtn_n_heads: int
    readout: str
    type_embedding_dim: int
    hist_embedding_dim: int
    bitmap_embedding_dim: int
    embedding_normalizer: Optional[str]
    gtn_dropout: float
    n_layers: int
    hidden_dim: int
    dropout: float
    activate: str
    use_batchnorm: bool
    op_groups: List[str]
    attention_layer_name: str
