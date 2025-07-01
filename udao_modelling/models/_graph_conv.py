import attrs

from typing import Optional, List


@attrs.define
class GraphConvStructuredParams:
    output_size: int
    readout: str
    gcn_n_layers: int
    type_embedding_dim: int
    hist_embedding_dim: int
    bitmap_embedding_dim: int
    embedding_normalizer: Optional[str]
    n_layers: int
    hidden_dim: int
    dropout: float
    activate: str
    use_batchnorm: bool
    op_groups: List[str]
