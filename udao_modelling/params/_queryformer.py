import attrs

from typing import Optional
from ._graph_transformer import GraphTransformerParams

@attrs.define
class QueryFormerParams(GraphTransformerParams):
    # TODO (glachaud): because of inheritance, I have to set default values...
    max_height: Optional[int] = 0
    max_dist: Optional[int] = 0