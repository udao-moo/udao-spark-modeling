from typing import Any, Tuple

import cattrs

from .model import Model
from ._learning import LearningParams
from udao_spark.model.utils import MyLearningParams
from udao_spark.utils.params import UdaoParams


# TODO (glachaud): fix the type hints of the function.
def get_model_and_learning_parameters(params: Any, model: Model, split_iterator_shape: Any) -> Tuple[UdaoParams, MyLearningParams]:
    unstructured_params = cattrs.unstructure(params)
    # TODO (glachaud): this remapping of parameters is temporary, to keep full backwards compatibility.
    model_params = cattrs.structure(unstructured_params, model.structured_params)
    learning_params = cattrs.structure(unstructured_params, LearningParams)
    # TODO (glachaud): this is necessary because `iterator_shape` has an unspecified type.
    model_params = cattrs.unstructure(model_params)
    model_params["iterator_shape"] = split_iterator_shape
    model_params = model.model_params.from_dict(model_params)
    learning_params = MyLearningParams.from_dict(cattrs.unstructure(learning_params))
    return model_params, learning_params
