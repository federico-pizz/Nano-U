"""Model factory: build Keras models from config dictionaries."""

from typing import Any, Dict

import tf_keras as keras

from src.models import create_nano_u, create_bu_net


def create_model_from_config(config: Dict[str, Any]) -> keras.Model:
    """Instantiate a model architecture from a hyperparameter config dict.

    Reads 'model_name', 'filters', 'bottleneck', and 'input_shape' from
    *config* and delegates to the appropriate builder function.

    Args:
        config: Dict that must contain at least 'model_name'. Optional keys:
            * filters (list[int]) — encoder filter counts
            * bottleneck (int) — bottleneck channel width
            * input_shape (tuple|list) — (H, W, C)

    Returns:
        Uncompiled Keras Model instance.
    """
    model_name = config.get("model_name", "nano_u")
    filters = config.get("filters", None)
    bottleneck = config.get("bottleneck", None)
    input_shape = config.get("input_shape", (60, 80, 3))

    if isinstance(input_shape, list):
        input_shape = tuple(input_shape)

    if model_name == "bu_net":
        return create_bu_net(
            input_shape=input_shape,
            filters=filters,
            bottleneck=bottleneck,
            name=model_name,
        )
    return create_nano_u(
        input_shape=input_shape,
        filters=filters,
        bottleneck=bottleneck,
        name=model_name,
    )
