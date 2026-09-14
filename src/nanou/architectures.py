"""The two architectures: the Nano-U student and the BU-Net teacher."""

from typing import Any

import tensorflow as tf
import tf_keras as keras
from tf_keras import Model, layers

from nanou.blocks import depthwise_sep_conv, triple_conv


@tf.keras.utils.register_keras_serializable(package="NanoU")
class PadToMatch(layers.Layer):
    """Center-crop the skip tensor to match the spatial dims of tnearesthe upsampled tensor.

    Cropping the (larger) encoder skip instead of zero-padding the upsampled
    decoder feature avoids introducing synthetic border values and keeps the
    decoder feature map spatially aligned with the encoder features.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        """Crop `inputs[1]` (the skip) to the spatial size of `inputs[0]`."""
        x_up, skip = inputs[0], inputs[1]
        x_h = tf.shape(x_up)[1]
        x_w = tf.shape(x_up)[2]
        crop_top = (tf.shape(skip)[1] - x_h) // 2
        crop_left = (tf.shape(skip)[2] - x_w) // 2
        return skip[:, crop_top : crop_top + x_h, crop_left : crop_left + x_w, :]


def create_nano_u(
    input_shape: tuple[int, int, int],
    filters: list[int],
    bottleneck: int,
    name: str = "nanou",
) -> Model:
    """Strictly sequential Nano-U encoder-decoder (no skip connections).

    Kept parameter-minimal by design; skip connections are omitted to reduce
    parameter count and avoid overfitting on small agricultural datasets.
    TFLite export is trivial - MCU-inference details live in ``firmware/``.

    The shape arguments are required: an architecture that silently defaults is
    an architecture nobody can read off the call site.

    Args:
        input_shape: (H, W, C) of the input image.
        filters: Three encoder widths, applied at full, half and quarter resolution.
        bottleneck: Channel width of the two bottleneck convolutions.
        name: Model name, which also names the exported artifacts.

    Returns:
        Uncompiled Keras Model.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    x = depthwise_sep_conv(inputs, filters[0], name="enc1a")
    x = depthwise_sep_conv(x, filters[0], name="enc1b")
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = depthwise_sep_conv(x, filters[1], name="enc2a")
    x = depthwise_sep_conv(x, filters[1], name="enc2b")
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = depthwise_sep_conv(x, filters[2], name="enc3a")
    x = depthwise_sep_conv(x, filters[2], name="enc3b")
    x = layers.MaxPooling2D(pool_size=(3, 2))(x)
    x = depthwise_sep_conv(x, bottleneck, name="bottleneck_a")
    x = depthwise_sep_conv(x, bottleneck, name="bottleneck_b")
    x = layers.UpSampling2D(size=(3, 2), interpolation="nearest", name="dec1_up")(x)
    x = depthwise_sep_conv(x, filters[2], name="dec1_conv_a")
    x = depthwise_sep_conv(x, filters[2], name="dec1_conv_b")
    x = layers.UpSampling2D(size=2, interpolation="nearest", name="dec2_up")(x)
    x = depthwise_sep_conv(x, filters[1], name="dec2_conv_a")
    x = depthwise_sep_conv(x, filters[1], name="dec2_conv_b")
    x = layers.UpSampling2D(size=2, interpolation="nearest", name="dec3_up")(x)
    x = depthwise_sep_conv(x, filters[0], name="dec3_conv_a")
    x = depthwise_sep_conv(x, filters[0], name="dec3_conv_b")
    outputs = layers.Conv2D(1, 1, padding="same", activation="linear", name="output")(x)

    return Model(inputs=inputs, outputs=outputs, name=name)


def create_bu_net(
    input_shape: tuple[int, int, int],
    filters: list[int],
    bottleneck: int,
    name: str = "bu_net",
) -> Model:
    """BU-Net teacher: a U-Net with skip connections, built with the functional API.

    Args:
        input_shape: (H, W, C) of the input image.
        filters: One encoder width per stage; the decoder mirrors them.
        bottleneck: Channel width of the bottleneck stage.
        name: Model name.

    Returns:
        Uncompiled Keras Model.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")
    num_stages = len(filters)

    encoder_outputs: list[tf.Tensor] = []
    x = inputs

    x = triple_conv(x, filters[0], name="enc1")
    encoder_outputs.append(x)
    x = layers.MaxPooling2D()(x)

    for i in range(1, num_stages):
        x = triple_conv(x, filters[i], name=f"enc{i + 1}")
        encoder_outputs.append(x)
        if i < num_stages - 1:
            x = layers.MaxPooling2D()(x)

    x = triple_conv(layers.MaxPooling2D()(x), bottleneck, name="bottleneck")

    def up_block(x_in, skip, out_filters, block_name):
        x_in = layers.UpSampling2D(size=2, interpolation="nearest", name=f"{block_name}_up")(x_in)
        skip_cropped = PadToMatch(name=f"{block_name}_pad")([x_in, skip])
        x_in = layers.Concatenate(name=f"{block_name}_concat")([skip_cropped, x_in])
        x_in = triple_conv(x_in, out_filters, name=f"{block_name}_conv")
        return x_in

    for i in range(num_stages - 1, -1, -1):
        x = up_block(x, encoder_outputs[i], filters[i], f"dec{num_stages - i}")

    outputs = layers.Conv2D(1, 1, activation="linear", padding="same", name="output_segmentation")(
        x
    )
    h, w = input_shape[0], input_shape[1]
    outputs = layers.Resizing(h, w, name="output_resize")(outputs)
    return Model(inputs=inputs, outputs=outputs, name=name)


def build_from_config(config: dict[str, Any]) -> keras.Model:
    """Instantiate one of the two architectures from a resolved config dict.

    Args:
        config: Must contain `model_name`, `filters`, `bottleneck` and
            `input_shape`. Every shipped preset sets all four; there are no
            fallbacks here on purpose, so a missing key fails loudly at build
            time instead of silently training a differently shaped model.

    Returns:
        Uncompiled Keras Model.

    Raises:
        KeyError: If any of the four required keys is missing or None.
    """
    required = ("model_name", "filters", "bottleneck", "input_shape")
    missing = [k for k in required if config.get(k) is None]
    if missing:
        raise KeyError(
            f"config is missing the architecture keys {missing}; "
            "every preset must set model_name, filters, bottleneck and input_shape"
        )

    model_name = config["model_name"]
    input_shape = config["input_shape"]
    if isinstance(input_shape, list):
        input_shape = tuple(input_shape)

    builder = create_bu_net if model_name == "bu_net" else create_nano_u
    return builder(
        input_shape=input_shape,
        filters=list(config["filters"]),
        bottleneck=int(config["bottleneck"]),
        name=model_name,
    )
