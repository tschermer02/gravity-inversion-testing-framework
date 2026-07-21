from __future__ import absolute_import, division, print_function, unicode_literals

from tensorflow.keras import layers, models
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

"""Neural-network architectures for gravity inversion.

Each function builds a Keras model that maps surface gravity-anomaly data
and, in most versions, grid-spacing information to a 3D subsurface model.

Tensor convention
-----------------
All convolutional tensors use ``channels_first`` format:
    (channels, height, width)

For the final models:
    gravity input:  (1, 64, 64)
    spacing input:  (2,) containing dx and dy
    model output:   (24, 64, 64)

The older model definitions are retained for comparison. The selected
architecture is ``inv_model7_3by3()``.
"""

# ---------------------------------------------------------------------------
# Early single-input baseline
# ---------------------------------------------------------------------------
def inv_model():
    """Build the earliest 20 x 20, single-input baseline model.

    The model compresses the gravity map with convolutions and pooling, then
    uses a fully connected layer to predict all 4,000 cells of a
    10 x 20 x 20 subsurface volume.
    """
    model = models.Sequential()

    # Define the expected input separately.
    model.add(layers.Input(shape=(1, 20, 20)))

    # First convolutional layer.
    model.add(
        layers.Conv2D(
            4,
            (2, 2),
            strides=(2, 2),
            activation="relu",
            data_format="channels_first",
        )
    )
    model.add(layers.AveragePooling2D(pool_size=(2, 2), data_format='channels_first'))
    model.add(layers.Conv2D(10, (2, 2), activation='relu', data_format="channels_first"))
    model.add(layers.Conv2D(10, (2, 2), activation='relu', padding='same', data_format="channels_first"))
    model.add(layers.Flatten())
    model.add(layers.Dense(4000, activation='sigmoid'))
    # model.add(layers.Flatten(input_shape=(1, 20, 20)))
    # model.add(layers.Dense(4000, activation='sigmoid'))
    model.add(layers.Reshape((10, 20, 20)))
    return model


# ---------------------------------------------------------------------------
# Two-input models that concatenate gravity features and grid spacing
# ---------------------------------------------------------------------------

def inv_model2():
    """Add dx and dy as a second input and concatenate them with CNN features."""
    # Surface gravity anomaly: one 20 x 20 channel.
    anomal_input = Input(shape=(1, 20, 20), name='anomal_input')
    # Two scalar acquisition/grid parameters, normally dx and dy.
    dxdy_input = Input(shape=(2,), name='dxdy_input')

    # Gravity branch: extract spatial features from the anomaly map.
    x = layers.Conv2D(4, (2, 2), activation='relu', strides=(2, 2), data_format="channels_first")(anomal_input)
    x = layers.AveragePooling2D(pool_size=(2, 2), data_format='channels_first')(x)
    x = layers.Conv2D(10, (2, 2), activation='relu', data_format="channels_first")(x)
    x = layers.Conv2D(10, (2, 2), activation='relu', padding='same', data_format="channels_first")(x)
    # Flatten spatial feature maps into one feature vector.
    x = layers.Flatten()(x)

    # Join acquisition geometry with learned gravity features.
    y = layers.concatenate([dxdy_input, x])
    y = layers.Dense(160, activation='relu')(y)
    y = layers.Dense(160, activation='relu')(y)

    y = layers.Dense(4000, activation='sigmoid')(y)
    # Give the final prediction its intended depth x height x width shape.
    model_output = layers.Reshape((10, 20, 20), name='model_output')(y)
    # Package the connected computation graph as a Keras model.
    model = Model(inputs=[anomal_input, dxdy_input], outputs=model_output)
    return model


def inv_model3():
    """Scale the concatenation approach to 64 x 64 input and 24 depth layers."""
    # Surface gravity anomaly: one 64 x 64 channel.
    anomal_input = Input(shape=(1, 64, 64), name='anomal_input')
    # Two scalar acquisition/grid parameters, normally dx and dy.
    dxdy_input = Input(shape=(2,), name='dxdy_input')

    # Gravity branch: extract spatial features from the anomaly map.
    x = layers.Conv2D(8, (2, 2), activation='relu', padding='same', data_format="channels_first")(anomal_input)
    x = layers.AveragePooling2D(pool_size=(2, 2), data_format='channels_first')(x)
    x = layers.Conv2D(16, (2, 2), activation='relu',padding='same', data_format="channels_first")(x)
    x = layers.AveragePooling2D(pool_size=(2, 2), data_format='channels_first')(x)
    x = layers.Conv2D(32, (2, 2), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.AveragePooling2D(pool_size=(2, 2), data_format='channels_first')(x)
    x = layers.Conv2D(64, (2, 2), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.AveragePooling2D(pool_size=(2, 2), data_format='channels_first')(x)
    # Flatten spatial feature maps into one feature vector.
    x = layers.Flatten()(x)

    # Join acquisition geometry with learned gravity features.
    y = layers.concatenate([dxdy_input, x])
    y = layers.Dense(400, activation='relu')(y)
    y = layers.Dense(400, activation='relu')(y)

    y = layers.Dense(98304, activation='sigmoid')(y)
    # Give the final prediction its intended depth x height x width shape.
    model_output = layers.Reshape((24, 64, 64), name='model_output')(y)
    # Package the connected computation graph as a Keras model.
    model = Model(inputs=[anomal_input, dxdy_input], outputs=model_output)
    return model


def inv_model4():
    """Predict each of the 10 depth slices with a separate dense branch."""
    # Surface gravity anomaly: one 20 x 20 channel.
    anomal_input = Input(shape=(1, 20, 20), name='anomal_input')
    # Two scalar acquisition/grid parameters, normally dx and dy.
    dxdy_input = Input(shape=(2,), name='dxdy_input')

    # Gravity branch: extract spatial features from the anomaly map.
    x = layers.Conv2D(4, (2, 2), activation='relu', strides=(2, 2), data_format="channels_first")(anomal_input)
    x = layers.AveragePooling2D(pool_size=(2, 2), data_format='channels_first')(x)
    x = layers.Conv2D(10, (2, 2), activation='relu', data_format="channels_first")(x)
    x = layers.Conv2D(10, (2, 2), activation='relu', padding='same', data_format="channels_first")(x)
    # Flatten spatial feature maps into one feature vector.
    x = layers.Flatten()(x)

    # Join acquisition geometry with learned gravity features.
    y = layers.concatenate([dxdy_input, x])
    y1 = layers.Dense(40, activation='relu')(y)
    y2 = layers.Dense(40, activation='relu')(y)
    y3 = layers.Dense(40, activation='relu')(y)
    y4 = layers.Dense(40, activation='relu')(y)
    y5 = layers.Dense(40, activation='relu')(y)
    y6 = layers.Dense(40, activation='relu')(y)
    y7 = layers.Dense(40, activation='relu')(y)
    y8 = layers.Dense(40, activation='relu')(y)
    y9 = layers.Dense(40, activation='relu')(y)
    y10 = layers.Dense(40, activation='relu')(y)

    y1 = layers.Dense(400, activation='sigmoid')(y1)
    y2 = layers.Dense(400, activation='sigmoid')(y2)
    y3 = layers.Dense(400, activation='sigmoid')(y3)
    y4 = layers.Dense(400, activation='sigmoid')(y4)
    y5 = layers.Dense(400, activation='sigmoid')(y5)
    y6 = layers.Dense(400, activation='sigmoid')(y6)
    y7 = layers.Dense(400, activation='sigmoid')(y7)
    y8 = layers.Dense(400, activation='sigmoid')(y8)
    y9 = layers.Dense(400, activation='sigmoid')(y9)
    y10 = layers.Dense(400, activation='sigmoid')(y10)

    y = layers.concatenate([y1, y2, y3, y4, y5, y6, y7, y8, y9, y10])

    # y = layers.Dense(4000, activation='sigmoid')(y)
    # Give the final prediction its intended depth x height x width shape.
    model_output = layers.Reshape((10, 20, 20), name='model_output')(y)
    # Package the connected computation graph as a Keras model.
    model = Model(inputs=[anomal_input, dxdy_input], outputs=model_output)
    return model


# ---------------------------------------------------------------------------
# Multiplicative two-branch models
# ---------------------------------------------------------------------------

def inv_model5():
    """Combine a gravity branch and a spacing branch by elementwise multiplication.

    The gravity branch estimates spatial structure. The dx/dy branch creates a
    second 3D tensor that gates or rescales that structure.
    """
    # Surface gravity anomaly: one 20 x 20 channel.
    anomal_input = Input(shape=(1, 20, 20), name='anomal_input')
    # Two scalar acquisition/grid parameters, normally dx and dy.
    dxdy_input = Input(shape=(2,), name='dxdy_input')

    # Gravity branch: extract spatial features from the anomaly map.
    x = layers.Conv2D(2, (2, 2), activation='relu', padding='same', data_format="channels_first")(anomal_input)
    x = layers.Conv2D(4, (2, 2), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(6, (2, 2), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(8, (2, 2), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(10, (2, 2), activation='sigmoid', padding='same', data_format="channels_first")(x)

    # Spacing branch: transform grid dimensions into learned conditioning features.
    y = layers.Dense(10, activation='relu')(dxdy_input)
    y = layers.Dense(25, activation='relu')(y)
    y = layers.Dense(50, activation='relu')(y)
    y = layers.Dense(4000, activation='sigmoid')(y)
    # Convert the dense spacing representation back into a spatial tensor.
    y = layers.Reshape((10, 20, 20))(y)

    # Elementwise gating: both branches must agree at every output cell.
    z = layers.Multiply()([x, y])
    # Give the final prediction its intended depth x height x width shape.
    model_output = layers.Reshape((10, 20, 20), name='model_output')(z)
    # Package the connected computation graph as a Keras model.
    model = Model(inputs=[anomal_input, dxdy_input], outputs=model_output)
    return model


def inv_model5_train_model():
    """Train only the gravity branch; freeze the dx/dy branch."""
    # Surface gravity anomaly: one 20 x 20 channel.
    anomal_input = Input(shape=(1, 20, 20), name='anomal_input')
    # Two scalar acquisition/grid parameters, normally dx and dy.
    dxdy_input = Input(shape=(2,), name='dxdy_input')

    # Gravity branch: extract spatial features from the anomaly map.
    x = layers.Conv2D(2, (2, 2), activation='relu', padding='same', data_format="channels_first")(anomal_input)
    x = layers.Conv2D(4, (2, 2), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(6, (2, 2), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(8, (2, 2), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(10, (2, 2), activation='sigmoid', padding='same', data_format="channels_first")(x)

    # Spacing branch: transform grid dimensions into learned conditioning features.
    y = layers.Dense(10, activation='relu', trainable=False)(dxdy_input)
    y = layers.Dense(25, activation='relu', trainable=False)(y)
    y = layers.Dense(50, activation='relu', trainable=False)(y)
    y = layers.Dense(4000, activation='sigmoid', trainable=False)(y)
    # Convert the dense spacing representation back into a spatial tensor.
    y = layers.Reshape((10, 20, 20))(y)

    # Elementwise gating: both branches must agree at every output cell.
    z = layers.Multiply()([x, y])
    # Give the final prediction its intended depth x height x width shape.
    model_output = layers.Reshape((10, 20, 20), name='model_output')(z)
    # Package the connected computation graph as a Keras model.
    model = Model(inputs=[anomal_input, dxdy_input], outputs=model_output)
    return model


def inv_model5_train_dxdy():
    """Train only the dx/dy branch; freeze the gravity branch."""
    # Surface gravity anomaly: one 20 x 20 channel.
    anomal_input = Input(shape=(1, 20, 20), name='anomal_input')
    # Two scalar acquisition/grid parameters, normally dx and dy.
    dxdy_input = Input(shape=(2,), name='dxdy_input')

    # Gravity branch: extract spatial features from the anomaly map.
    x = layers.Conv2D(2, (2, 2), activation='relu', padding='same', trainable=False, data_format="channels_first")(anomal_input)
    x = layers.Conv2D(4, (2, 2), activation='relu', padding='same', trainable=False, data_format="channels_first")(x)
    x = layers.Conv2D(6, (2, 2), activation='relu', padding='same', trainable=False, data_format="channels_first")(x)
    x = layers.Conv2D(8, (2, 2), activation='relu', padding='same', trainable=False, data_format="channels_first")(x)
    x = layers.Conv2D(10, (2, 2), activation='sigmoid', padding='same', trainable=False, data_format="channels_first")(x)

    # Spacing branch: transform grid dimensions into learned conditioning features.
    y = layers.Dense(10, activation='relu')(dxdy_input)
    y = layers.Dense(25, activation='relu')(y)
    y = layers.Dense(50, activation='relu')(y)
    y = layers.Dense(4000, activation='sigmoid')(y)
    # Convert the dense spacing representation back into a spatial tensor.
    y = layers.Reshape((10, 20, 20))(y)

    # Elementwise gating: both branches must agree at every output cell.
    z = layers.Multiply()([x, y])
    # Give the final prediction its intended depth x height x width shape.
    model_output = layers.Reshape((10, 20, 20), name='model_output')(z)
    # Package the connected computation graph as a Keras model.
    model = Model(inputs=[anomal_input, dxdy_input], outputs=model_output)
    return model


def inv_model6():
    """Convert dx/dy into a 2D map before expanding it to the output volume."""
    # Surface gravity anomaly: one 20 x 20 channel.
    anomal_input = Input(shape=(1, 20, 20), name='anomal_input')
    # Two scalar acquisition/grid parameters, normally dx and dy.
    dxdy_input = Input(shape=(2,), name='dxdy_input')

    # Gravity branch: extract spatial features from the anomaly map.
    x = layers.Conv2D(2, (2, 2), activation='relu', padding='same', data_format="channels_first")(anomal_input)
    x = layers.Conv2D(4, (2, 2), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(6, (2, 2), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(8, (2, 2), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(10, (2, 2), activation='sigmoid', padding='same', data_format="channels_first")(x)

    # Spacing branch: transform grid dimensions into learned conditioning features.
    y = layers.Dense(10, activation='relu')(dxdy_input)
    y = layers.Dense(25, activation='relu')(y)
    y = layers.Dense(50, activation='relu')(y)
    y = layers.Dense(400, activation='relu')(y)

    # Convert the dense spacing representation back into a spatial tensor.
    y = layers.Reshape((1, 20, 20))(y)
    y = layers.Conv2D(5, (2, 2), activation='relu', padding='same', data_format="channels_first")(y)
    y = layers.Conv2D(10, (2, 2), activation='relu', padding='same', data_format="channels_first")(y)

    # Elementwise gating: both branches must agree at every output cell.
    z = layers.Multiply()([x, y])
    # Give the final prediction its intended depth x height x width shape.
    model_output = layers.Reshape((10, 20, 20), name='model_output')(z)
    # Package the connected computation graph as a Keras model.
    model = Model(inputs=[anomal_input, dxdy_input], outputs=model_output)
    return model


def inv_model6_train_model():
    """Train the gravity branch of model 6 while freezing the spacing branch."""
    # Surface gravity anomaly: one 20 x 20 channel.
    anomal_input = Input(shape=(1, 20, 20), name='anomal_input')
    # Two scalar acquisition/grid parameters, normally dx and dy.
    dxdy_input = Input(shape=(2,), name='dxdy_input')

    # Gravity branch: extract spatial features from the anomaly map.
    x = layers.Conv2D(2, (3, 3), activation='relu', padding='same', data_format="channels_first")(anomal_input)
    x = layers.Conv2D(4, (3, 3), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(6, (3, 3), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(10, (3, 3), activation='sigmoid', padding='same', data_format="channels_first")(x)

    # Spacing branch: transform grid dimensions into learned conditioning features.
    y = layers.Dense(10, trainable=False, activation='relu')(dxdy_input)
    y = layers.Dense(25, trainable=False, activation='relu')(y)
    y = layers.Dense(50, trainable=False, activation='relu')(y)
    y = layers.Dense(400, trainable=False, activation='relu')(y)

    # Convert the dense spacing representation back into a spatial tensor.
    y = layers.Reshape((1, 20, 20))(y)
    y = layers.Conv2D(5, (3, 3), activation='relu', trainable=False, padding='same', data_format="channels_first")(y)
    y = layers.Conv2D(10, (3, 3), activation='relu', trainable=False, padding='same', data_format="channels_first")(y)

    # Elementwise gating: both branches must agree at every output cell.
    z = layers.Multiply()([x, y])
    # Give the final prediction its intended depth x height x width shape.
    model_output = layers.Reshape((10, 20, 20), name='model_output')(z)
    # Package the connected computation graph as a Keras model.
    model = Model(inputs=[anomal_input, dxdy_input], outputs=model_output)
    return model


def inv_model6_train_dxdy():
    """Train the spacing branch of model 6 while freezing the gravity branch."""
    # Surface gravity anomaly: one 20 x 20 channel.
    anomal_input = Input(shape=(1, 20, 20), name='anomal_input')
    # Two scalar acquisition/grid parameters, normally dx and dy.
    dxdy_input = Input(shape=(2,), name='dxdy_input')

    # Gravity branch: extract spatial features from the anomaly map.
    x = layers.Conv2D(2, (3, 3), trainable=False, activation='relu', padding='same', data_format="channels_first")(anomal_input)
    x = layers.Conv2D(4, (3, 3), trainable=False, activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(6, (3, 3), trainable=False, activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(8, (3, 3), trainable=False, activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(10, (3, 3), trainable=False, activation='sigmoid', padding='same', data_format="channels_first")(x)

    # Spacing branch: transform grid dimensions into learned conditioning features.
    y = layers.Dense(10, activation='relu')(dxdy_input)
    y = layers.Dense(25, activation='relu')(y)
    y = layers.Dense(50, activation='relu')(y)
    y = layers.Dense(400, activation='relu')(y)

    # Convert the dense spacing representation back into a spatial tensor.
    y = layers.Reshape((1, 20, 20))(y)
    y = layers.Conv2D(5, (3, 3), activation='relu', padding='same', data_format="channels_first")(y)
    y = layers.Conv2D(10, (3, 3), activation='relu', padding='same', data_format="channels_first")(y)

    # Elementwise gating: both branches must agree at every output cell.
    z = layers.Multiply()([x, y])
    # Give the final prediction its intended depth x height x width shape.
    model_output = layers.Reshape((10, 20, 20), name='model_output')(z)
    # Package the connected computation graph as a Keras model.
    model = Model(inputs=[anomal_input, dxdy_input], outputs=model_output)
    return model



# ---------------------------------------------------------------------------
# 64 x 64 models producing a 24 x 64 x 64 subsurface volume
# ---------------------------------------------------------------------------

def inv_model7_relu():
    """Model 7 variant whose spacing branch ends with ReLU."""
    # Surface gravity anomaly: one 64 x 64 channel.
    anomal_input = Input(shape=(1, 64, 64), name='anomal_input')
    # Two scalar acquisition/grid parameters, normally dx and dy.
    dxdy_input = Input(shape=(2,), name='dxdy_input')

    # Gravity branch: extract spatial features from the anomaly map.
    x = layers.Conv2D(2, (3, 3), activation='relu', padding='same', data_format="channels_first")(anomal_input)
    x = layers.Conv2D(4, (3, 3), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(24, (3, 3), activation='sigmoid', padding='same', data_format="channels_first")(x)

    # Spacing branch: transform grid dimensions into learned conditioning features.
    y = layers.Dense(4, activation='relu')(dxdy_input)
    y = layers.Dense(64, activation='relu')(y)
    y = layers.Dense(128, activation='relu')(y)
    y = layers.Dense(4096, activation='relu')(y)

    # Convert the dense spacing representation back into a spatial tensor.
    y = layers.Reshape((1, 64, 64))(y)
    y = layers.Conv2D(8, (3, 3), activation='relu', padding='same', data_format="channels_first")(y)
    y = layers.Conv2D(24, (3, 3), activation='relu', padding='same', data_format="channels_first")(y)

    # Elementwise gating: both branches must agree at every output cell.
    z = layers.Multiply()([x, y])
    # Give the final prediction its intended depth x height x width shape.
    model_output = layers.Reshape((24, 64, 64), name='model_output')(z)
    # Package the connected computation graph as a Keras model.
    model = Model(inputs=[anomal_input, dxdy_input], outputs=model_output)
    return model


def inv_model7_3by3():
    """Build the selected model using 3 x 3 convolution kernels.

    Branch x learns spatial features from the measured gravity anomaly.
    Branch y converts dx and dy into a spatial/depth-dependent gating tensor.
    Their elementwise product is the predicted 3D density model.
    """
    # Surface gravity anomaly: one 64 x 64 channel.
    anomal_input = Input(shape=(1, 64, 64), name='anomal_input')
    # Two scalar acquisition/grid parameters, normally dx and dy.
    dxdy_input = Input(shape=(2,), name='dxdy_input')

    # Gravity branch: extract spatial features from the anomaly map.
    x = layers.Conv2D(2, (3, 3), activation='relu', padding='same', data_format="channels_first")(anomal_input)
    x = layers.Conv2D(4, (3, 3), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(24, (3, 3), activation='sigmoid', padding='same', data_format="channels_first")(x)

    # Spacing branch: transform grid dimensions into learned conditioning features.
    y = layers.Dense(4, activation='relu')(dxdy_input)
    y = layers.Dense(64, activation='relu')(y)
    y = layers.Dense(128, activation='relu')(y)
    y = layers.Dense(4096, activation='relu')(y)

    # Convert the dense spacing representation back into a spatial tensor.
    y = layers.Reshape((1, 64, 64))(y)
    y = layers.Conv2D(8, (3, 3), activation='relu', padding='same', data_format="channels_first")(y)
    y = layers.Conv2D(24, (3, 3), activation='sigmoid', padding='same', data_format="channels_first")(y)

    # Elementwise gating: both branches must agree at every output cell.
    z = layers.Multiply()([x, y])
    # Give the final prediction its intended depth x height x width shape.
    model_output = layers.Reshape((24, 64, 64), name='model_output')(z)
    # Package the connected computation graph as a Keras model.
    model = Model(inputs=[anomal_input, dxdy_input], outputs=model_output)
    return model

def inv_model7_5by5():
    """Model 7 comparison variant using wider 5 x 5 kernels."""
    # Surface gravity anomaly: one 64 x 64 channel.
    anomal_input = Input(shape=(1, 64, 64), name='anomal_input')
    # Two scalar acquisition/grid parameters, normally dx and dy.
    dxdy_input = Input(shape=(2,), name='dxdy_input')

    # Gravity branch: extract spatial features from the anomaly map.
    x = layers.Conv2D(2, (5, 5), activation='relu', padding='same', data_format="channels_first")(anomal_input)
    x = layers.Conv2D(4, (5, 5), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(8, (5, 5), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(16, (5, 5), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(24, (5, 5), activation='sigmoid', padding='same', data_format="channels_first")(x)

    # Spacing branch: transform grid dimensions into learned conditioning features.
    y = layers.Dense(4, activation='relu')(dxdy_input)
    y = layers.Dense(64, activation='relu')(y)
    y = layers.Dense(128, activation='relu')(y)
    y = layers.Dense(4096, activation='relu')(y)

    # Convert the dense spacing representation back into a spatial tensor.
    y = layers.Reshape((1, 64, 64))(y)
    y = layers.Conv2D(8, (5, 5), activation='relu', padding='same', data_format="channels_first")(y)
    y = layers.Conv2D(24, (5, 5), activation='sigmoid', padding='same', data_format="channels_first")(y)

    # Elementwise gating: both branches must agree at every output cell.
    z = layers.Multiply()([x, y])
    # Give the final prediction its intended depth x height x width shape.
    model_output = layers.Reshape((24, 64, 64), name='model_output')(z)
    # Package the connected computation graph as a Keras model.
    model = Model(inputs=[anomal_input, dxdy_input], outputs=model_output)
    return model

def inv_model7_7by7():
    """Model 7 comparison variant using 7 x 7 kernels."""
    # Surface gravity anomaly: one 64 x 64 channel.
    anomal_input = Input(shape=(1, 64, 64), name='anomal_input')
    # Two scalar acquisition/grid parameters, normally dx and dy.
    dxdy_input = Input(shape=(2,), name='dxdy_input')

    # Gravity branch: extract spatial features from the anomaly map.
    x = layers.Conv2D(2, (7, 7), activation='relu', padding='same', data_format="channels_first")(anomal_input)
    x = layers.Conv2D(4, (7, 7), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(8, (7, 7), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(16, (7, 7), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(24, (7, 7), activation='sigmoid', padding='same', data_format="channels_first")(x)

    # Spacing branch: transform grid dimensions into learned conditioning features.
    y = layers.Dense(4, activation='relu')(dxdy_input)
    y = layers.Dense(64, activation='relu')(y)
    y = layers.Dense(128, activation='relu')(y)
    y = layers.Dense(4096, activation='relu')(y)

    # Convert the dense spacing representation back into a spatial tensor.
    y = layers.Reshape((1, 64, 64))(y)
    y = layers.Conv2D(8, (7, 7), activation='relu', padding='same', data_format="channels_first")(y)
    y = layers.Conv2D(24, (7, 7), activation='sigmoid', padding='same', data_format="channels_first")(y)

    # Elementwise gating: both branches must agree at every output cell.
    z = layers.Multiply()([x, y])
    # Give the final prediction its intended depth x height x width shape.
    model_output = layers.Reshape((24, 64, 64), name='model_output')(z)
    # Package the connected computation graph as a Keras model.
    model = Model(inputs=[anomal_input, dxdy_input], outputs=model_output)
    return model


def inv_model7_9by9():
    """Model 7 comparison variant using 9 x 9 kernels."""
    # Surface gravity anomaly: one 64 x 64 channel.
    anomal_input = Input(shape=(1, 64, 64), name='anomal_input')
    # Two scalar acquisition/grid parameters, normally dx and dy.
    dxdy_input = Input(shape=(2,), name='dxdy_input')

    # Gravity branch: extract spatial features from the anomaly map.
    x = layers.Conv2D(2, (9, 9), activation='relu', padding='same', data_format="channels_first")(anomal_input)
    x = layers.Conv2D(4, (9, 9), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(8, (9, 9), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(16, (9, 9), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(24, (9, 9), activation='sigmoid', padding='same', data_format="channels_first")(x)

    # Spacing branch: transform grid dimensions into learned conditioning features.
    y = layers.Dense(4, activation='relu')(dxdy_input)
    y = layers.Dense(64, activation='relu')(y)
    y = layers.Dense(128, activation='relu')(y)
    y = layers.Dense(4096, activation='relu')(y)

    # Convert the dense spacing representation back into a spatial tensor.
    y = layers.Reshape((1, 64, 64))(y)
    y = layers.Conv2D(8, (9, 9), activation='relu', padding='same', data_format="channels_first")(y)
    y = layers.Conv2D(24, (9, 9), activation='sigmoid', padding='same', data_format="channels_first")(y)

    # Elementwise gating: both branches must agree at every output cell.
    z = layers.Multiply()([x, y])
    # Give the final prediction its intended depth x height x width shape.
    model_output = layers.Reshape((24, 64, 64), name='model_output')(z)
    # Package the connected computation graph as a Keras model.
    model = Model(inputs=[anomal_input, dxdy_input], outputs=model_output)
    return model


def inv_model7_dz():
    """Model 7 variant that conditions on dx, dy, and dz instead of only dx/dy."""
    # Surface gravity anomaly: one 64 x 64 channel.
    anomal_input = Input(shape=(1, 64, 64), name='anomal_input')
    # Three scalar grid dimensions: dx, dy, and dz.
    dxdydz_input = Input(shape=(3,), name='dxdydz_input')

    # Gravity branch: extract spatial features from the anomaly map.
    x = layers.Conv2D(2, (3, 3), activation='relu', padding='same', data_format="channels_first")(anomal_input)
    x = layers.Conv2D(4, (3, 3), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(24, (3, 3), activation='sigmoid', padding='same', data_format="channels_first")(x)

    # Spacing branch: transform grid dimensions into learned conditioning features.
    y = layers.Dense(12, activation='relu')(dxdydz_input)
    y = layers.Dense(64, activation='relu')(y)
    y = layers.Dense(128, activation='relu')(y)
    y = layers.Dense(4096, activation='relu')(y)

    # Convert the dense spacing representation back into a spatial tensor.
    y = layers.Reshape((1, 64, 64))(y)
    y = layers.Conv2D(8, (3, 3), activation='relu', padding='same', data_format="channels_first")(y)
    y = layers.Conv2D(24, (3, 3), activation='relu', padding='same', data_format="channels_first")(y)

    # Elementwise gating: both branches must agree at every output cell.
    z = layers.Multiply()([x, y])
    # Give the final prediction its intended depth x height x width shape.
    model_output = layers.Reshape((24, 64, 64), name='model_output')(z)
    # Package the connected computation graph as a Keras model.
    model = Model(inputs=[anomal_input, dxdydz_input], outputs=model_output)
    return model


def inv_model7_train_model():
    """Train only model 7's gravity branch; freeze its spacing branch."""
    # Surface gravity anomaly: one 64 x 64 channel.
    anomal_input = Input(shape=(1, 64, 64), name='anomal_input')
    # Two scalar acquisition/grid parameters, normally dx and dy.
    dxdy_input = Input(shape=(2,), name='dxdy_input')

    # Gravity branch: extract spatial features from the anomaly map.
    x = layers.Conv2D(2, (3, 3), activation='relu', padding='same', data_format="channels_first")(anomal_input)
    x = layers.Conv2D(4, (3, 3), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(24, (3, 3), activation='sigmoid', padding='same', data_format="channels_first")(x)

    # Spacing branch: transform grid dimensions into learned conditioning features.
    y = layers.Dense(4, trainable=False, activation='relu')(dxdy_input)
    y = layers.Dense(64, trainable=False, activation='relu')(y)
    y = layers.Dense(128, trainable=False, activation='relu')(y)
    y = layers.Dense(4096, trainable=False, activation='relu')(y)

    # Convert the dense spacing representation back into a spatial tensor.
    y = layers.Reshape((1, 64, 64))(y)
    y = layers.Conv2D(8, (3, 3), trainable=False, activation='relu', padding='same', data_format="channels_first")(y)
    y = layers.Conv2D(24, (3, 3), trainable=False, activation='relu', padding='same', data_format="channels_first")(y)

    # Elementwise gating: both branches must agree at every output cell.
    z = layers.Multiply()([x, y])
    # Give the final prediction its intended depth x height x width shape.
    model_output = layers.Reshape((24, 64, 64), name='model_output')(z)
    # Package the connected computation graph as a Keras model.
    model = Model(inputs=[anomal_input, dxdy_input], outputs=model_output)
    return model


def inv_model7_train_dxdy():
    """Train only model 7's dx/dy branch; freeze its gravity branch."""
    # Surface gravity anomaly: one 64 x 64 channel.
    anomal_input = Input(shape=(1, 64, 64), name='anomal_input')
    # Two scalar acquisition/grid parameters, normally dx and dy.
    dxdy_input = Input(shape=(2,), name='dxdy_input')

    # Gravity branch: extract spatial features from the anomaly map.
    x = layers.Conv2D(2, (3, 3), trainable=False, activation='relu', padding='same', data_format="channels_first")(anomal_input)
    x = layers.Conv2D(4, (3, 3), trainable=False, activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(8, (3, 3), trainable=False, activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(16, (3, 3), trainable=False, activation='relu', padding='same', data_format="channels_first")(x)
    x = layers.Conv2D(24, (3, 3), trainable=False, activation='sigmoid', padding='same', data_format="channels_first")(x)

    # Spacing branch: transform grid dimensions into learned conditioning features.
    y = layers.Dense(4, activation='relu')(dxdy_input)
    y = layers.Dense(64, activation='relu')(y)
    y = layers.Dense(128, activation='relu')(y)
    y = layers.Dense(4096, activation='relu')(y)

    # Convert the dense spacing representation back into a spatial tensor.
    y = layers.Reshape((1, 64, 64))(y)
    y = layers.Conv2D(8, (3, 3), activation='relu', padding='same', data_format="channels_first")(y)
    y = layers.Conv2D(24, (3, 3), activation='relu', padding='same', data_format="channels_first")(y)

    # Elementwise gating: both branches must agree at every output cell.
    z = layers.Multiply()([x, y])
    # Give the final prediction its intended depth x height x width shape.
    model_output = layers.Reshape((24, 64, 64), name='model_output')(z)
    # Package the connected computation graph as a Keras model.
    model = Model(inputs=[anomal_input, dxdy_input], outputs=model_output)
    return model