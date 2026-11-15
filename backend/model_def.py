import keras
from keras import layers


def build_augmentation():
    return keras.Sequential([
        layers.RandomTranslation(0.05, 0.05, fill_mode="constant", interpolation="nearest", fill_value=255.0),
        layers.RandomRotation(0.01, fill_mode="constant", fill_value=255.0),
        layers.RandomZoom(0.05, fill_mode="constant", fill_value=255.0),
        layers.RandomContrast(0.1),
    ], name="augmentation")


def build_densenet121(height: int, width: int, num_classes: int):
    backbone = keras.applications.DenseNet121(
        include_top=False, weights=None, input_shape=(height, width, 3)
    )

    inputs = layers.Input(shape=(height, width, 1))
    x = build_augmentation()(inputs)  # tidak aktif saat inferensi, tapi harus ada agar bobot cocok
    x = layers.Concatenate()([x, x, x])
    x = layers.Rescaling(1.0 / 255)(x)
    x = layers.Normalization(
        mean=[0.485, 0.456, 0.406],
        variance=[0.229 ** 2, 0.224 ** 2, 0.225 ** 2],
    )(x)
    x = backbone(x, training=False)

    x = layers.Concatenate()([
        layers.GlobalAveragePooling2D()(x),
        layers.GlobalMaxPooling2D()(x),
    ])
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs, name="DenseNet121")
