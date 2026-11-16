"""Arsitektur DenseNet121 — port 1:1 dari notebook (bagian 4.6, build_model("DenseNet121")).
Urutan pembuatan layer harus sama persis dengan notebook agar load_weights cocok."""
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
    inputs = layers.Input(shape=(height, width, 1), name="grayscale_input")
    x = build_augmentation()(inputs)  # tidak aktif saat inferensi, tapi harus ada agar bobot cocok
    x = layers.Concatenate(name="gray_to_rgb")([x, x, x])

    x = layers.Rescaling(1 / 255, name="densenet_rescale")(x)
    x = layers.Normalization(
        mean=[0.485, 0.456, 0.406],
        variance=[0.229 ** 2, 0.224 ** 2, 0.225 ** 2],
        name="densenet_imagenet_normalization",
    )(x)
    features = backbone(x, training=False)
    projected = layers.Conv2D(
        256, kernel_size=1, padding="same", use_bias=False,
        name="densenet_feature_projection",
    )(features)
    projected = layers.BatchNormalization(name="densenet_projection_bn")(projected)
    projected = layers.Activation("relu", name="densenet_projection_relu")(projected)

    regional = layers.Resizing(2, 4, interpolation="bilinear", name="densenet_regional_pool")(projected)
    regional = layers.Flatten(name="densenet_regional_flatten")(regional)
    global_avg = layers.GlobalAveragePooling2D(name="densenet_global_avg")(projected)
    global_max = layers.GlobalMaxPooling2D(name="densenet_global_max")(projected)
    x = layers.Concatenate(name="densenet_feature_fusion")([regional, global_avg, global_max])
    x = layers.Dropout(0.30, name="densenet_feature_dropout")(x)
    x = layers.Dense(
        512, use_bias=False,
        kernel_regularizer=keras.regularizers.l2(5e-5),
        name="densenet_dense_512",
    )(x)
    x = layers.BatchNormalization(name="densenet_head_bn")(x)
    x = layers.Activation("relu", name="densenet_head_relu")(x)
    x = layers.Dropout(0.40, name="densenet_head_dropout")(x)
    x = layers.Dense(
        256, activation="relu",
        kernel_regularizer=keras.regularizers.l2(5e-5),
        name="densenet_dense_256",
    )(x)
    x = layers.Dropout(0.25, name="densenet_embedding_dropout")(x)

    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32", name="writer_classifier")(x)
    return keras.Model(inputs, outputs, name="DenseNet121")
