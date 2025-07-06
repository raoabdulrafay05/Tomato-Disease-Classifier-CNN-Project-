import tensorflow as tf

model = tf.keras.models.load_model(
    r"model\model_trained_to_31_epochs.h5"
)

MODEL_VERSION = "1.0.0"