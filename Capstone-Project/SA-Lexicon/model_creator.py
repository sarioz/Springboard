import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from vocab_util import VocabUtil


class ModelCreator:
    def __init__(self, vu: VocabUtil, batch_size: int, features_dim: int = 3):
        self.vu = vu
        self.batch_size = batch_size
        self.features_dim = features_dim

    def create_two_dense_model(self, hidden_layer_size: int = 10) -> tf.keras.Model:
        model = keras.Sequential(
            layers=[
                tf.keras.Input(shape=(self.batch_size, self.features_dim), name="input"),
                layers.Dense(units=hidden_layer_size, name="hidden_dense"),
                layers.Dense(units=self.vu.get_output_vocab_size(), activation="softmax", name="output_dense"),
            ],
            name="sequential_two_dense_model"
        )
        model.summary()

        opt = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
        model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

        return model
