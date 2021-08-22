import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from vocab_util import VocabUtil


class LstmModelCreator:
    def __init__(self, vu: VocabUtil, embedding_dim: int = 128, lstm_dim: int = 256, mask_zero: bool = True):
        self.vu = vu
        self.embedding_dim = embedding_dim
        self.lstm_dim = lstm_dim
        self.mask_zero = mask_zero

    def create_uni_lstm_model(self) -> tf.keras.Model:
        model = keras.Sequential(
            layers=[
                layers.Embedding(input_dim=self.vu.get_input_vocab_size(),
                                 output_dim=self.embedding_dim, mask_zero=self.mask_zero, name="embedding"),
                layers.LSTM(units=self.lstm_dim, return_sequences=True, name="lstm"),
                layers.Dense(units=self.vu.get_output_vocab_size(), activation="softmax", name="dense"),
            ],
            name="sequential_lstm_model"
        )
        model.summary()

        opt = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
        model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

        return model

    def create_bi_lstm_model(self) -> tf.keras.Model:
        model = keras.Sequential(
            layers=[
                layers.Embedding(input_dim=self.vu.get_input_vocab_size(),
                                 output_dim=self.embedding_dim, mask_zero=self.mask_zero, name="embedding"),
                layers.Bidirectional(layers.LSTM(self.lstm_dim, return_sequences=True), name="bi_lstm"),
                layers.Dense(units=self.vu.get_output_vocab_size(), activation="softmax", name="dense"),
            ],
            name="sequential_bi_lstm_model"
        )
        model.summary()

        opt = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
        model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

        return model
