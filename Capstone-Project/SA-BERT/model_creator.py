from tensorflow import keras
import bert
import tensorflow_addons as tfa

from vocab_util import TargetVocabUtil


class BertModelCreator:
    def __init__(self, model_dir: str, tvu: TargetVocabUtil, max_seq_len: int = 128):
        bert_params = bert.params_from_pretrained_ckpt(model_dir)
        self.l_bert = bert.BertModelLayer.from_params(bert_params, name="bert")
        self.tvu = tvu
        self.max_seq_len = max_seq_len

    def create_model(self) -> keras.Model:
        l_input_ids = keras.layers.Input(shape=(self.max_seq_len,), dtype='int32')
        self.l_bert.trainable = False
        # using the default token_type/segment id 0
        bert_layer_output = self.l_bert(l_input_ids)  # output: [batch_size, max_seq_len, hidden_size]
        dense_layer = keras.layers.Dense(units=self.tvu.get_output_vocab_size(),
                                         activation="softmax", name="dense")
        output = dense_layer(bert_layer_output)
        cls_out = keras.layers.Lambda(lambda seq: seq[:, 0, :])(output)

        model = keras.Model(inputs=l_input_ids, outputs=cls_out)
        model.build(input_shape=(None, self.max_seq_len))

        model.summary()

        # LinCE paper uses AdamW(Loshchilov and Hutter, 2019)(η=5e−5, ε = 1e−8).
        opt = tfa.optimizers.AdamW(learning_rate=5e-5, weight_decay=1e-8)
        model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

        return model
