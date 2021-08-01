from tensorflow import keras
import bert

from vocab_util import TargetVocabUtil


class BertModelCreator:
    def __init__(self, model_dir: str, tvu: TargetVocabUtil, max_seq_len: int, freeze_bert_layer: bool):
        bert_params = bert.params_from_pretrained_ckpt(model_dir)
        self.l_bert = bert.BertModelLayer.from_params(bert_params, name="bert")
        self.vu = tvu
        self.max_seq_len = max_seq_len
        self.freeze_bert_layer = freeze_bert_layer

    def create_model(self) -> keras.Model:
        l_input_ids = keras.layers.Input(shape=(self.max_seq_len,), dtype='int32')
        if self.freeze_bert_layer:
            self.l_bert.trainable = False
        # using the default token_type/segment id 0
        bert_layer_output = self.l_bert(l_input_ids)  # output: [batch_size, max_seq_len, hidden_size]
        dense_layer = keras.layers.Dense(units=self.vu.get_output_vocab_size(),
                                         activation="softmax", name="dense")
        output = dense_layer(bert_layer_output)

        model = keras.Model(inputs=l_input_ids, outputs=output)
        model.build(input_shape=(None, self.max_seq_len))

        model.summary()

        opt = keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
        model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

        return model
