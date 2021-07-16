import numpy as np
import tensorflow as tf

from tensorflow.keras.callbacks import ModelCheckpoint

from labeled_data_loader import LabeledDataLoader
from model_creator import LstmModelCreator
from nn_input_preparer import NNInputPreparer
from vocab_util import VocabUtil

BASE_DIR = f'models/01_uni_LSTM_masked_a_bigeta_oov/'

MAX_EPOCHS = 50


def main_training():
    loader = LabeledDataLoader('../data/pos/train.conll')
    tweets = loader.parse_tokens_and_labels(loader.load_lines())

    unique_input_tokens = set([item[0] for tweet in tweets for item in tweet])
    sorted_input_tokens = sorted(unique_input_tokens)

    print(len(sorted_input_tokens))

    vu = VocabUtil(sorted_input_tokens)

    irregular_inputs = [[vu.nn_input_token_to_int[item[0]] for item in tweet] for tweet in tweets]
    irregular_targets = [[vu.nn_pos_to_int[item[1]] for item in tweet] for tweet in tweets]

    nn_input_preparer = NNInputPreparer(vu)

    rectangular_inputs = nn_input_preparer.rectangularize_inputs(irregular_inputs)
    rectangular_targets = nn_input_preparer.rectangularize_targets(irregular_targets)

    targets_one_hot_encoded = nn_input_preparer.rectangular_targets_to_one_hot(rectangular_targets)

    model_creator = LstmModelCreator(vu)
    model = model_creator.create_lstm_model()

    cp_filepath = BASE_DIR + '{epoch}_{val_accuracy:.5f}.h5'

    checkpoint = ModelCheckpoint(cp_filepath, monitor='val_accuracy', verbose=1,
                                 save_best_only=False)

    model.fit(rectangular_inputs, targets_one_hot_encoded, batch_size=32, epochs=MAX_EPOCHS,
              validation_split=0.1, callbacks=[checkpoint])


if __name__ == '__main__':
    main_training()
