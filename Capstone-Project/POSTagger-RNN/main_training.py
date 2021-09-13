from typing import Tuple

import tensorflow as tf

import numpy as np

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

from labeled_data_loader import LabeledDataLoader
from model_creator import LstmModelCreator
from nn_input_preparer import NNInputPreparer
from vocab_util import VocabUtil


MAX_SEQ_LEN = 128
EMBEDDING_DIM = 64
LSTM_DIM = 64
MASK_ZERO = True

EXPERIMENT_NAME = f'08_mzt_bi_LSTM_{EMBEDDING_DIM}_{LSTM_DIM}'
MAX_EPOCHS = 100

BASE_DIR = f'models/{EXPERIMENT_NAME}/'

CONTINUE_TRAINING = False
INITIAL_EPOCH = 16 if CONTINUE_TRAINING else 0
TRAINING_MODEL_FILENAME_TO_CONTINUE = BASE_DIR + 'ep_3_valacc_0.96525.h5'

TRAINING_INPUT_FILENAME = '../data/pos/train.conll'
DEV_INPUT_FILENAME = '../data/pos/dev.conll'


def create_vocab_util_from_training_set(tr_input_filename: str) -> VocabUtil:
    """To keep things simple, we use the entire training set without filtering."""
    tr_loader = LabeledDataLoader(tr_input_filename)
    tr_tweets = tr_loader.parse_tokens_and_labels(tr_loader.load_lines())
    tr_unique_tokens = set([item[0] for tweet in tr_tweets for item in tweet])
    sorted_tr_tokens = sorted(tr_unique_tokens)
    return VocabUtil(sorted_tr_tokens)


def prep_validation_set(input_filename: str, nn_input_preparer: NNInputPreparer, vu: VocabUtil) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = LabeledDataLoader(input_filename)
    tweets = loader.parse_tokens_and_labels(loader.load_lines())
    tweets = nn_input_preparer.filter_out_long_sequences(tweets)
    print(f'processing all not-too-long {len(tweets)} tweets from {input_filename}')
    irregular_inputs = [[vu.nn_input_token_to_int[item[0]]
                         if item[0] in vu.nn_input_token_to_int
                         else vu.nn_input_token_to_int['<OOV>']
                         for item in tweet]
                        for tweet in tweets]
    irregular_targets = [[vu.nn_pos_to_int[item[1]] for item in tweet] for tweet in tweets]
    rectangular_inputs = nn_input_preparer.rectangularize_inputs(irregular_inputs)
    rectangular_targets = nn_input_preparer.rectangularize_targets(irregular_targets)
    targets_one_hot_encoded = nn_input_preparer.rectangular_targets_to_one_hot(rectangular_targets)
    return rectangular_inputs, rectangular_targets, targets_one_hot_encoded


def main_training():
    print(f'Using TensorFlow version {tf.__version__}')

    vu = create_vocab_util_from_training_set(TRAINING_INPUT_FILENAME)
    nn_input_preparer = NNInputPreparer(vu, max_seq_len=MAX_SEQ_LEN)

    tr_loader = LabeledDataLoader(TRAINING_INPUT_FILENAME)
    tr_tweets = tr_loader.parse_tokens_and_labels(tr_loader.load_lines())
    tr_tweets = nn_input_preparer.filter_out_long_sequences(tr_tweets)
    print(f'Training on {len(tr_tweets)} tweets, each no longer than {MAX_SEQ_LEN} tokens')
    tr_irregular_inputs = [[vu.nn_input_token_to_int[item[0]] for item in tweet] for tweet in tr_tweets]
    tr_irregular_targets = [[vu.nn_pos_to_int[item[1]] for item in tweet] for tweet in tr_tweets]
    tr_rectangular_inputs = nn_input_preparer.rectangularize_inputs(tr_irregular_inputs)
    tr_rectangular_targets = nn_input_preparer.rectangularize_targets(tr_irregular_targets)
    tr_targets_one_hot_encoded = nn_input_preparer.rectangular_targets_to_one_hot(tr_rectangular_targets)

    if CONTINUE_TRAINING:
        print('Continuing training from', TRAINING_MODEL_FILENAME_TO_CONTINUE)
        model = load_model(TRAINING_MODEL_FILENAME_TO_CONTINUE)
        model.summary()
    else:
        print("Commencing new training run")
        model_creator = LstmModelCreator(vu, embedding_dim=EMBEDDING_DIM, lstm_dim=LSTM_DIM, mask_zero=MASK_ZERO)
        model = model_creator.create_bi_lstm_model()

    cp_filepath = BASE_DIR + 'ep_{epoch}_valacc_{val_accuracy:.5f}.h5'

    checkpoint = ModelCheckpoint(cp_filepath, monitor='val_accuracy', verbose=1,
                                 save_best_only=False)

    rectangular_inputs, _, targets_one_hot_encoded = \
        prep_validation_set(DEV_INPUT_FILENAME, nn_input_preparer, vu)

    model.fit(x=tr_rectangular_inputs, y=tr_targets_one_hot_encoded, batch_size=32,
              initial_epoch=INITIAL_EPOCH,
              epochs=MAX_EPOCHS,
              validation_data=(rectangular_inputs, targets_one_hot_encoded),
              callbacks=[checkpoint])


if __name__ == '__main__':
    main_training()
