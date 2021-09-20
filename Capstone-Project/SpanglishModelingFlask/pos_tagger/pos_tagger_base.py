import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from typing import List

from pos_tagger.util.labeled_data_loader import LabeledDataLoader
from pos_tagger.util.nn_input_preparer import NNInputPreparer
from pos_tagger.util.vocab_util import VocabUtil

MAX_SEQ_LEN = 128
EXPERIMENT_NAME = '08_mzt_bi_LSTM_64_64'
TRAINING_MODEL_FILENAME = f'pos_tagger/models/{EXPERIMENT_NAME}/ep_71_valacc_0.95505.h5'
TRAINING_INPUT_FILENAME = 'data/pos/train.conll'


def create_vocab_util_from_training_set(tr_input_filename: str) -> VocabUtil:
    """To keep things simple, we use the entire training set without filtering."""
    tr_loader = LabeledDataLoader(tr_input_filename)
    tr_tweets = tr_loader.parse_tokens_and_labels(tr_loader.load_lines())
    tr_unique_tokens = set([item[0] for tweet in tr_tweets for item in tweet])
    sorted_tr_tokens = sorted(tr_unique_tokens)
    return VocabUtil(sorted_tr_tokens)


def prep_single_tweet(tweet: str, nn_input_preparer: NNInputPreparer, vu: VocabUtil):
    tokenized_tweet = tweet.strip().split()
    if len(tokenized_tweet) > MAX_SEQ_LEN:
        return None, None
    irregular_input = [[vu.nn_input_token_to_int[item]
                       if item in vu.nn_input_token_to_int
                       else vu.nn_input_token_to_int['<OOV>']
                       for item in tokenized_tweet]]
    rectangular_inputs = nn_input_preparer.rectangularize_inputs(irregular_input)
    return tokenized_tweet, rectangular_inputs


class PosTagger:
    def __init__(self):
        print('Initializing POS Tagger')
        print(f'Using TensorFlow version {tf.__version__}')
        print(f'Loading model {TRAINING_MODEL_FILENAME}')
        self.trained_model = load_model(TRAINING_MODEL_FILENAME, compile=False)
        self.vu = create_vocab_util_from_training_set(TRAINING_INPUT_FILENAME)
        self.nn_input_preparer = NNInputPreparer(self.vu, max_seq_len=MAX_SEQ_LEN)

    def make_prediction(self, tweet: str) -> (List[str], List[str]):
        tokenized_tweet, rectangular_inputs = prep_single_tweet(tweet, self.nn_input_preparer, self.vu)
        if not tokenized_tweet:
            return ["Error"], [f"Input tweet can be at most {MAX_SEQ_LEN} tokens long."]

        rectangular_input_2d = np.array(rectangular_inputs)
        rectangular_input_2d.shape = (1, MAX_SEQ_LEN)
        predicted_probabilities_sequence = self.trained_model(rectangular_input_2d, training=False)[0]
        predicted_tags = []

        for tweet_token, predicted_probabilities in zip(tokenized_tweet, predicted_probabilities_sequence):
            # we predict by taking the class with the largest probability
            argmax_index = np.argmax(predicted_probabilities)
            predicted_tags.append(self.vu.nn_pos_tuple[argmax_index])

        return tokenized_tweet, predicted_tags
