import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from typing import List

from sentiment_analyzer.util.combined_tokenizer import CombinedTokenizer
from sentiment_analyzer.util.labeled_data_loader import LabeledDataLoader
from sentiment_analyzer.util.nn_input_preparer import NNInputPreparer
from sentiment_analyzer.util.vocab_util import VocabUtil

MAX_SEQ_LEN = 128
EXPERIMENT_NAME = '14_bi_LSTM_64_64'
TRAINING_MODEL_FILENAME = f'sentiment_analyzer/models/{EXPERIMENT_NAME}/ep_6_valacc_0.56912.h5'
TRAINING_INPUT_FILENAME = 'data/sa/train.conll'


def create_vocab_util_from_training_set(tr_input_filename: str) -> VocabUtil:
    """To keep things simple, we use the entire training set without filtering."""
    tr_loader = LabeledDataLoader(tr_input_filename)
    tr_labeled_tweets = tr_loader.parse_tokens_and_labels(tr_loader.load_lines())
    tr_unique_tokens = set([item for labeled_tweet in tr_labeled_tweets for item in labeled_tweet[0]])
    tr_sorted_tokens = sorted(tr_unique_tokens)
    print(f"Creating Sentiment Analyze VocabUtil from {len(tr_sorted_tokens)} unique tokens.")
    return VocabUtil(tr_sorted_tokens)


def prep_single_tweet(tweet: str, tokenizer: CombinedTokenizer, nn_input_preparer: NNInputPreparer, vu: VocabUtil):
    tokenized_tweet = tokenizer.tokenize(tweet)
    if len(tokenized_tweet) > MAX_SEQ_LEN:
        raise ValueError(f"The Sentiment Analyzer has tokenized the input to {len(tokenized_tweet)} tokens. "
                         f"Maximum allowed is {MAX_SEQ_LEN}.")
    irregular_input = [[vu.nn_input_token_to_int[item]
                       if item in vu.nn_input_token_to_int
                       else vu.nn_input_token_to_int['<OOV>']
                       for item in tokenized_tweet]]
    print(irregular_input)
    rectangular_inputs = nn_input_preparer.rectangularize_inputs(irregular_input)
    print(rectangular_inputs)
    return tokenized_tweet, rectangular_inputs


class SentimentAnalyzer:
    def __init__(self):
        print('Initializing Sentiment Analyzer', flush=True)
        print(f'Using TensorFlow version {tf.__version__}', flush=True)
        print(f'Loading model {TRAINING_MODEL_FILENAME}', flush=True)
        self.trained_model = load_model(TRAINING_MODEL_FILENAME, compile=False)
        self.tokenizer = CombinedTokenizer()
        self.vu = create_vocab_util_from_training_set(TRAINING_INPUT_FILENAME)
        self.nn_input_preparer = NNInputPreparer(self.vu, max_seq_len=MAX_SEQ_LEN)

    def make_prediction(self, tweet: str) -> (List[str], str):
        tokenized_tweet, rectangular_inputs = prep_single_tweet(tweet, self.tokenizer, self.nn_input_preparer, self.vu)

        rectangular_input_2d = np.array(rectangular_inputs)
        rectangular_input_2d.shape = (1, MAX_SEQ_LEN)
        predicted_probabilities = self.trained_model(rectangular_input_2d, training=False)[0]
        print(predicted_probabilities, flush=True)
        argmax_index = np.argmax(predicted_probabilities)
        predicted_sentiment = self.vu.raw_sentiment_labels[argmax_index]

        return tokenized_tweet, predicted_sentiment
