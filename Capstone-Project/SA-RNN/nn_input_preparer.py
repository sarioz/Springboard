import numpy as np
from typing import List
from collections import Counter

from vocab_util import VocabUtil


class NNInputPreparer:
    def __init__(self, vu: VocabUtil, max_seq_len: int):
        self.vu = vu
        self.max_seq_len = max_seq_len

    def filter_out_long_tweets(self, labeled_tweets: List[tuple]) -> list:
        return [labeled_tweet for labeled_tweet in labeled_tweets if len(labeled_tweet[0]) <= self.max_seq_len]

    def crude_upsample(self, labeled_tweets: List[tuple]) -> List[tuple]:
        result = []
        multipliers = {'negative': 3, 'neutral': 2, 'positive': 1}
        for tweet, label in labeled_tweets:
            for _ in range(multipliers[label]):
                result.append((tweet, label))
        c = Counter([upsampled_labeled_tweet[1] for upsampled_labeled_tweet in result])
        print(c)
        return result

    def pad_tweet_batch(self, tweet_batch: list) -> list:
        """Pad tweets with <PAD> so that each tweet globally has the same length"""
        return [tweet + [self.vu.nn_input_token_to_int['<PAD>']] * (self.max_seq_len - len(tweet))
                for tweet in tweet_batch]

    def rectangularize_inputs(self, tweets_batch_ints: list) -> np.ndarray:
        return np.array(self.pad_tweet_batch(tweets_batch_ints))

    def rectangular_inputs_to_one_hot(self, rectangular_inputs: np.ndarray) -> np.ndarray:
        encoded_data = np.zeros(
            (len(rectangular_inputs), len(rectangular_inputs[0]), self.vu.get_input_vocab_size()), dtype="float32"
        )

        for i, padded_input in enumerate(rectangular_inputs):
            for t, int_value in enumerate(padded_input):
                encoded_data[i, t, int_value] = 1.0

        return encoded_data

    def rectangular_targets_to_one_hot(self, rectangular_targets: list) -> np.ndarray:
        encoded_data = np.zeros(
            (len(rectangular_targets), self.vu.get_output_vocab_size()), dtype="float32"
        )

        for i, rectangular_target in enumerate(rectangular_targets):
            encoded_data[i][self.vu.nn_rsl_to_int[rectangular_target]] = 1.0

        return encoded_data
