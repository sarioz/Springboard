import numpy as np

from vocab_util import TargetVocabUtil


class NNInputPreparer:
    def __init__(self, tvu: TargetVocabUtil, max_seq_len: int):
        self.tvu = tvu
        self.max_seq_len = max_seq_len

    def filter_out_long_sequences(self, labeled_sequences: list) -> list:
        return [(sequence, label) for (sequence, label) in labeled_sequences
                if len(sequence) <= self.max_seq_len]

    def pad_tweet_batch(self, tweet_batch: list) -> list:
        """Pad tweets with [PAD] so that each tweet globally has the same length"""
        return [tweet + [0] * (self.max_seq_len - len(tweet)) for tweet in tweet_batch]

    def rectangularize_inputs(self, tweets_batch_ints: list) -> np.ndarray:
        return np.array(self.pad_tweet_batch(tweets_batch_ints))

    def rectangular_targets_to_one_hot(self, rectangular_targets: list) -> np.ndarray:
        encoded_data = np.zeros(
            (len(rectangular_targets), self.tvu.get_output_vocab_size()), dtype="float32"
        )

        for i, rectangular_target in enumerate(rectangular_targets):
            encoded_data[i][rectangular_target] = 1.0

        return encoded_data
