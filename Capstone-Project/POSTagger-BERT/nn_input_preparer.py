import numpy as np

from vocab_util import TargetVocabUtil


class NNInputPreparer:
    def __init__(self, tvu: TargetVocabUtil, max_seq_len: int):
        self.tvu = tvu
        self.max_seq_len = max_seq_len

    def filter_out_long_sequences(self, sequences: list) -> list:
        return [sequence for sequence in sequences if len(sequence) <= self.max_seq_len]

    def pad_tweet_batch(self, tweet_batch: list) -> list:
        """Pad tweets with [PAD] so that each tweet of a batch has the same length"""
        return [tweet + [0] * (self.max_seq_len - len(tweet))
                for tweet in tweet_batch]

    def pad_label_seq_batch(self, label_seqs_batch: list) -> list:
        """Pad label sequences with [PAD] so that each label sequence of a batch has the same length"""
        return [label_seq + [0] * (self.max_seq_len - len(label_seq))
                for label_seq in label_seqs_batch]

    def rectangularize_inputs(self, tweets_batch_ints: list) -> np.ndarray:
        padded_tweets_batch = np.array(self.pad_tweet_batch(tweets_batch_ints))

        return padded_tweets_batch

    def compute_mask(self, tweets_batch_ints: list) -> np.ndarray:
        mask = np.zeros(
            (len(tweets_batch_ints), self.max_seq_len), dtype="bool"
        )
        for i, tweet in enumerate(tweets_batch_ints):
            for j in range(len(tweet)):
                mask[i][j] = True

        return mask

    def rectangularize_targets(self, label_seqs_batch_ints: list) -> np.ndarray:
        max_length_in_batch = max([len(tweet_label) for tweet_label in label_seqs_batch_ints])
        padded_labels_batch = np.array(self.pad_label_seq_batch(label_seqs_batch_ints))

        return padded_labels_batch

    def rectangular_targets_to_one_hot(self, rectangular_targets: np.ndarray) -> np.ndarray:
        encoded_data = np.zeros(
            (len(rectangular_targets), len(rectangular_targets[0]), self.tvu.get_output_vocab_size()), dtype="float32"
        )

        for i, padded_target in enumerate(rectangular_targets):
            for t, int_value in enumerate(padded_target):
                encoded_data[i, t, int_value] = 1.0

        return encoded_data
