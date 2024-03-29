import numpy as np

from .vocab_util import VocabUtil


class NNInputPreparer:
    def __init__(self, vu: VocabUtil, max_seq_len: int):
        self.vu = vu
        self.max_seq_len = max_seq_len

    def filter_out_long_sequences(self, sequences: list) -> list:
        return [sequence for sequence in sequences if len(sequence) <= self.max_seq_len]

    def pad_tweet_batch(self, tweet_batch: list) -> list:
        """Pad tweets with <PAD> so that each tweet has length max_seq_len"""
        return [tweet + [self.vu.nn_input_token_to_int['<PAD>']] * (self.max_seq_len - len(tweet))
                for tweet in tweet_batch]

    def pad_label_seq_batch(self, label_seqs_batch: list) -> list:
        """Pad label sequences with <PAD> so that each label sequence has length max_seq_len"""
        return [label_seq + [self.vu.nn_pos_to_int['<PAD>']] * (self.max_seq_len - len(label_seq))
                for label_seq in label_seqs_batch]

    def rectangularize_inputs(self, tweets_batch_ints: list) -> np.ndarray:
        return np.array(self.pad_tweet_batch(tweets_batch_ints))

    def rectangularize_targets(self, label_seqs_batch_ints: list) -> np.ndarray:
        return np.array(self.pad_label_seq_batch(label_seqs_batch_ints))

    def rectangular_inputs_to_one_hot(self, rectangular_inputs: np.ndarray) -> np.ndarray:
        encoded_data = np.zeros(
            (len(rectangular_inputs), len(rectangular_inputs[0]), self.vu.get_input_vocab_size()), dtype="float32"
        )

        for i, padded_input in enumerate(rectangular_inputs):
            for t, int_value in enumerate(padded_input):
                encoded_data[i, t, int_value] = 1.0

        return encoded_data

    def rectangular_targets_to_one_hot(self, rectangular_targets: np.ndarray) -> np.ndarray:
        encoded_data = np.zeros(
            (len(rectangular_targets), len(rectangular_targets[0]), self.vu.get_output_vocab_size()), dtype="float32"
        )

        for i, padded_target in enumerate(rectangular_targets):
            for t, int_value in enumerate(padded_target):
                encoded_data[i, t, int_value] = 1.0

        return encoded_data
