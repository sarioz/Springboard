import numpy as np
from typing import List

from vocab_util import VocabUtil


class NNInputPreparer:
    def __init__(self, vu: VocabUtil):
        self.vu = vu

    def rectangular_targets_to_one_hot(self, rectangular_targets: List[str]) -> np.ndarray:
        encoded_data = np.zeros(
            (len(rectangular_targets), self.vu.get_output_vocab_size()), dtype="float32"
        )

        for i, rectangular_target in enumerate(rectangular_targets):
            encoded_data[i][self.vu.nn_rsl_to_int[rectangular_target]] = 1.0

        return encoded_data
