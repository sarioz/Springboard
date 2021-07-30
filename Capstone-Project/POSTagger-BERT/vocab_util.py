from frozendict import frozendict

# $ cut -f 3 train.conll | grep -v '#' | sort | uniq
RAW_POS_TUPLE = ('ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'DET', 'INTJ', 'NOUN', 'NUM',
                 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'UNK', 'VERB', 'X')

NN_POS_TUPLE = ('[PAD]',) + RAW_POS_TUPLE

NN_POS_TO_INT = frozendict({v: i for i, v in enumerate(NN_POS_TUPLE)})


class TargetVocabUtil:

    def __init__(self):
        self.nn_pos_tuple = NN_POS_TUPLE
        self.nn_pos_to_int = NN_POS_TO_INT

    def get_output_vocab_size(self) -> int:
        return len(self.nn_pos_tuple)
