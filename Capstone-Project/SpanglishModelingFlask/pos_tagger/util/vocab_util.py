from frozendict import frozendict

# $ cut -f 3 train.conll | grep -v '#' | sort | uniq
RAW_POS_TUPLE = ('ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'DET', 'INTJ', 'NOUN', 'NUM',
                 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'UNK', 'VERB', 'X')

NN_POS_TUPLE = ('<PAD>',) + RAW_POS_TUPLE

NN_POS_TO_INT = frozendict({v: i for i, v in enumerate(NN_POS_TUPLE)})


class VocabUtil:

    def __init__(self, sorted_input_tokens):
        # Made sure that '<PAD>' and '<OOV>' aren't in the training corpus
        self.nn_input_tokens = ('<PAD>',) + tuple(sorted_input_tokens) + ('<OOV>',)
        self.nn_input_token_to_int = frozendict({v: i for i, v in enumerate(self.nn_input_tokens)})
        self.nn_pos_tuple = NN_POS_TUPLE
        self.nn_pos_to_int = NN_POS_TO_INT

    def get_input_vocab_size(self) -> int:
        return len(self.nn_input_tokens)

    def get_output_vocab_size(self) -> int:
        return len(self.nn_pos_tuple)
