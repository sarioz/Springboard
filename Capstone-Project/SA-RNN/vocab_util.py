from frozendict import frozendict

RAW_SENTIMENT_LABELS = ('negative', 'neutral', 'positive')
NN_RSL_TO_INT = frozendict({v: i for i, v in enumerate(RAW_SENTIMENT_LABELS)})


class VocabUtil:

    def __init__(self, sorted_input_tokens):
        # Made sure that '<PAD>' and '<OOV>' aren't in the SA corpora
        self.nn_input_tokens = ('<PAD>',) + tuple(sorted_input_tokens) + ('<OOV>',)
        self.nn_input_token_to_int = frozendict({v: i for i, v in enumerate(self.nn_input_tokens)})
        self.raw_sentiment_labels = RAW_SENTIMENT_LABELS
        self.nn_rsl_to_int = NN_RSL_TO_INT

    def get_input_vocab_size(self) -> int:
        return len(self.nn_input_tokens)

    def get_output_vocab_size(self) -> int:
        return len(self.nn_rsl_to_int)
