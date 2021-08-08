from frozendict import frozendict

RAW_SENTIMENT_LABELS = ('negative', 'neutral', 'positive')
NN_RSL_TO_INT = frozendict({v: i for i, v in enumerate(RAW_SENTIMENT_LABELS)})


class TargetVocabUtil:

    def __init__(self):
        self.raw_sentiment_labels = RAW_SENTIMENT_LABELS
        self.nn_rsl_to_int = NN_RSL_TO_INT

    def get_output_vocab_size(self) -> int:
        return len(self.nn_rsl_to_int)
