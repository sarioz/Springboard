from frozendict import frozendict


# $ cut -f 3 train.conll | grep -v '#' | sort | uniq
POS_TUPLE = ('ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'DET', 'INTJ', 'NOUN', 'NUM',
             'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'UNK', 'VERB', 'X')

POS_TO_INT = frozendict({value: index for index, value in enumerate(POS_TUPLE)})
