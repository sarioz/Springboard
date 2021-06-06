from frozendict import frozendict


ALPHABET_TUPLE = tuple("abcdefghijklmnopqrstuvwxyzáéíóúüñ")
ALPHABET_SET = frozenset(ALPHABET_TUPLE)

DISACCENT_DICT = frozendict({'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u', 'ü': 'u', 'ñ': 'n'})

VOWELS_SET = frozenset("aeiouáéíóúü")

DIGITS_TUPLE = tuple("0123456789")

PUNCTUATION_TUPLE = tuple(" ¿?¡!.,;#:<>()'“”\"")

RAW_VOCAB_TUPLE = ALPHABET_TUPLE + DIGITS_TUPLE + PUNCTUATION_TUPLE
RAW_VOCAB_STR = ''.join(RAW_VOCAB_TUPLE)

# These symbols may be used by the Neural Network but will not appear in raw inputs or outputs.
SPECIAL_NN_SYMBOLS = ('<GO>', '<EOT>', '<PAD>')

NN_VOCAB_TUPLE = RAW_VOCAB_TUPLE + SPECIAL_NN_SYMBOLS
LEN_NN_VOCAB = len(NN_VOCAB_TUPLE)
NN_VOCAB_TO_INT = frozendict({v: i for i, v in enumerate(NN_VOCAB_TUPLE)})
# We don't need an int-to-vocab dict: just index into NN_VOCAB_TUPLE.