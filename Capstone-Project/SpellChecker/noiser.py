"""In this module we assume that the input has no uppercase characters."""

import random
from typing import List

DISACCENT_MAP = {'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u', 'ü': 'u', 'ñ': 'n'}


def drop_accents(tweet: [str], drop_probability: float = 1.0) -> [str]:
    output = []
    for c in tweet:
        if c in DISACCENT_MAP and random.random() < drop_probability:
            output.append(DISACCENT_MAP[c])
        else:
            output.append(c)
    return output


VOWELS_SET = set("aeiouáéíóúü")


def drop_vowels(tweet: List[str], drop_probability: float = 1.0) -> List[str]:
    output = []
    for c in tweet:
        if c in VOWELS_SET and random.random() < drop_probability:
            continue
        output.append(c)
    return output


def repeat_vowels(tweet: List[str], repeat_probability=0.05, max_repeat=6) -> List[str]:
    output = []
    for c in tweet:
        if c in VOWELS_SET and random.random() < repeat_probability:
            for _ in range(random.randint(1, max_repeat)):
                output.append(c)
        output.append(c)
    return output


CREATIVE_SUB_MAP = {'c': 's', 's': 'z', 'b': 'v', 'v': 'b'}


def substitute_creatively(tweet: List[str], substitution_probability=0.1) -> List[str]:
    output = []
    for c in tweet:
        if c in CREATIVE_SUB_MAP and random.random() < substitution_probability:
            output.append(CREATIVE_SUB_MAP[c])
        else:
            output.append(c)
    return output


def simulate_creative_misspellings(tweet: List[str]) -> List[str]:
    if random.random() < 0.7:
        return repeat_vowels(tweet)
    else:
        return substitute_creatively(tweet)


def simulate_intentional_shortenings(tweet: List[str]) -> List[str]:
    input_tokens = ''.join(tweet).split(' ')
    output_tokens = []
    for token in input_tokens:
        if random.random() < 0.3:
            output_tokens.append(''.join(drop_vowels(list(token), 0.5)))
        else:
            output_tokens.append(token)
    return list(' '.join(output_tokens))


ALPHABET_L = tuple("abcdefghijklmnopqrstuvwxyzáéíóúüñ")
ALPHABET_S = frozenset(ALPHABET_L)


def simulate_bona_fide_spelling_mistakes(tweet: List[str], modify_rate=0.1) -> List[str]:
    # equally add, omit, or substitute each character for a total probability of modify_rate
    add_cmf_val = modify_rate / 3
    omit_cmf_val = 2 * modify_rate / 3
    substitute_cmf_val = modify_rate

    output = []
    for c in tweet:
        r = random.random()
        if r < add_cmf_val:
            output.append(c)
            output.append(random.choice(ALPHABET_L))
        elif r < omit_cmf_val:
            continue
        elif c in ALPHABET_S and r < substitute_cmf_val:
            # we don't apply substitutions to non-alpha
            output.append(random.choice(ALPHABET_L))
        else:
            output.append(c)
    return output


def identity(tweet: List[str]) -> List[str]:
    return tweet


DEFAULT_WEIGHT_MAP = ((14, simulate_creative_misspellings),
                      (13, drop_accents),
                      (10, simulate_intentional_shortenings),
                      (8, simulate_bona_fide_spelling_mistakes),
                      (10, identity))


class DisjointNoiser:
    def __init__(self, weight_map=DEFAULT_WEIGHT_MAP):
        self.weight_map = [[weight, func] for (weight, func) in weight_map]
        total = sum([self.weight_map[i][0] for i in range(len(self.weight_map))])
        # we normalize the pmf
        for i in range(len(self.weight_map)):
            self.weight_map[i][0] /= total
        # we make the pmf into a cmf
        for i in range(1, len(self.weight_map) - 1):
            self.weight_map[i][0] += self.weight_map[i - 1][0]
        self.weight_map[-1][0] = 1.0  # so as not to worry about rounding errors

    def add_noise(self, tweet: List[str]) -> List[str]:
        """Applies at most 1 kind of noising to the tweet according to the weights in the weight map.
        Each 'noising' could alter the tweet in multiple places or not at all.
        """
        tweet = list(tweet)
        my_random_number = random.random()
        for max_prob, noise_function in self.weight_map:
            if my_random_number < max_prob:
                # print('calling', noise_function)
                return noise_function(tweet)
