from typing import List


class FeatureExtractor:

    def __init__(self, lexicon: dict):
        self.lexicon = lexicon

    def extract_average_polarity(self, known_tokens) -> float:
        return sum([self.lexicon[token] for token in known_tokens]) / len(known_tokens) \
            if known_tokens else 0.0

    def extract_min_polarity(self, known_tokens) -> float:
        return min([self.lexicon[token] for token in known_tokens]) \
            if known_tokens else 0.0

    def extract_max_polarity(self, known_tokens) -> float:
        return max([self.lexicon[token] for token in known_tokens]) \
            if known_tokens else 0.0

    def compute_feature_vector(self, known_tokens) -> List[float]:
        return [self.extract_min_polarity(known_tokens),
                self.extract_average_polarity(known_tokens),
                self.extract_max_polarity(known_tokens)]
