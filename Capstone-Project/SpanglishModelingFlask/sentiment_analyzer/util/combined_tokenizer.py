from typing import List

from spacy.lang.en import English
from spacy.lang.es import Spanish


class CombinedTokenizer:
    def __init__(self):
        self.nlp_english = English()
        self.nlp_spanish = Spanish()

    def tokenize(self, tweet: str) -> List[str]:
        tweet = tweet.strip()
        spanish_result = self.nlp_spanish(tweet)
        spanish_tokens = [token.text for token in spanish_result]
        all_english_tokens = []
        for spanish_token in spanish_tokens:
            english_token_result = self.nlp_english(spanish_token)
            all_english_tokens.extend([token.text for token in english_token_result])

        return all_english_tokens
