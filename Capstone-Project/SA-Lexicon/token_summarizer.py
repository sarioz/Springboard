from typing import List


class TokenSummarizer:

    def __init__(self, lexicon: dict):
        self.lexicon = lexicon

    def get_known_tokens(self, tweet) -> List[str]:
        """Returns singleton sequence followed by pair sequence; the same token may occur multiple times."""
        singletons = [token for token in tweet if token in self.lexicon]
        pairs = []
        i = 0
        while i < len(tweet) - 1:
            candidate_pair = tweet[i] + '_' + tweet[i + 1]
            if candidate_pair in self.lexicon:
                pairs.append(candidate_pair)
                if tweet[i] in singletons:
                    singletons.remove(tweet[i])
                if tweet[i + 1] in singletons:
                    singletons.remove(tweet[i + 1])
                i += 2
            else:
                i += 1

        # For now we assign the same weight to singletons and pairs, but we could double up on pairs if we wish below.
        return singletons + pairs
