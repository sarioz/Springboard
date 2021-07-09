from typing import List


class TweetSelector:
    def __init__(self, min_length: int, max_length: int):
        self.min_length = min_length
        self.max_length = max_length

    def select(self, tweet: List[str]) -> bool:
        return self.min_length <= len(tweet) <= self.max_length
