import re

from vocab_util import RAW_VOCAB_STR

class TweetCleaner:
    def clean_tweet(self, tweet: str) -> str:
        # take out URLs
        tweet = re.sub("http.*(\s|$)", ' ', tweet)
        # take out mentions
        tweet = re.sub("@[^\s]+", ' ', tweet)
        # take out hashtags
        tweet = re.sub("#[^\s]+", ' ', tweet)
        tweet = tweet.lower()
        # take out all characters outside of those we specify
        tweet = re.sub("[^" + RAW_VOCAB_STR + "]", ' ', tweet)
        # reset spaces
        tweet = re.sub("\s+", ' ', tweet)
        tweet = re.sub("^\s+", '', tweet)
        tweet = re.sub("\s$", '', tweet)

        return tweet