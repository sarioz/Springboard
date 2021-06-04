import re

class TweetCleaner:
    def clean_tweet(self, tweet: str) -> str:
        # take out URLs
        tweet = re.sub("http.*(\s|$)", ' ', tweet)
        # take out mentions
        tweet = re.sub("@[^\s]+", ' ', tweet)
        # take out hashtags
        tweet = re.sub("#[^\s]+", ' ', tweet)
        # take out all characters outside of those we enumerate
        tweet = re.sub("[^\da-zA-ZáéíóúüñÁÉÍÑÓÚÜ¿?¡!.,;#:<>()'“”\"\s]", ' ', tweet)
        # lowercase everything
        tweet = tweet.lower()
        # reset spaces
        tweet = re.sub("\s+", ' ', tweet)
        tweet = re.sub("^\s+", '', tweet)
        tweet = re.sub("\s$", '', tweet)

        return tweet