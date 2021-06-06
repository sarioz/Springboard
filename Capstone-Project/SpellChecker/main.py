from loader import Loader
from tweet_cleaner import TweetCleaner
from noiser import DisjointNoiser

import random

from vocab_util import NN_VOCAB_TO_INT

TRAINING_INPUT_PATH = '../data/lid_train_lines.txt'


def main():
    random.seed(42)

    raw_tweets = Loader(TRAINING_INPUT_PATH).load()
    num_training_tweets = len(raw_tweets)
    cleaner = TweetCleaner()
    clean_tweets = [cleaner.clean_tweet(t) for t in raw_tweets]


    noiser = DisjointNoiser()
    noisy_tweets_as_lists = [noiser.add_noise(list(t)) for t in clean_tweets]
    noisy_tweets_readable = [''.join(t) for t in noisy_tweets_as_lists]
    clean_tweets_as_lists = [list(t) for t in clean_tweets]

    example = clean_tweets_as_lists[0]
    example_ints = [NN_VOCAB_TO_INT[c] for c in example]

    print(example_ints)


if __name__ == '__main__':
    main()
