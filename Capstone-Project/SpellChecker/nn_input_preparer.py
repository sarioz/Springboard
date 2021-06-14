import numpy as np
from typing import List

from noiser import DisjointNoiser
from vocab_util import LEN_NN_VOCAB, NN_VOCAB_TO_INT, NN_VOCAB_TUPLE


class NNInputPreparer:

    def pad_tweet_batch(self, tweet_batch: list, max_tweet_length_in_batch: int) -> list:
        """Pad tweets with <PAD> so that each sentence of a batch has the same length"""
        return [tweet + [NN_VOCAB_TO_INT['<PAD>']] * (max_tweet_length_in_batch - len(tweet))
                for tweet in tweet_batch]

    def one_hot_encode(self, padded_batch: np.ndarray) -> np.ndarray:
        encoded_data = np.zeros(
            (len(padded_batch), len(padded_batch[0]), LEN_NN_VOCAB), dtype="float32"
        )

        for i, padded_text in enumerate(padded_batch):
            for t, int_value in enumerate(padded_text):
                encoded_data[i, t, int_value] = 1.0

        return encoded_data

    def decode_tweet(self, encoded_tweet: np.ndarray) -> str:
        return ''.join([NN_VOCAB_TUPLE[np.argmax(encoded_char)] for encoded_char in encoded_tweet])

    def decode_batch(self, encoded_tweets: np.ndarray) -> List[str]:
        return [self.decode_tweet(t) for t in encoded_tweets]

    def get_batches(self, tweets: list, noiser: DisjointNoiser, batch_size: int):
        for batch_i in range(0, len(tweets) // batch_size):
            start_i = batch_i * batch_size

            tweets_batch = tweets[start_i:start_i + batch_size]

            tweets_batch_noised = [noiser.add_noise(tweet) for tweet in tweets_batch]

            tweets_batch_ints = [[NN_VOCAB_TO_INT[c] for c in tweet] for tweet in tweets_batch]
            tweets_batch_noised_ints = [[NN_VOCAB_TO_INT[c] for c in tweet]
                                        for tweet in tweets_batch_noised]

            tweets_batch_eot = [tweet + [NN_VOCAB_TO_INT['<EOT>']] for tweet in tweets_batch_ints]
            tweets_batch_delayed_eot = [[NN_VOCAB_TO_INT['<GO>']] + tweet
                                        + [NN_VOCAB_TO_INT['<EOT>']]
                                        for tweet in tweets_batch_ints]
            tweets_batch_noised_eot = [tweet + [NN_VOCAB_TO_INT['<EOT>']]
                                       for tweet in tweets_batch_noised_ints]

            padded_tweets_batch = np.array(self.pad_tweet_batch(
                tweets_batch_eot,
                1 + max([len(tweet) for tweet in tweets_batch_eot])))
            padded_tweets_delayed_batch = np.array(self.pad_tweet_batch(
                tweets_batch_delayed_eot,
                max([len(tweet) for tweet in tweets_batch_delayed_eot])))
            padded_tweets_noised_batch = np.array(self.pad_tweet_batch(
                tweets_batch_noised_eot,
                max([len(tweet) for tweet in tweets_batch_noised_eot])))

            padded_tweets_noised_encoded_batch = self.one_hot_encode(padded_tweets_noised_batch)
            padded_tweets_encoded_batch = self.one_hot_encode(padded_tweets_batch)
            padded_tweets_delayed_encoded_batch = self.one_hot_encode(padded_tweets_delayed_batch)

            yield (padded_tweets_noised_encoded_batch, padded_tweets_encoded_batch,
                   padded_tweets_delayed_encoded_batch)
