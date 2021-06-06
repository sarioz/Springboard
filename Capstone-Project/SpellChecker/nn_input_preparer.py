import numpy as np

from noiser import DisjointNoiser
from vocab_util import LEN_NN_VOCAB, NN_VOCAB_TO_INT


class NNInputPreparer:

    def pad_tweet_batch(self, tweet_batch, max_tweet_length_in_batch):
        """Pad tweets with <PAD> so that each sentence of a batch has the same length"""
        return [tweet + [NN_VOCAB_TO_INT['<PAD>']] * (max_tweet_length_in_batch - len(tweet))
                for tweet in tweet_batch]

    def one_hot_encode(self, padded_batch) -> np.ndarray:
        encoded_data = np.zeros(
            (len(padded_batch), len(padded_batch[0]), LEN_NN_VOCAB), dtype="float32"
        )

        for i, padded_text in enumerate(padded_batch):
            for t, int_value in enumerate(padded_text):
                encoded_data[i, t, int_value] = 1.0

        return encoded_data

    def get_batches(self, tweets: list, noiser: DisjointNoiser, batch_size):
        for batch_i in range(0, len(tweets) // batch_size):
            start_i = batch_i * batch_size
            tweets_batch = tweets[start_i:start_i + batch_size]
            tweets_batch_noised = [noiser.add_noise(tweet) for tweet in tweets_batch]

            tweets_batch_ints = [[NN_VOCAB_TO_INT[v] for v in tweet] for tweet in tweets_batch]
            tweets_batch_noised_ints = [[NN_VOCAB_TO_INT[v] for v in tweet]
                                        for tweet in tweets_batch_noised]

            tweets_batch_eot = [tweet + [NN_VOCAB_TO_INT['<EOT>']] for tweet in tweets_batch_ints]
            tweets_batch_delayed_eot = [[NN_VOCAB_TO_INT['<GO>']] + tweet
                                        + [NN_VOCAB_TO_INT['<EOT>']]
                                        for tweet in tweets_batch_ints]
            tweets_batch_noised_eot = [tweet + [NN_VOCAB_TO_INT['<EOT>']]
                                       for tweet in tweets_batch_noised_ints]

            pad_tweets_batch = np.array(self.pad_tweet_batch(
                tweets_batch_eot,
                1 + max([len(tweet) for tweet in tweets_batch_eot])))
            pad_tweets_delayed_batch = np.array(self.pad_tweet_batch(
                tweets_batch_eot, max([len(tweet) for tweet in tweets_batch_delayed_eot])))
            pad_tweets_noised_batch = np.array(self.pad_tweet_batch(
                tweets_batch_noised_eot, max([len(tweet) for tweet in tweets_batch_noised_eot])))

            pad_tweets_enc_batch = self.one_hot_encode(pad_tweets_batch)
            pad_tweets_delayed_enc_batch = self.one_hot_encode(pad_tweets_delayed_batch)
            pad_tweets_noised_enc_batch = self.one_hot_encode(pad_tweets_noised_batch)

            yield pad_tweets_noised_enc_batch, pad_tweets_enc_batch, pad_tweets_delayed_enc_batch
