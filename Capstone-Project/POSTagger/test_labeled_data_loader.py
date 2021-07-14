import unittest

from labeled_data_loader import LabeledDataLoader


class LabeledDataLoaderTest(unittest.TestCase):
    """The Numbers are from https://arxiv.org/abs/2005.04322"""

    def test_loads_expected_number_of_tweets(self):
        # We need to use the dev set as the test set, so we will optimize based on this training set.
        loader = LabeledDataLoader('../data/pos/train.conll')
        train_tweets_with_labeled_tokens = loader.parse_tokens_and_labels(loader.load_lines())
        self.assertEqual(27893, len(train_tweets_with_labeled_tokens))

        # The test data is NOT labeled, so we want to use this dev set as the test set.
        loader = LabeledDataLoader('../data/pos/dev.conll')
        dev_tweets_with_labeled_tokens = loader.parse_tokens_and_labels(loader.load_lines())
        self.assertEqual(4298, len(dev_tweets_with_labeled_tokens))

        # The test data is NOT labeled, so we can't use it with our labeled data loader

    def test_loads_expected_number_of_tokens(self):
        # We need to use the dev set as the test set, so we will optimize based on this training set.
        loader = LabeledDataLoader('../data/pos/train.conll')
        train_tweets_with_labeled_tokens = loader.parse_tokens_and_labels(loader.load_lines())
        self.assertEqual(217068, sum((len(tweet) for tweet in train_tweets_with_labeled_tokens)))

        # The test data is NOT labeled, so we need to use this dev set as the test set.
        loader = LabeledDataLoader('../data/pos/dev.conll')
        dev_tweets_with_labeled_tokens = loader.parse_tokens_and_labels(loader.load_lines())
        self.assertEqual(33345, sum((len(tweet) for tweet in dev_tweets_with_labeled_tokens)))

        # The test data is NOT labeled, so we can't use it with our labeled data loader


if __name__ == '__main__':
    unittest.main()
