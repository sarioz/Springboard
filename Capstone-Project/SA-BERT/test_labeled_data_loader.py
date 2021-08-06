import unittest

from labeled_data_loader import LabeledDataLoader


class LabeledDataLoaderTest(unittest.TestCase):
    """The Numbers are from https://arxiv.org/abs/2005.04322"""

    def test_loads_expected_number_of_tweets(self):
        # We need to use the dev set as the test set, so we will optimize based on this training set.
        loader = LabeledDataLoader('../data/sa/train.conll')
        train_tweets_with_labels = loader.parse_tokens_and_labels(loader.load_lines())
        self.assertEqual(12194, len(train_tweets_with_labels))

        # The test data is NOT labeled, so we want to use this dev set as the test set.
        loader = LabeledDataLoader('../data/sa/dev.conll')
        dev_tweets_with_labels = loader.parse_tokens_and_labels(loader.load_lines())
        self.assertEqual(1859, len(dev_tweets_with_labels))

        # The test data is NOT labeled, so we can't use it with our labeled data loader

    def test_loads_expected_number_of_tokens(self):
        # We need to use the dev set as the test set, so we will optimize based on this training set.
        loader = LabeledDataLoader('../data/sa/train.conll')
        train_tweets_with_labels = loader.parse_tokens_and_labels(loader.load_lines())
        self.assertEqual(186602, sum((len(labeled_tweet[0]) for labeled_tweet in train_tweets_with_labels)))

        # The test data is NOT labeled, so we need to use this dev set as the test set.
        loader = LabeledDataLoader('../data/sa/dev.conll')
        dev_tweets_with_labels = loader.parse_tokens_and_labels(loader.load_lines())
        self.assertEqual(28202, sum((len(labeled_tweet[0]) for labeled_tweet in dev_tweets_with_labels)))

    def test_has_one_of_three_labels(self):
        # We need to use the dev set as the test set, so we will optimize based on this training set.
        loader = LabeledDataLoader('../data/sa/train.conll')
        train_tweets_with_labels = loader.parse_tokens_and_labels(loader.load_lines())
        train_labels_set = set([labeled_tweet[1] for labeled_tweet in train_tweets_with_labels])
        self.assertEqual({'positive', 'negative', 'neutral'}, train_labels_set)

        loader = LabeledDataLoader('../data/sa/dev.conll')
        dev_tweets_with_labels = loader.parse_tokens_and_labels(loader.load_lines())
        dev_labels_set = set([labeled_tweet[1] for labeled_tweet in dev_tweets_with_labels])
        self.assertEqual({'positive', 'negative', 'neutral'}, dev_labels_set)


if __name__ == '__main__':
    unittest.main()
