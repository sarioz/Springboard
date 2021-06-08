import unittest

from tweet_cleaner import TweetCleaner


class TweetCleanerTest(unittest.TestCase):

    def test_removes_mentions(self):
        input_example = "@Lyanne_DLC muajajajaja"

        self.assertEqual(
            "muajajajaja",
            TweetCleaner().clean_tweet(input_example),
        )


    def test_removes_emojis(self):
        input_example = "muajajajaja 😈😈 ya te descubri ante todo twitter lmfao"

        self.assertEqual(
            "muajajajaja ya te descubri ante todo twitter lmfao",
            TweetCleaner().clean_tweet(input_example),
        )


    def test_lowercases(self):
        input_example = "yo le voy a dedicar la de amor brutal de Traileros Del Norte"

        self.assertEqual(
            "yo le voy a dedicar la de amor brutal de traileros del norte",
            TweetCleaner().clean_tweet(input_example)
        )

    def test_preserves_double_quotes(self):
        input_example = ', " SI NO ME QUIERES TE MATO !!! " Jajaja'

        self.assertEqual(
            ', " si no me quieres te mato !!! " jajaja',
            TweetCleaner().clean_tweet(input_example)
        )


    def test_preserves_digits(self):
        input_example = 'Concidering the fact that I have too be awake in 30 mins , this sucks'

        self.assertEqual(
            'concidering the fact that i have too be awake in 30 mins , this sucks',
            TweetCleaner().clean_tweet(input_example)
        )


if __name__ == '__main__':
    unittest.main()
