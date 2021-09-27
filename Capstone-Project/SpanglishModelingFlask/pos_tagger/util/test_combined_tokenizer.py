import unittest

from combined_tokenizer import CombinedTokenizer


class CombinedTokenizerTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CombinedTokenizer()

    def test_tokenize_spanish_text(self):
        self.assertEqual(self.tokenizer.tokenize('¿Quién eres tú? ¡Hola! ¿Dónde estoy?'),
                         ['¿', 'Quién', 'eres', 'tú', '?', '¡', 'Hola', '!', '¿', 'Dónde', 'estoy', '?'])

    def test_tokenize_english_text(self):
        self.assertEqual(self.tokenizer.tokenize("She doesn't want to run."),
                         ['She', 'does', "n't", 'want', 'to', 'run', '.'])

    def test_tokenize_multi_sentence_tweet(self):
        self.assertEqual(self.tokenizer.tokenize('This is good. This is fine. This is OK.'),
                         ['This', 'is', 'good', '.', 'This', 'is', 'fine', '.', 'This', 'is', 'OK', '.'])


if __name__ == '__main__':
    unittest.main()
