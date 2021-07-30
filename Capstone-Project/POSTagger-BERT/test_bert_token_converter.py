import unittest

from bert_token_converter import BertTokenConverter

BERT_PRETRAINED_MODEL_DIR = "../multi_cased_L-12_H-768_A-12/"


class MultilingualBertTokenConverterTestCase(unittest.TestCase):

    def test_convert(self):
        converter = BertTokenConverter(BERT_PRETRAINED_MODEL_DIR)

        result = converter.convert(
            [[('maybe', 'ADV'), ('the', 'DET'), ('teachers', 'NOUN'), ('are', 'VERB'), ('not', 'PART'), ('.', 'PUNCT')],
             [('yeah', 'INTJ'), ('.', 'PUNCT')], [('cada', 'DET'), ('qué', 'DET'), ('tiempo', 'NOUN'), ('?', 'PUNCT')],
             [('sí', 'INTJ'), ('sí', 'INTJ'), ('ellos', 'PRON'), ('pasan', 'VERB'), ('.', 'PUNCT')],
             [('is', 'VERB'), ('it', 'PRON'), ('a', 'DET'), ('miniature', 'NOUN'), ('?', 'PUNCT')]]
        )
        self.assertEqual([[('may', 'ADV'), ('##be', 'ADV'), ('the', 'DET'), ('teachers', 'NOUN'), ('are', 'VERB'),
                           ('not', 'PART'), ('.', 'PUNCT')], [('ye', 'INTJ'), ('##ah', 'INTJ'), ('.', 'PUNCT')],
                          [('cada', 'DET'), ('qué', 'DET'), ('tiempo', 'NOUN'), ('?', 'PUNCT')],
                          [('sí', 'INTJ'), ('sí', 'INTJ'), ('ellos', 'PRON'), ('pasa', 'VERB'), ('##n', 'VERB'),
                           ('.', 'PUNCT')],
                          [('is', 'VERB'), ('it', 'PRON'), ('a', 'DET'), ('mini', 'NOUN'), ('##ature', 'NOUN'),
                           ('?', 'PUNCT')]],
                         result)

    def test_tokenize_value_oov_token(self):
        converter = BertTokenConverter(BERT_PRETRAINED_MODEL_DIR)
        self.assertEqual(['h', '^', 'l', '##j', '%', 'k', '##h'],
                         converter.tokenize_value('h^lj%kh'))

    def test_tokenize_value_continuation(self):
        converter = BertTokenConverter(BERT_PRETRAINED_MODEL_DIR)
        self.assertEqual(['##le'],
                         converter.tokenize_value('+le'))

    def test_tokenize_value_continued(self):
        converter = BertTokenConverter(BERT_PRETRAINED_MODEL_DIR)
        self.assertEqual(['hacer'],
                         converter.tokenize_value('hacer+'))


if __name__ == '__main__':
    unittest.main()
