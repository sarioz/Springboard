import unittest

from bert_token_converter import BertTokenConverter
from vocab_util import TargetVocabUtil

BERT_PRETRAINED_MODEL_DIR = "../multi_cased_L-12_H-768_A-12/"


class MultilingualBertTokenConverterTestCase(unittest.TestCase):

    def test_convert(self):
        converter = BertTokenConverter(BERT_PRETRAINED_MODEL_DIR, TargetVocabUtil())

        result = converter.convert([
             (['this', 'list', 'has', 'plus+', '+es'], 'neutral'),
             (['this', 'list', 'has', 'plus+', '+es', 'too'], 'neutral'),
             (['I', "'m", 'getting', 'this', 'do', "n't"], 'neutral'),
             (['I', "'m", 'getting', 'this', 'do', "n't", "go"], 'neutral'),
             (['Apostrophe', "'", 'in', 'the', 'middle'], 'neutral'),
        ])

        self.assertEqual([
            (['this', 'list', 'has', 'pluse', '##s'], 'neutral'),
            (['this', 'list', 'has', 'pluse', '##s', 'too'], 'neutral'),
            (['I', "'", 'm', 'getting', 'this', 'don', "'", 't'], 'neutral'),
            (['I', "'", 'm', 'getting', 'this', 'don', "'", 't', 'go'], 'neutral'),
            (['A', '##post', '##rophe', "'", 'in', 'the', 'middle'], 'neutral'),
            ],
            result)

    def test_to_ids(self):
        converter = BertTokenConverter(BERT_PRETRAINED_MODEL_DIR, TargetVocabUtil())
        convert_result = converter.convert([(['this', 'is', 'a', 'good', 'enough', 'test'], 'positive')])
        to_ids_result = converter.convert_to_ids(convert_result)
        self.assertEqual([([10531, 10124, 169, 15198, 21408, 15839], 2)],
                         to_ids_result)


if __name__ == '__main__':
    unittest.main()
