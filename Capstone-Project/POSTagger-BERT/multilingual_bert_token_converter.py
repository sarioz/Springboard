from typing import List, Tuple

import bert


class MultilingualBertTokenConverter:
    def __init__(self, model_dir):
        self.bert_tokenizer = bert.bert_tokenization.FullTokenizer(
            vocab_file=f'{model_dir}/vocab.txt',
            do_lower_case=False)

    def tokenize_raw(self, value: str) -> List[str]:
        return self.bert_tokenizer.tokenize(value)

    def tokenize_value(self, value: str) -> List[str]:
        if len(value) > 1:
            if value[0] == '+':
                return ['##' + value[1:]]
            elif value[-1] == '+':
                value = value[:-1]

        return self.tokenize_raw(value)

    def convert(self, raw_tokens_and_labels: List[List[Tuple[str, str]]]) -> List[List[Tuple[str, str]]]:
        output = []
        for raw_labeled_tweet in raw_tokens_and_labels:
            output_labeled_tweet: List[Tuple[str, str]] = []
            for value, label in raw_labeled_tweet:
                subvalues = self.tokenize_value(value)
                for subvalue in subvalues:
                    output_labeled_tweet.append((subvalue, label))
            output.append(output_labeled_tweet)

        return output
