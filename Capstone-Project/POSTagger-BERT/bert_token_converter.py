from typing import List, Tuple

import bert

from vocab_util import TargetVocabUtil


class BertTokenConverter:
    def __init__(self, model_dir, tvu: TargetVocabUtil):
        self.bert_tokenizer = bert.bert_tokenization.FullTokenizer(
            vocab_file=f'{model_dir}/vocab.txt',
            do_lower_case=False)
        self.tvu = tvu

    def tokenize_raw(self, value: str) -> List[str]:
        return self.bert_tokenizer.tokenize(value)

    def tokenize_value(self, value: str) -> List[str]:
        if len(value) > 1:
            if value[0] == '+':
                return ['##' + value[1:]]
            elif value[-1] == '+':
                value = value[:-1]

        return self.tokenize_raw(value)

    def convert_to_tokens(self, raw_tokens_and_labels: List[List[Tuple[str, str]]]) -> List[List[Tuple[str, str]]]:
        output = []
        for raw_labeled_tweet in raw_tokens_and_labels:
            output_labeled_tweet: List[Tuple[str, str]] = []
            for value, label in raw_labeled_tweet:
                subvalues = self.tokenize_value(value)
                for subvalue in subvalues:
                    output_labeled_tweet.append((subvalue, label))
            output.append(output_labeled_tweet)

        return output

    def convert_to_ids(self, raw_tokens_and_labels: List[List[Tuple[str, str]]]) -> List[List[Tuple[int, int]]]:
        output = []
        for raw_labeled_tweet in raw_tokens_and_labels:
            output_labeled_tweet: List[Tuple[int, int]] = []
            for value, label in raw_labeled_tweet:
                output_value = self.bert_tokenizer.convert_tokens_to_ids([value])[0]
                output_label = self.tvu.nn_pos_to_int[label]
                output_labeled_tweet.append((output_value, output_label))
            output.append(output_labeled_tweet)

        return output
