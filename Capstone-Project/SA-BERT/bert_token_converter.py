from typing import List, Tuple

import bert

from vocab_util import TargetVocabUtil


class BertTokenConverter:
    def __init__(self, model_dir, tvu: TargetVocabUtil):
        self.bert_tokenizer = bert.bert_tokenization.FullTokenizer(
            vocab_file=f'{model_dir}/vocab.txt',
            do_lower_case=False)
        self.tvu = tvu

    def join_on_pluses(self, tokens: List[str]) -> List[str]:
        output_token_sequence: List[str] = []
        i = 0
        while i < len(tokens) - 1:
            if len(tokens[i]) > 1 and tokens[i][-1] == '+' and tokens[i + 1][0] == '+':
                output_token_sequence.append(tokens[i][:-1] + tokens[i + 1][1:])
                i += 2
            else:
                output_token_sequence.append(tokens[i])
                i += 1
        if i < len(tokens):
            output_token_sequence.append(tokens[i])
        return output_token_sequence

    def join_on_apostrophes(self, tokens: List[str]) -> List[str]:
        output_token_sequence: List[str] = []
        i = 0
        while i < len(tokens) - 1:
            if len(tokens[i + 1]) > 1 and (tokens[i + 1][0] == "'" or tokens[i + 1] == "n't"):
                output_token_sequence.append(tokens[i] + tokens[i + 1])
                i += 2
            else:
                output_token_sequence.append(tokens[i])
                i += 1
        if i < len(tokens):
            output_token_sequence.append(tokens[i])
        return output_token_sequence

    def convert_to_bert_tokens(self, tokens: List[str]) -> List[str]:
        tweet: str = ' '.join(tokens)
        return self.bert_tokenizer.tokenize(tweet)

    def convert(self, labeled_tweets: List[Tuple[List[str], str]]) -> List[Tuple[List[str], str]]:
        output: List[Tuple[List[str], str]] = []
        for (tweet, label) in labeled_tweets:
            tweet = self.join_on_pluses(tweet)
            tweet = self.join_on_apostrophes(tweet)
            tweet = self.convert_to_bert_tokens(tweet)
            output.append((tweet, label))
        return output

    def convert_to_ids(self, labeled_tweets: List[Tuple[List[str], str]]) -> List[Tuple[List[int], int]]:
        output_labeled_tweet: List[Tuple[List[int], int]] = []
        for raw_labeled_tweet in labeled_tweets:
            output_values = self.bert_tokenizer.convert_tokens_to_ids(raw_labeled_tweet[0])
            output_label = self.tvu.nn_rsl_to_int[raw_labeled_tweet[1]]
            output_labeled_tweet.append((output_values, output_label))
        return output_labeled_tweet
