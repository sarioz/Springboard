from typing import List, Tuple


class LabeledDataLoader:
    def __init__(self, input_filename):
        self.input_filename = input_filename
        self.token_field_index = 0

    def load_lines(self) -> List[str]:
        lines = []
        with open(self.input_filename, 'r') as input_file:
            for line in input_file:
                lines.append(line)

        return lines

    def parse_tokens_and_labels(self, lines: List[str]) -> List[Tuple[List[str], str]]:
        labeled_tweets = []
        tweet = []
        for line in lines:
            if not line or line == '\n':
                labeled_tweets.append((tweet, label))
                tweet = []
                continue
            if line.startswith('# sent_enum'):
                label = line.rstrip().split("\t")[-1]
                continue
            fields = line.rstrip().split("\t")
            tweet.append(fields[self.token_field_index])

        return labeled_tweets
