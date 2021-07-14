from labeled_data_loader import LabeledDataLoader
from vocab_util import POS_TO_INT


def main():
    loader = LabeledDataLoader('../data/pos/train.conll')
    tweets = loader.parse_tokens_and_labels(loader.load_lines())
    tweets = [[(token, POS_TO_INT[pos_name]) for token, pos_name in tweet] for tweet in tweets]
    unique_input_tokens = set([item[0] for tweet in tweets for item in tweet])
    sorted_input_tokens = sorted(unique_input_tokens)
    print(sorted_input_tokens)
    print(len(sorted_input_tokens))


if __name__ == '__main__':
    main()
