import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from bert import BertModelLayer

from bert_token_converter import BertTokenConverter
from labeled_data_loader import LabeledDataLoader
from main_training import BERT_PRETRAINED_MODEL_DIR
from nn_input_preparer import NNInputPreparer
from vocab_util import TargetVocabUtil


EXPERIMENT_NAME = '03_MLBERT_frozen_full'
TRAINING_MODEL_FILENAME = f'models/{EXPERIMENT_NAME}/ep_9_valacc_0.98173.h5'

TRAINING_INPUT_FILENAME = '../data/pos/train.conll'
DEV_INPUT_FILENAME = '../data/pos/dev.conll'


def main_inference():
    print(f'Using TensorFlow version {tf.__version__}')
    print(f'Loading model {TRAINING_MODEL_FILENAME}')
    loaded_training_model = load_model(TRAINING_MODEL_FILENAME,
                                       custom_objects={"BertModelLayer": BertModelLayer})

    print('Loaded fine-tuned model:')
    loaded_training_model.summary()

    max_seq_len = 128
    tvu = TargetVocabUtil()
    btc = BertTokenConverter(BERT_PRETRAINED_MODEL_DIR, tvu)
    nn_input_preparer = NNInputPreparer(tvu=tvu, max_seq_len=max_seq_len)

    k = 1000
    for input_filename in [TRAINING_INPUT_FILENAME, DEV_INPUT_FILENAME]:
        loader = LabeledDataLoader(input_filename)
        tweets = loader.parse_raw_tokens_and_labels(loader.load_lines())[:k]
        print(f'processing the first {len(tweets)} tweets from {input_filename}')
        tweets = btc.convert_to_tokens(tweets)
        tweets_orig_tokens = tweets.copy()
        tweets = btc.convert_to_ids(tweets)
        tweets = nn_input_preparer.filter_out_long_sequences(tweets)
        irregular_inputs = [[item[0] for item in tweet] for tweet in tweets]
        rectangular_inputs = nn_input_preparer.rectangularize_inputs(irregular_inputs)

        predicted_probabilities = loaded_training_model.predict(rectangular_inputs)
        # could also consider sampling from the probability distribution
        predicted_index_sequences = [[np.argmax(q) for q in p] for p in predicted_probabilities]
        # print("token\tpredicted\ttarget\n")

        predictions, correct_predictions = 0, 0
        tweet_accuracy_sum = 0.0
        for (tweet_orig_tokens, predicted_index_sequence) in zip(tweets_orig_tokens, predicted_index_sequences):
            tweet_level_predictions = tweet_level_correct_predictions = 0
            tokens_human = [token[0] for token in tweet_orig_tokens]
            predicted_human = [tvu.nn_pos_tuple[index] for index in predicted_index_sequence]
            targets_human = [token[1] for token in tweet_orig_tokens]
            for tok, pred, tar in zip(tokens_human, predicted_human, targets_human):
                predictions += 1
                tweet_level_predictions += 1
                if pred == tar:
                    correct_predictions += 1
                    tweet_level_correct_predictions += 1
                # print(f"{tok}\t{pred}\t{tar}")
            tweet_accuracy = tweet_level_correct_predictions / tweet_level_predictions
            tweet_accuracy_sum += tweet_accuracy
            # print()

        print(f'Token-level accuracy for {input_filename}:', correct_predictions / predictions)
        print(f'Tweet-level accuracy for {input_filename}:', tweet_accuracy_sum / len(tweets_orig_tokens))


if __name__ == '__main__':
    main_inference()
