import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from bert import BertModelLayer
from tqdm import tqdm

from bert_token_converter import BertTokenConverter
from labeled_data_loader import LabeledDataLoader
from main_training import BERT_PRETRAINED_MODEL_DIR
from nn_input_preparer import NNInputPreparer
from vocab_util import TargetVocabUtil


EXPERIMENT_NAME = '03_MLBERT_frozen_full'
TRAINING_MODEL_FILENAME = f'models/{EXPERIMENT_NAME}/ep_9_valacc_0.98173.h5'
MAX_SEQ_LEN = 128  # the model was trained for this

TRAINING_INPUT_FILENAME = '../data/pos/train.conll'
DEV_INPUT_FILENAME = '../data/pos/dev.conll'


def main_inference():
    print(f'Using TensorFlow version {tf.__version__}')
    print(f'Loading model {TRAINING_MODEL_FILENAME}')
    trained_model = load_model(TRAINING_MODEL_FILENAME,
                               custom_objects={"BertModelLayer": BertModelLayer})

    print('Loaded fine-tuned model:')
    trained_model.summary()

    tvu = TargetVocabUtil()
    btc = BertTokenConverter(BERT_PRETRAINED_MODEL_DIR, tvu)
    nn_input_preparer = NNInputPreparer(tvu=tvu, max_seq_len=MAX_SEQ_LEN)

    for input_filename in [TRAINING_INPUT_FILENAME, DEV_INPUT_FILENAME]:
        loader = LabeledDataLoader(input_filename)
        tweets = loader.parse_raw_tokens_and_labels(loader.load_lines())
        tweets = btc.convert_to_tokens(tweets)
        tweets = btc.convert_to_ids(tweets)
        tweets = nn_input_preparer.filter_out_long_sequences(tweets)
        print(f'processing all {len(tweets)} not-too-long tweets from {input_filename}')
        irregular_inputs = [[item[0] for item in tweet] for tweet in tweets]
        irregular_targets = [[item[1] for item in tweet] for tweet in tweets]

        num_tokens_in_dataset = num_token_level_correct_argmax_predictions = 0
        tweet_level_argmax_accuracy_sum = tweet_level_expected_sampling_accuracy_sum = \
            token_level_expected_sampling_accuracy_sum = 0.0
        for (irregular_input, irregular_target_indices) in tqdm(zip(irregular_inputs, irregular_targets)):
            num_current_tweet_correct_argmax_predictions = 0
            current_tweet_expected_sampling_accuracy_sum = 0.0
            rectangular_inputs = nn_input_preparer.rectangularize_inputs([irregular_input])
            predicted_probability_sequence = trained_model.predict(rectangular_inputs)[0]
            for predicted_probabilities, target_index in zip(predicted_probability_sequence, irregular_target_indices):
                # the predicted index if we take the class with the largest probability
                if np.argmax(predicted_probabilities) == target_index:
                    num_token_level_correct_argmax_predictions += 1
                    num_current_tweet_correct_argmax_predictions += 1
                # probability of guessing target_index if we sample according to predicted probabilities
                prob_sampling_success_on_token = predicted_probabilities[target_index]
                current_tweet_expected_sampling_accuracy_sum += prob_sampling_success_on_token
                token_level_expected_sampling_accuracy_sum += prob_sampling_success_on_token
            num_tokens_in_current_tweet = len(irregular_input)
            num_tokens_in_dataset += num_tokens_in_current_tweet

            current_tweet_argmax_accuracy = num_current_tweet_correct_argmax_predictions / num_tokens_in_current_tweet
            current_tweet_expected_sampling_accuracy = \
                current_tweet_expected_sampling_accuracy_sum / num_tokens_in_current_tweet

            tweet_level_argmax_accuracy_sum += current_tweet_argmax_accuracy
            tweet_level_expected_sampling_accuracy_sum += current_tweet_expected_sampling_accuracy

        print(f'Token-level argmax accuracy for {input_filename}:',
              num_token_level_correct_argmax_predictions / num_tokens_in_dataset)
        print(f'Token-level expected sampling accuracy for {input_filename}:',
              token_level_expected_sampling_accuracy_sum / num_tokens_in_dataset)

        num_tweets_in_dataset = len(irregular_inputs)
        print(f'Tweet-level argmax accuracy for {input_filename}:',
              tweet_level_argmax_accuracy_sum / num_tweets_in_dataset)
        print(f'Tweet-level expected sampling accuracy for {input_filename}:',
              tweet_level_expected_sampling_accuracy_sum / num_tweets_in_dataset)


if __name__ == '__main__':
    main_inference()
