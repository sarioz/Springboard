import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tqdm import tqdm

from labeled_data_loader import LabeledDataLoader
from nn_input_preparer import NNInputPreparer
from vocab_util import VocabUtil

MAX_SEQ_LEN = 128
EXPERIMENT_NAME = '05_filtered_bi_LSTM_256_256'
TRAINING_MODEL_FILENAME = f'models/{EXPERIMENT_NAME}/ep_64_valacc_0.95508.h5'

TRAINING_INPUT_FILENAME = '../data/pos/train.conll'
DEV_INPUT_FILENAME = '../data/pos/dev.conll'


def main_inference():
    print(f'Using TensorFlow version {tf.__version__}')
    print(f'Loading model {TRAINING_MODEL_FILENAME}')
    trained_model = load_model(TRAINING_MODEL_FILENAME)

    training_loader = LabeledDataLoader(TRAINING_INPUT_FILENAME)
    training_tweets = training_loader.parse_tokens_and_labels(training_loader.load_lines())
    unique_training_tokens = set([item[0] for tweet in training_tweets for item in tweet])
    sorted_training_tokens = sorted(unique_training_tokens)
    # we should instantiate a vocab util only based on the training tokens, not dev/test tokens
    vu = VocabUtil(sorted_training_tokens)
    nn_input_preparer = NNInputPreparer(vu, max_seq_len=MAX_SEQ_LEN)

    for input_filename in [DEV_INPUT_FILENAME]:
        loader = LabeledDataLoader(input_filename)
        tweets = loader.parse_tokens_and_labels(loader.load_lines())
        tweets = nn_input_preparer.filter_out_long_sequences(tweets)
        print(f'processing all not-too-long {len(tweets)} tweets from {input_filename}')
        irregular_inputs = [[vu.nn_input_token_to_int[item[0]]
                             if item[0] in vu.nn_input_token_to_int
                             else vu.nn_input_token_to_int['<OOV>']
                             for item in tweet]
                            for tweet in tweets]
        irregular_targets = [[vu.nn_pos_to_int[item[1]] for item in tweet] for tweet in tweets]
        rectangular_inputs = nn_input_preparer.rectangularize_inputs(irregular_inputs)
        rectangular_targets = nn_input_preparer.rectangularize_targets(irregular_targets)

        num_tokens_in_dataset = num_token_level_correct_argmax_predictions = \
            num_token_level_correct_argmax_predictions_incl_pads = 0
        tweet_level_argmax_accuracy_sum = tweet_level_expected_sampling_accuracy_sum = \
            token_level_expected_sampling_accuracy_sum = token_level_expected_sampling_accuracy_sum_incl_pads = 0.0
        for rectangular_input, rectangular_target_indices in tqdm(zip(rectangular_inputs, rectangular_targets)):
            num_tokens_in_current_tweet = num_current_tweet_correct_argmax_predictions = 0
            current_tweet_expected_sampling_accuracy_sum = 0.0
            predicted_probabilities_sequence = trained_model.predict(rectangular_input)
            for predicted_probabilities, target_index in zip(
                    predicted_probabilities_sequence, rectangular_target_indices):
                predicted_probabilities = predicted_probabilities[0]
                # the predicted index if we take the class with the largest probability
                argmax_index = np.argmax(predicted_probabilities)
                # probability of guessing target_index if we sample according to predicted probabilities
                prob_sampling_success_on_token = predicted_probabilities[target_index]
                if argmax_index == target_index:
                    num_token_level_correct_argmax_predictions_incl_pads += 1
                token_level_expected_sampling_accuracy_sum_incl_pads += prob_sampling_success_on_token

                if target_index != 0:
                    if argmax_index == target_index:
                        num_token_level_correct_argmax_predictions += 1
                        num_current_tweet_correct_argmax_predictions += 1
                    current_tweet_expected_sampling_accuracy_sum += prob_sampling_success_on_token
                    token_level_expected_sampling_accuracy_sum += prob_sampling_success_on_token
                    num_tokens_in_current_tweet += 1
                    num_tokens_in_dataset += 1

            # every tweet has at least one non-padding token, so we don't worry about division by zero
            current_tweet_argmax_accuracy = num_current_tweet_correct_argmax_predictions / num_tokens_in_current_tweet
            current_tweet_expected_sampling_accuracy = \
                current_tweet_expected_sampling_accuracy_sum / num_tokens_in_current_tweet

            tweet_level_argmax_accuracy_sum += current_tweet_argmax_accuracy
            tweet_level_expected_sampling_accuracy_sum += current_tweet_expected_sampling_accuracy

        num_tokens_in_dataset_incl_pads = MAX_SEQ_LEN * len(rectangular_inputs)
        print(f'Argmax accuracy for {input_filename} including padding:',
              num_token_level_correct_argmax_predictions_incl_pads / num_tokens_in_dataset_incl_pads)
        print(f'Expected sampling accuracy for {input_filename} including padding:',
              token_level_expected_sampling_accuracy_sum_incl_pads / num_tokens_in_dataset_incl_pads)

        print(f'Token-level argmax accuracy for {input_filename}:',
              num_token_level_correct_argmax_predictions / num_tokens_in_dataset)
        print(f'Token-level expected sampling accuracy for {input_filename}:',
              token_level_expected_sampling_accuracy_sum / num_tokens_in_dataset)

        num_tweets_in_dataset = len(rectangular_inputs)
        print(f'Tweet-level argmax accuracy for {input_filename}:',
              tweet_level_argmax_accuracy_sum / num_tweets_in_dataset)
        print(f'Tweet-level expected sampling accuracy for {input_filename}:',
              tweet_level_expected_sampling_accuracy_sum / num_tweets_in_dataset)


if __name__ == '__main__':
    main_inference()
