import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tqdm import tqdm

from main_training import create_vocab_util_from_training_set, prep_validation_set
from nn_input_preparer import NNInputPreparer

MAX_SEQ_LEN = 128
EXPERIMENT_NAME = '08_mzt_bi_LSTM_64_64'
TRAINING_MODEL_FILENAME = f'models/{EXPERIMENT_NAME}/ep_71_valacc_0.95505.h5'
TRAINING_INPUT_FILENAME = '../data/pos/train.conll'
DEV_INPUT_FILENAME = '../data/pos/dev.conll'


def main_inference():
    print(f'Using TensorFlow version {tf.__version__}')
    print(f'Loading model {TRAINING_MODEL_FILENAME}')
    trained_model = load_model(TRAINING_MODEL_FILENAME)
    vu = create_vocab_util_from_training_set(TRAINING_INPUT_FILENAME)
    nn_input_preparer = NNInputPreparer(vu, max_seq_len=MAX_SEQ_LEN)

    for input_filename in [DEV_INPUT_FILENAME]:
        rectangular_inputs, rectangular_targets, targets_one_hot_encoded = \
            prep_validation_set(input_filename, nn_input_preparer, vu)
        trained_model.evaluate(rectangular_inputs, targets_one_hot_encoded, batch_size=32)
        num_tokens_in_dataset = num_token_level_correct_argmax_predictions = \
            num_token_level_correct_argmax_predictions_incl_pads = 0
        tweet_level_argmax_accuracy_sum = tweet_level_expected_sampling_accuracy_sum = \
            token_level_expected_sampling_accuracy_sum = token_level_expected_sampling_accuracy_sum_incl_pads = 0.0
        for rectangular_input, rectangular_target_indices in tqdm(zip(rectangular_inputs, rectangular_targets)):
            num_tokens_in_current_tweet = num_current_tweet_correct_argmax_predictions = 0
            current_tweet_expected_sampling_accuracy_sum = 0.0
            rectangular_input_2d = np.array(rectangular_input)
            rectangular_input_2d.shape = (1, MAX_SEQ_LEN)
            predicted_probabilities_sequence = trained_model(rectangular_input_2d, training=False)[0]
            for predicted_probabilities, target_index in zip(
                    predicted_probabilities_sequence, rectangular_target_indices):
                # the predicted index if we take the class with the largest probability
                argmax_index = np.argmax(predicted_probabilities)
                # probability of guessing target_index if we sample according to predicted probabilities
                prob_sampling_success_on_token = \
                    tf.keras.backend.get_value(predicted_probabilities[target_index])
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
