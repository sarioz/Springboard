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


EXPERIMENT_NAME = '05_MLBERT_frozen_on_dev'
TRAINING_MODEL_FILENAME = f'models/{EXPERIMENT_NAME}/ep_2_valacc_0.98186.h5'
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

    for input_filename in [DEV_INPUT_FILENAME]:
        loader = LabeledDataLoader(input_filename)
        tweets = loader.parse_raw_tokens_and_labels(loader.load_lines())
        tweets = btc.convert_to_tokens(tweets)
        tweets = btc.convert_to_ids(tweets)
        tweets = nn_input_preparer.filter_out_long_sequences(tweets)
        print(f'processing all {len(tweets)} not-too-long tweets from {input_filename}')
        irregular_inputs = [[item[0] for item in tweet] for tweet in tweets]
        irregular_targets = [[item[1] for item in tweet] for tweet in tweets]

        num_tokens_in_dataset = num_token_level_correct_argmax_predictions = \
            num_token_level_correct_argmax_predictions_incl_pads = 0
        tweet_level_argmax_accuracy_sum = tweet_level_expected_sampling_accuracy_sum = \
            token_level_expected_sampling_accuracy_sum = token_level_expected_sampling_accuracy_sum_incl_pads = 0.0

        for (irregular_input, irregular_target_indices) in tqdm(zip(irregular_inputs, irregular_targets)):
            rectangular_inputs = nn_input_preparer.rectangularize_inputs([irregular_input])
            rectangular_targets = nn_input_preparer.rectangularize_targets([irregular_target_indices])
            num_tokens_in_current_tweet = num_current_tweet_correct_argmax_predictions = 0
            current_tweet_expected_sampling_accuracy_sum = 0.0
            predicted_probabilities_sequence = trained_model.predict(rectangular_inputs)
            for predicted_probabilities, target_index in zip(
                    predicted_probabilities_sequence[0], rectangular_targets[0]):
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

        num_tweets_in_dataset = len(tweets)
        num_tokens_in_dataset_incl_pads = MAX_SEQ_LEN * num_tweets_in_dataset

        print(f'Argmax accuracy for {input_filename} including padding:',
              num_token_level_correct_argmax_predictions_incl_pads / num_tokens_in_dataset_incl_pads)
        print(f'Expected sampling accuracy for {input_filename} including padding:',
              token_level_expected_sampling_accuracy_sum_incl_pads / num_tokens_in_dataset_incl_pads)

        print(f'Token-level argmax accuracy for {input_filename}:',
              num_token_level_correct_argmax_predictions / num_tokens_in_dataset)
        print(f'Token-level expected sampling accuracy for {input_filename}:',
              token_level_expected_sampling_accuracy_sum / num_tokens_in_dataset)

        print(f'Tweet-level argmax accuracy for {input_filename}:',
              tweet_level_argmax_accuracy_sum / num_tweets_in_dataset)
        print(f'Tweet-level expected sampling accuracy for {input_filename}:',
              tweet_level_expected_sampling_accuracy_sum / num_tweets_in_dataset)


if __name__ == '__main__':
    main_inference()
