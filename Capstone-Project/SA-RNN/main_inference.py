import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tqdm import tqdm

from labeled_data_loader import LabeledDataLoader
from main_training import MAX_SEQ_LEN
from nn_input_preparer import NNInputPreparer
from vocab_util import VocabUtil

EXPERIMENT_NAME = '06_mzf_bi_LSTM_256_256'
TRAINING_MODEL_FILENAME = f'models/{EXPERIMENT_NAME}/ep_10_valacc_0.84131.h5'

TRAINING_INPUT_FILENAME = '../data/sa/train.conll'
DEV_INPUT_FILENAME = '../data/sa/dev.conll'


def main_inference():
    print(f'Using TensorFlow version {tf.__version__}')

    print(f'Loading model {TRAINING_MODEL_FILENAME}')
    trained_model = load_model(TRAINING_MODEL_FILENAME)
    trained_model.summary()

    tr_loader = LabeledDataLoader(TRAINING_INPUT_FILENAME)
    tr_labeled_tweets = tr_loader.parse_tokens_and_labels(tr_loader.load_lines())
    tr_unique_tokens = set([item for labeled_tweet in tr_labeled_tweets for item in labeled_tweet[0]])
    tr_sorted_tokens = sorted(tr_unique_tokens)
    # we instantiate a vocab util based on the training tokens, not dev/test tokens
    vu = VocabUtil(tr_sorted_tokens)

    nn_input_preparer = NNInputPreparer(vu, MAX_SEQ_LEN)

    for input_filename in [DEV_INPUT_FILENAME]:
        loader = LabeledDataLoader(input_filename)
        labeled_tweets = loader.parse_tokens_and_labels(loader.load_lines())
        labeled_tweets = nn_input_preparer.filter_out_long_tweets(labeled_tweets)
        print(f'processing all not-too-long {len(labeled_tweets)} tweets from {input_filename}')

        irregular_inputs = [[vu.nn_input_token_to_int[token]
                             if token in vu.nn_input_token_to_int
                             else vu.nn_input_token_to_int['<OOV>']
                             for token in labeled_tweet[0]]
                            for labeled_tweet in labeled_tweets]
        rectangular_inputs = nn_input_preparer.rectangularize_inputs(irregular_inputs)
        rectangular_targets = [labeled_tweet[1] for labeled_tweet in labeled_tweets]

        argmax_confusion_matrix = np.zeros((vu.get_output_vocab_size(), vu.get_output_vocab_size()), dtype=int)
        expected_sampling_confusion_matrix = np.zeros((vu.get_output_vocab_size(), vu.get_output_vocab_size()))

        expected_sampling_accuracy_sum = 0.0
        num_correct_argmax_predictions = 0
        for rectangular_input, target_human in tqdm(zip(rectangular_inputs, rectangular_targets)):
            target_index = vu.nn_rsl_to_int[target_human]
            predicted_probabilities = trained_model.predict([rectangular_input])[0]
            # the predicted index if we take the class with the largest probability
            argmax_index = np.argmax(predicted_probabilities)
            if argmax_index == target_index:
                num_correct_argmax_predictions += 1
            argmax_confusion_matrix[target_index][argmax_index] += 1
            # rhs is the probability of guessing target_index if we sample according to predicted probabilities
            expected_sampling_accuracy_sum += predicted_probabilities[target_index]
            for i in range(vu.get_output_vocab_size()):
                expected_sampling_confusion_matrix[target_index][i] += predicted_probabilities[i]
        num_tweets_in_dataset = len(rectangular_targets)

        print(f'Argmax accuracy for {input_filename}:',
              num_correct_argmax_predictions / num_tweets_in_dataset)
        print(f'Expected sampling accuracy for {input_filename}:',
              expected_sampling_accuracy_sum / num_tweets_in_dataset)

        print(f"Argmax confusion matrix of targets vs predicted for {input_filename}:\n"
              f"{vu.raw_sentiment_labels}\n",
              argmax_confusion_matrix)
        print(f"Expected sampling confusion matrix of targets vs predicted for {input_filename}:\n"
              f"{vu.raw_sentiment_labels}\n",
              expected_sampling_confusion_matrix)


if __name__ == '__main__':
    main_inference()
