import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tqdm import tqdm

from labeled_data_loader import LabeledDataLoader
from vocab_util import VocabUtil

EXPERIMENT_NAME = '03_bi_LSTM_256_256'
TRAINING_MODEL_FILENAME = f'models/{EXPERIMENT_NAME}/ep_20_valacc_0.88770.h5'

TRAINING_INPUT_FILENAME = '../data/sa/train.conll'
DEV_INPUT_FILENAME = '../data/sa/dev.conll'


def main_inference():
    print(f'Using TensorFlow version {tf.__version__}')

    print(f'Loading model {TRAINING_MODEL_FILENAME}')
    loaded_training_model = load_model(TRAINING_MODEL_FILENAME)
    loaded_training_model.summary()

    training_loader = LabeledDataLoader(TRAINING_INPUT_FILENAME)
    labeled_training_tweets = training_loader.parse_tokens_and_labels(training_loader.load_lines())
    unique_training_tokens = set([item for labeled_tweet in labeled_training_tweets for item in labeled_tweet[0]])
    sorted_training_tokens = sorted(unique_training_tokens)
    # we should instantiate a vocab util only based on the training tokens, not dev/test tokens
    vu = VocabUtil(sorted_training_tokens)

    for input_filename in [TRAINING_INPUT_FILENAME, DEV_INPUT_FILENAME]:
        loader = LabeledDataLoader(input_filename)
        labeled_tweets = loader.parse_tokens_and_labels(loader.load_lines())
        print(f'processing all {len(labeled_tweets)} tweets from {input_filename}')

        irregular_inputs = [[vu.nn_input_token_to_int[token]
                             if token in vu.nn_input_token_to_int
                             else vu.nn_input_token_to_int['<OOV>']
                             for token in labeled_tweet[0]]
                            for labeled_tweet in labeled_tweets]

        rectangular_targets = [labeled_tweet[1] for labeled_tweet in labeled_tweets]

        expected_sampling_accuracy_sum = 0.0
        num_correct_argmax_predictions = 0
        for irregular_input, target_human in tqdm(zip(irregular_inputs, rectangular_targets)):
            target_index = vu.nn_rsl_to_int[target_human]
            predicted_probabilities = loaded_training_model.predict(irregular_input)[0]
            # the predicted index if we take the class with the largest probability
            if np.argmax(predicted_probabilities) == target_index:
                num_correct_argmax_predictions += 1
            # rhs is the probability of guessing target_index if we sample according to predicted probabilities
            expected_sampling_accuracy_sum += predicted_probabilities[target_index]

        num_tweets_in_dataset = len(rectangular_targets)

        print(f'Argmax accuracy for {input_filename}:',
              num_correct_argmax_predictions / num_tweets_in_dataset)
        print(f'Expected sampling accuracy for {input_filename}:',
              expected_sampling_accuracy_sum / num_tweets_in_dataset)


if __name__ == '__main__':
    main_inference()
