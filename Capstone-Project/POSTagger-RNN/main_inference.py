import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from labeled_data_loader import LabeledDataLoader
from nn_input_preparer import NNInputPreparer
from vocab_util import VocabUtil


EXPERIMENT_NAME = '03_bi_LSTM_128_256'
TRAINING_MODEL_FILENAME = f'models/{EXPERIMENT_NAME}/ep_51_valacc_0.94030.h5'

TRAINING_INPUT_FILENAME = '../data/pos/train.conll'
DEV_INPUT_FILENAME = '../data/pos/dev.conll'


def main_inference():
    print(f'Using TensorFlow version {tf.__version__}')
    print(f'Loading model {TRAINING_MODEL_FILENAME}')
    loaded_training_model = load_model(TRAINING_MODEL_FILENAME)

    training_loader = LabeledDataLoader(TRAINING_INPUT_FILENAME)
    training_tweets = training_loader.parse_tokens_and_labels(training_loader.load_lines())
    unique_training_tokens = set([item[0] for tweet in training_tweets for item in tweet])
    sorted_training_tokens = sorted(unique_training_tokens)
    # we should instantiate a vocab util only based on the training tokens, not dev/test tokens
    vu = VocabUtil(sorted_training_tokens)
    nn_input_preparer = NNInputPreparer(vu)

    k = 1000
    for input_filename in [TRAINING_INPUT_FILENAME, DEV_INPUT_FILENAME]:
        print(f'processing the first {k} tweets from {input_filename}')
        loader = LabeledDataLoader(input_filename)
        tweets = loader.parse_tokens_and_labels(loader.load_lines())
        irregular_inputs = [[vu.nn_input_token_to_int[item[0]]
                             if item[0] in vu.nn_input_token_to_int
                             else vu.nn_input_token_to_int['<OOV>']
                             for item in tweet]
                            for tweet in tweets]
        irregular_targets = [[vu.nn_pos_to_int[item[1]] for item in tweet] for tweet in tweets]

        rectangular_inputs = nn_input_preparer.rectangularize_inputs(irregular_inputs)[:k]
        rectangular_targets = nn_input_preparer.rectangularize_targets(irregular_targets)[:k]

        # print("token\tpredicted\ttarget\n")
        predictions = correct_predictions = 0
        tweet_accuracy_sum = 0.0
        for rectangular_input, rectangular_target in zip(rectangular_inputs, rectangular_targets):
            tweet_level_predictions = tweet_level_correct_predictions = 0
            tokens_human = [vu.nn_input_tokens[i] for i in rectangular_input if i != 0]
            predicted = loaded_training_model.predict(rectangular_input)
            # could also consider sampling from the probability distribution
            predicted_indices = [np.argmax(predicted[i]) for i in range(len(tokens_human))]
            predicted_human = [vu.nn_pos_tuple[i] for i in predicted_indices]
            target_indices = [rectangular_target[i] for i in range(len(tokens_human))]
            target_human = [vu.nn_pos_tuple[i] for i in target_indices]
            for tok, pred, tar in zip(tokens_human, predicted_human, target_human):
                predictions += 1
                tweet_level_predictions += 1
                # print(f"{tok}\t{pred}\t{tar}")
                if pred == tar:
                    correct_predictions += 1
                    tweet_level_correct_predictions += 1
            tweet_accuracy = tweet_level_correct_predictions / tweet_level_predictions
            tweet_accuracy_sum += tweet_accuracy
            # print()

        print(f'Token-level accuracy for {input_filename}:', correct_predictions / predictions)
        print(f'Tweet-level accuracy for {input_filename}:', tweet_accuracy_sum / len(rectangular_inputs))


if __name__ == '__main__':
    main_inference()
