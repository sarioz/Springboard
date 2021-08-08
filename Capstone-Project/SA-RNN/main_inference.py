import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from numpy.random import default_rng

from labeled_data_loader import LabeledDataLoader
from nn_input_preparer import NNInputPreparer
from vocab_util import VocabUtil

EXPERIMENT_NAME = '03_bi_LSTM_256_256'
TRAINING_MODEL_FILENAME = f'models/{EXPERIMENT_NAME}/ep_20_valacc_0.85574.h5'

TRAINING_INPUT_FILENAME = '../data/sa/train.conll'
DEV_INPUT_FILENAME = '../data/sa/dev.conll'


def main_inference():
    print(f'Using TensorFlow version {tf.__version__}')

    rng = default_rng()

    print(f'Loading model {TRAINING_MODEL_FILENAME}')
    loaded_training_model = load_model(TRAINING_MODEL_FILENAME)
    loaded_training_model.summary()

    training_loader = LabeledDataLoader(TRAINING_INPUT_FILENAME)
    labeled_training_tweets = training_loader.parse_tokens_and_labels(training_loader.load_lines())
    unique_training_tokens = set([item for labeled_tweet in labeled_training_tweets for item in labeled_tweet[0]])
    sorted_training_tokens = sorted(unique_training_tokens)
    # we should instantiate a vocab util only based on the training tokens, not dev/test tokens
    vu = VocabUtil(sorted_training_tokens)
    nn_input_preparer = NNInputPreparer(vu)

    for input_filename in [TRAINING_INPUT_FILENAME, DEV_INPUT_FILENAME]:
    # for input_filename in [TRAINING_INPUT_FILENAME]:
        k = 500
        print(f'processing the first {k} tweets from {input_filename}')
        loader = LabeledDataLoader(input_filename)
        labeled_tweets = loader.parse_tokens_and_labels(loader.load_lines())

        irregular_inputs = [[vu.nn_input_token_to_int[token]
                             if token in vu.nn_input_token_to_int
                             else vu.nn_input_token_to_int['<OOV>']
                             for token in labeled_tweet[0]]
                            for labeled_tweet in labeled_tweets]

        # rectangular_inputs = nn_input_preparer.rectangularize_inputs(irregular_inputs[:k])
        rectangular_targets = [labeled_tweet[1] for labeled_tweet in labeled_tweets[:k]]
        # predicted_probabilities_sequence = loaded_training_model.predict(rectangular_inputs)

        matches_argmax = matches_sampling = 0
        for i, target in enumerate(rectangular_targets):
            # print('irregular_inputs[i]:', irregular_inputs[i])
            # print('len(irregular_inputs[i]):', len(irregular_inputs[i]))
            # print('target:', target)
            predicted_probabilities_2 = loaded_training_model.predict([irregular_inputs[i]])[0]
            # print('predicted_probabilities_2.shape:', predicted_probabilities_2.shape)
            # print(predicted_probabilities_2)

            predicted_label_argmax = np.argmax(predicted_probabilities_2)
            predicted_human_argmax = vu.raw_sentiment_labels[predicted_label_argmax]
            if predicted_human_argmax == target:
                matches_argmax += 1

            predicted_label_sampling = rng.choice(3, 1, p=predicted_probabilities_2)[0]
            predicted_human_sampling = vu.raw_sentiment_labels[predicted_label_sampling]
            if predicted_human_sampling == target:
                matches_sampling += 1

        print('accuracy_argmax:', matches_argmax / k)
        print('accuracy_sampling:', matches_sampling / k)


if __name__ == '__main__':
    main_inference()
