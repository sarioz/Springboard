import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tqdm import tqdm

from main_training import MAX_SEQ_LEN, create_vocab_util_from_training_set, prep_validation_set
from nn_input_preparer import NNInputPreparer

EXPERIMENT_NAME = '15_upsampled_bi_LSTM_64_64'
TRAINING_MODEL_FILENAME = f'models/{EXPERIMENT_NAME}/ep_8_valacc_0.46249.h5'

TRAINING_INPUT_FILENAME = '../data/sa/train.conll'
DEV_INPUT_FILENAME = '../data/sa/dev.conll'

UPSAMPLE = True


def main_inference():
    print(f'Using TensorFlow version {tf.__version__}')

    print(f'Loading model {TRAINING_MODEL_FILENAME}')
    trained_model = load_model(TRAINING_MODEL_FILENAME)
    trained_model.summary()

    vu = create_vocab_util_from_training_set(TRAINING_INPUT_FILENAME)
    nn_input_preparer = NNInputPreparer(vu, MAX_SEQ_LEN)

    for input_filename in [DEV_INPUT_FILENAME]:
        rectangular_inputs, rectangular_targets, targets_one_hot_encoded = \
            prep_validation_set(input_filename, nn_input_preparer, vu, UPSAMPLE)

        trained_model.evaluate(rectangular_inputs, targets_one_hot_encoded, batch_size=32)

        argmax_confusion_matrix = np.zeros((vu.get_output_vocab_size(), vu.get_output_vocab_size()), dtype=int)
        expected_sampling_confusion_matrix = np.zeros((vu.get_output_vocab_size(), vu.get_output_vocab_size()))

        expected_sampling_accuracy_sum = 0.0
        num_correct_argmax_predictions = 0
        it = 0
        for rectangular_input, target_human in tqdm(zip(rectangular_inputs, rectangular_targets)):
            it += 1
            target_index = vu.nn_rsl_to_int[target_human]
            rectangular_input_2d = np.array(rectangular_input)
            rectangular_input_2d.shape = (1, MAX_SEQ_LEN)
            predicted_probabilities = trained_model(rectangular_input_2d, training=False)[0]
            if it < 10:
                print(rectangular_input)
                print('target_index:', target_index)
                print(predicted_probabilities)
                print()
            # the predicted index if we take the class with the largest probability
            argmax_index = np.argmax(predicted_probabilities)
            if argmax_index == target_index:
                num_correct_argmax_predictions += 1
            argmax_confusion_matrix[target_index][argmax_index] += 1
            # rhs is the probability of guessing target_index if we sample according to predicted probabilities
            expected_sampling_accuracy_sum += tf.keras.backend.get_value(predicted_probabilities[target_index])
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
