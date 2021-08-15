import tensorflow as tf

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tqdm import tqdm
import numpy as np

from labeled_data_loader import LabeledDataLoader
from model_creator import LstmAndPoolingModelCreator
from nn_input_preparer import NNInputPreparer
from vocab_util import VocabUtil

EMBEDDING_DIM = 256
LSTM_DIM = 256

EXPERIMENT_NAME = f'04_bi_LSTM_{EMBEDDING_DIM}_{LSTM_DIM}'
MAX_EPOCHS = 20

BASE_DIR = f'models/{EXPERIMENT_NAME}/'

CONTINUE_TRAINING = False
INITIAL_EPOCH = 50 if CONTINUE_TRAINING else 0

TRAINING_MODEL_FILENAME_TO_CONTINUE = BASE_DIR + 'ep_6_valacc_0.87869.h5'

TRAINING_INPUT_FILENAME = '../data/sa/train.conll'
DEV_INPUT_FILENAME = '../data/sa/dev.conll'


def main_training():
    print(f'Using TensorFlow version {tf.__version__}')
    loader = LabeledDataLoader('../data/sa/train.conll')
    labeled_tweets = loader.parse_tokens_and_labels(loader.load_lines())

    unique_input_tokens = set([item for labeled_tweet in labeled_tweets for item in labeled_tweet[0]])
    sorted_input_tokens = sorted(unique_input_tokens)

    vu = VocabUtil(sorted_input_tokens)

    irregular_inputs = [[vu.nn_input_token_to_int[item[0]] for item in tweet] for tweet in labeled_tweets]
    rectangular_targets = [tweet[1] for tweet in labeled_tweets]

    nn_input_preparer = NNInputPreparer(vu)

    rectangular_inputs = nn_input_preparer.rectangularize_inputs(irregular_inputs)
    targets_one_hot_encoded = nn_input_preparer.rectangular_targets_to_one_hot(rectangular_targets)

    if CONTINUE_TRAINING:
        print('Continuing training from', TRAINING_MODEL_FILENAME_TO_CONTINUE)
        model = load_model(TRAINING_MODEL_FILENAME_TO_CONTINUE)
        model.summary()
    else:
        print("Commencing new training run")
        model_creator = LstmAndPoolingModelCreator(vu, embedding_dim=EMBEDDING_DIM, lstm_dim=LSTM_DIM)
        model = model_creator.create_bi_lstm_based_model()

    cp_filepath = BASE_DIR + 'ep_{epoch}_valacc_{val_accuracy:.5f}.h5'

    checkpoint = ModelCheckpoint(cp_filepath, monitor='val_accuracy', verbose=1,
                                 save_best_only=False)

    model.fit(rectangular_inputs, targets_one_hot_encoded, batch_size=32,
              initial_epoch=INITIAL_EPOCH,
              epochs=MAX_EPOCHS,
              validation_split=0.1, callbacks=[checkpoint])

    # training_loader = LabeledDataLoader(TRAINING_INPUT_FILENAME)
    # labeled_training_tweets = training_loader.parse_tokens_and_labels(training_loader.load_lines())
    # unique_training_tokens = set([item for labeled_tweet in labeled_training_tweets for item in labeled_tweet[0]])
    # sorted_training_tokens = sorted(unique_training_tokens)
    # # we should instantiate a vocab util only based on the training tokens, not dev/test tokens
    # vu = VocabUtil(sorted_training_tokens)

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

        argmax_confusion_matrix = np.zeros((vu.get_output_vocab_size(), vu.get_output_vocab_size()), dtype=int)
        expected_sampling_confusion_matrix = np.zeros((vu.get_output_vocab_size(), vu.get_output_vocab_size()))

        expected_sampling_accuracy_sum = 0.0
        num_correct_argmax_predictions = 0
        for irregular_input, target_human in tqdm(zip(irregular_inputs, rectangular_targets)):
            target_index = vu.nn_rsl_to_int[target_human]
            predicted_probabilities = model.predict(irregular_input)[0]
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
    main_training()
