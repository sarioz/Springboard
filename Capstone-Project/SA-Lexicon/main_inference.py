import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tqdm import tqdm

from feature_extractor import FeatureExtractor
from labeled_tweets_loader import LabeledTweetsLoader
from lexicon_loader import LexiconLoader
from nn_input_preparer import NNInputPreparer
from token_summarizer import TokenSummarizer
from vocab_util import VocabUtil

EXPERIMENT_NAME = '01_two_dense_hidden_size_10'
TRAINING_MODEL_FILENAME = f'models/{EXPERIMENT_NAME}/ep_8_valacc_0.56105.h5'

DEV_INPUT_FILENAME = '../data/sa/dev.conll'


def main_inference():
    print(f'Using TensorFlow version {tf.__version__}')

    print(f'Loading model {TRAINING_MODEL_FILENAME}')
    trained_model = load_model(TRAINING_MODEL_FILENAME)
    trained_model.summary()

    lexicon_loader = LexiconLoader()
    scored_lexicon: dict = lexicon_loader.load_all_and_merge()
    token_summarizer = TokenSummarizer(scored_lexicon)
    feature_extractor = FeatureExtractor(scored_lexicon)
    vu = VocabUtil()
    nn_input_preparer = NNInputPreparer(vu)

    for input_filename in [DEV_INPUT_FILENAME]:
        tweets_loader = LabeledTweetsLoader(DEV_INPUT_FILENAME)
        labeled_tweets = tweets_loader.parse_tokens_and_labels(tweets_loader.load_lines())
        feature_vectors = []  # 2D array of feature vectors
        for labeled_tweet in labeled_tweets:
            known_token_sequence = token_summarizer.get_known_tokens(labeled_tweet[0])
            feature_vector = feature_extractor.compute_feature_vector(known_token_sequence)
            feature_vectors.append(feature_vector)
        network_input = np.array(feature_vectors)
        print('network_input.shape:', network_input.shape)
        targets = [labeled_tweet[1] for labeled_tweet in labeled_tweets]
        targets_one_hot_encoded = nn_input_preparer.rectangular_targets_to_one_hot(targets)

        argmax_confusion_matrix = np.zeros((vu.get_output_vocab_size(), vu.get_output_vocab_size()), dtype=int)
        expected_sampling_confusion_matrix = np.zeros((vu.get_output_vocab_size(), vu.get_output_vocab_size()))

        expected_sampling_accuracy_sum = 0.0
        num_correct_argmax_predictions = 0
        for rectangular_input, target_human in tqdm(zip(network_input, targets)):
            rectangular_input.shape = (1, 3)
            target_index = vu.nn_rsl_to_int[target_human]
            predicted_probabilities = trained_model.predict(rectangular_input)[0]
            # the predicted index if we take the class with the largest probability
            argmax_index = np.argmax(predicted_probabilities)
            if argmax_index == target_index:
                num_correct_argmax_predictions += 1
            argmax_confusion_matrix[target_index][argmax_index] += 1
            # rhs is the probability of guessing target_index if we sample according to predicted probabilities
            expected_sampling_accuracy_sum += predicted_probabilities[target_index]
            for i in range(vu.get_output_vocab_size()):
                expected_sampling_confusion_matrix[target_index][i] += predicted_probabilities[i]
        num_tweets_in_dataset = len(targets)

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
