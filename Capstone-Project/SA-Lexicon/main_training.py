import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint

from feature_extractor import FeatureExtractor
from labeled_tweets_loader import LabeledTweetsLoader
from lexicon_loader import LexiconLoader
from model_creator import ModelCreator
from nn_input_preparer import NNInputPreparer
from token_summarizer import TokenSummarizer
from vocab_util import VocabUtil

TRAINING_INPUT_FILENAME = '../data/sa/train.conll'
DEV_INPUT_FILENAME = '../data/sa/dev.conll'
INITIAL_EPOCH = 0
MAX_EPOCHS = 50

HIDDEN_SIZE = 10
EXPERIMENT_NAME = f'01_two_dense_hidden_size_{HIDDEN_SIZE}'

BASE_DIR = f'models/{EXPERIMENT_NAME}/'


def main_training():
    lexicon_loader = LexiconLoader()
    scored_lexicon: dict = lexicon_loader.load_all_and_merge()
    tr_tweets_loader = LabeledTweetsLoader(TRAINING_INPUT_FILENAME)
    tr_labeled_tweets = tr_tweets_loader.parse_tokens_and_labels(tr_tweets_loader.load_lines())

    token_summarizer = TokenSummarizer(scored_lexicon)
    feature_extractor = FeatureExtractor(scored_lexicon)

    vu = VocabUtil()
    nn_input_preparer = NNInputPreparer(vu)

    tr_feature_vectors = []  # 2D array of feature vectors
    for labeled_tweet in tr_labeled_tweets:
        known_token_sequence = token_summarizer.get_known_tokens(labeled_tweet[0])
        feature_vector = feature_extractor.compute_feature_vector(known_token_sequence)
        tr_feature_vectors.append(feature_vector)
    tr_network_input = np.array(tr_feature_vectors)
    tr_targets = [labeled_tweet[1] for labeled_tweet in tr_labeled_tweets]
    tr_targets_one_hot_encoded = nn_input_preparer.rectangular_targets_to_one_hot(tr_targets)

    dev_tweets_loader = LabeledTweetsLoader(DEV_INPUT_FILENAME)
    dev_labeled_tweets = dev_tweets_loader.parse_tokens_and_labels(dev_tweets_loader.load_lines())
    dev_feature_vectors = []  # 2D array of feature vectors
    for labeled_tweet in dev_labeled_tweets:
        known_token_sequence = token_summarizer.get_known_tokens(labeled_tweet[0])
        feature_vector = feature_extractor.compute_feature_vector(known_token_sequence)
        dev_feature_vectors.append(feature_vector)
    dev_network_input = np.array(dev_feature_vectors)
    dev_targets = [labeled_tweet[1] for labeled_tweet in dev_labeled_tweets]
    dev_targets_one_hot_encoded = nn_input_preparer.rectangular_targets_to_one_hot(dev_targets)

    # Every epoch is cheap (< 1ms), so we don't need the ability to continue training from a previous model.
    print("Commencing new training run")
    model_creator = ModelCreator(vu)
    model = model_creator.create_two_dense_model(hidden_layer_size=HIDDEN_SIZE)

    cp_filepath = BASE_DIR + 'ep_{epoch}_valacc_{val_accuracy:.5f}.h5'
    checkpoint = ModelCheckpoint(cp_filepath, monitor='val_accuracy', verbose=1,
                                 save_best_only=False)

    model.fit(tr_network_input, tr_targets_one_hot_encoded, batch_size=32,
              epochs=MAX_EPOCHS,
              validation_data=(dev_network_input, dev_targets_one_hot_encoded),
              callbacks=[checkpoint])


if __name__ == '__main__':
    main_training()
