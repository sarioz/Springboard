import tensorflow as tf

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

from labeled_data_loader import LabeledDataLoader
from model_creator import LstmAndPoolingModelCreator
from nn_input_preparer import NNInputPreparer
from vocab_util import VocabUtil

MAX_SEQ_LEN = 128
EMBEDDING_DIM = 64
LSTM_DIM = 64
MASK_ZERO = True

EXPERIMENT_NAME = f'15_upsampled_bi_LSTM_{EMBEDDING_DIM}_{LSTM_DIM}'
MAX_EPOCHS = 100

BASE_DIR = f'models/{EXPERIMENT_NAME}/'

CONTINUE_TRAINING = False
INITIAL_EPOCH = 50 if CONTINUE_TRAINING else 0

TRAINING_MODEL_FILENAME_TO_CONTINUE = BASE_DIR + 'ep_6_valacc_0.87869.h5'

TRAINING_INPUT_FILENAME = '../data/sa/train.conll'
DEV_INPUT_FILENAME = '../data/sa/dev.conll'

UPSAMPLE = True


def create_vocab_util_from_training_set(tr_input_filename: str) -> VocabUtil:
    tr_loader = LabeledDataLoader(tr_input_filename)
    tr_labeled_tweets = tr_loader.parse_tokens_and_labels(tr_loader.load_lines())
    tr_unique_tokens = set([item for labeled_tweet in tr_labeled_tweets for item in labeled_tweet[0]])
    tr_sorted_tokens = sorted(tr_unique_tokens)
    print(f"Creating VocabUtil from {len(tr_sorted_tokens)} unique tokens.")
    return VocabUtil(tr_sorted_tokens)


def prep_validation_set(input_filename: str, nn_input_preparer: NNInputPreparer, vu: VocabUtil,
                        upsample: bool):
    loader = LabeledDataLoader(input_filename)
    labeled_tweets = loader.parse_tokens_and_labels(loader.load_lines())
    labeled_tweets = nn_input_preparer.filter_out_long_tweets(labeled_tweets)
    if upsample:
        labeled_tweets = nn_input_preparer.crude_upsample(labeled_tweets)
    irregular_inputs = [[vu.nn_input_token_to_int[token]
                         if token in vu.nn_input_token_to_int
                         else vu.nn_input_token_to_int['<OOV>']
                         for token in labeled_tweet[0]]
                        for labeled_tweet in labeled_tweets]
    rectangular_inputs = nn_input_preparer.rectangularize_inputs(irregular_inputs)
    rectangular_targets = [tweet[1] for tweet in labeled_tweets]
    targets_one_hot_encoded = nn_input_preparer.rectangular_targets_to_one_hot(rectangular_targets)

    return rectangular_inputs, rectangular_targets, targets_one_hot_encoded


def main_training():
    print(f'Using TensorFlow version {tf.__version__}')

    vu = create_vocab_util_from_training_set(TRAINING_INPUT_FILENAME)

    tr_loader = LabeledDataLoader(TRAINING_INPUT_FILENAME)
    tr_labeled_tweets = tr_loader.parse_tokens_and_labels(tr_loader.load_lines())

    nn_input_preparer = NNInputPreparer(vu, MAX_SEQ_LEN)

    tr_labeled_tweets = nn_input_preparer.filter_out_long_tweets(tr_labeled_tweets)
    if UPSAMPLE:
        tr_labeled_tweets = nn_input_preparer.crude_upsample(tr_labeled_tweets)
    tr_irregular_inputs = [[vu.nn_input_token_to_int[token]
                            if token in vu.nn_input_token_to_int
                            else vu.nn_input_token_to_int['<OOV>']
                            for token in labeled_tweet[0]]
                           for labeled_tweet in tr_labeled_tweets]
    tr_rectangular_targets = [labeled_tweet[1] for labeled_tweet in tr_labeled_tweets]
    tr_rectangular_inputs = nn_input_preparer.rectangularize_inputs(tr_irregular_inputs)
    tr_targets_one_hot_encoded = nn_input_preparer.rectangular_targets_to_one_hot(tr_rectangular_targets)

    print(tr_labeled_tweets[0])
    print(tr_irregular_inputs[0])
    print(tr_rectangular_inputs[0])
    print(tr_targets_one_hot_encoded[0])

    if CONTINUE_TRAINING:
        print('Continuing training from', TRAINING_MODEL_FILENAME_TO_CONTINUE)
        model = load_model(TRAINING_MODEL_FILENAME_TO_CONTINUE)
        model.summary()
    else:
        print("Commencing new training run")
        model_creator = LstmAndPoolingModelCreator(vu, embedding_dim=EMBEDDING_DIM, lstm_dim=LSTM_DIM,
                                                   mask_zero=MASK_ZERO)
        model = model_creator.create_bi_lstm_based_model()

    cp_filepath = BASE_DIR + 'ep_{epoch}_valacc_{val_accuracy:.5f}.h5'

    checkpoint = ModelCheckpoint(cp_filepath, monitor='val_accuracy', verbose=1,
                                 save_best_only=False)

    dev_rectangular_inputs, dev_rectangular_targets, dev_targets_one_hot_encoded = \
        prep_validation_set(DEV_INPUT_FILENAME, nn_input_preparer, vu, UPSAMPLE)

    print(f'Validating on {len(dev_rectangular_inputs)} dev tweets, each no longer than {MAX_SEQ_LEN} tokens')

    model.fit(tr_rectangular_inputs, tr_targets_one_hot_encoded, batch_size=32,
              initial_epoch=INITIAL_EPOCH,
              epochs=MAX_EPOCHS,
              validation_data=(dev_rectangular_inputs, dev_targets_one_hot_encoded),
              callbacks=[checkpoint])


if __name__ == '__main__':
    main_training()
