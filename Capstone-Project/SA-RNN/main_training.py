import tensorflow as tf

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

from labeled_data_loader import LabeledDataLoader
from model_creator import LstmAndPoolingModelCreator
from nn_input_preparer import NNInputPreparer
from vocab_util import VocabUtil

MAX_SEQ_LEN = 128
EMBEDDING_DIM = 256
LSTM_DIM = 256

EXPERIMENT_NAME = f'05_bi_LSTM_{EMBEDDING_DIM}_{LSTM_DIM}'
MAX_EPOCHS = 50

BASE_DIR = f'models/{EXPERIMENT_NAME}/'

CONTINUE_TRAINING = False
INITIAL_EPOCH = 50 if CONTINUE_TRAINING else 0

TRAINING_MODEL_FILENAME_TO_CONTINUE = BASE_DIR + 'ep_6_valacc_0.87869.h5'

TRAINING_INPUT_FILENAME = '../data/sa/train.conll'
DEV_INPUT_FILENAME = '../data/sa/dev.conll'


def main_training():
    print(f'Using TensorFlow version {tf.__version__}')
    tr_loader = LabeledDataLoader(TRAINING_INPUT_FILENAME)
    tr_labeled_tweets = tr_loader.parse_tokens_and_labels(tr_loader.load_lines())

    tr_unique_input_tokens = set([item for labeled_tweet in tr_labeled_tweets for item in labeled_tweet[0]])
    tr_sorted_input_tokens = sorted(tr_unique_input_tokens)
    vu = VocabUtil(tr_sorted_input_tokens)
    nn_input_preparer = NNInputPreparer(vu, MAX_SEQ_LEN)

    tr_labeled_tweets = nn_input_preparer.filter_out_long_tweets(tr_labeled_tweets)
    tr_irregular_inputs = [[vu.nn_input_token_to_int[item[0]] for item in tweet] for tweet in tr_labeled_tweets]
    tr_rectangular_targets = [tweet[1] for tweet in tr_labeled_tweets]
    tr_rectangular_inputs = nn_input_preparer.rectangularize_inputs(tr_irregular_inputs)
    tr_targets_one_hot_encoded = nn_input_preparer.rectangular_targets_to_one_hot(tr_rectangular_targets)

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

    dev_loader = LabeledDataLoader(DEV_INPUT_FILENAME)
    dev_labeled_tweets = dev_loader.parse_tokens_and_labels(dev_loader.load_lines())
    dev_labeled_tweets = nn_input_preparer.filter_out_long_tweets(dev_labeled_tweets)
    print(f'Validating on {len(dev_labeled_tweets)} dev tweets, each no longer than {MAX_SEQ_LEN} tokens')
    dev_irregular_inputs = [[vu.nn_input_token_to_int[item[0]]
                            if item[0] in vu.nn_input_token_to_int
                            else vu.nn_input_token_to_int['<OOV>']
                            for item in tweet]
                            for tweet in dev_labeled_tweets]
    dev_rectangular_inputs = nn_input_preparer.rectangularize_inputs(dev_irregular_inputs)
    dev_rectangular_targets = [tweet[1] for tweet in dev_labeled_tweets]
    dev_targets_one_hot_encoded = nn_input_preparer.rectangular_targets_to_one_hot(dev_rectangular_targets)

    model.fit(tr_rectangular_inputs, tr_targets_one_hot_encoded, batch_size=32,
              initial_epoch=INITIAL_EPOCH,
              epochs=MAX_EPOCHS,
              validation_data=(dev_rectangular_inputs, dev_targets_one_hot_encoded),
              callbacks=[checkpoint])


if __name__ == '__main__':
    main_training()
