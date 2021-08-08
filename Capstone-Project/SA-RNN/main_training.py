import tensorflow as tf

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

from labeled_data_loader import LabeledDataLoader
from model_creator import LstmAndPoolingModelCreator
from nn_input_preparer import NNInputPreparer
from vocab_util import VocabUtil

EMBEDDING_DIM = 256
LSTM_DIM = 256

EXPERIMENT_NAME = f'03_bi_LSTM_{EMBEDDING_DIM}_{LSTM_DIM}'
MAX_EPOCHS = 20

BASE_DIR = f'models/{EXPERIMENT_NAME}/'

CONTINUE_TRAINING = False
INITIAL_EPOCH = 50 if CONTINUE_TRAINING else 0

TRAINING_MODEL_FILENAME_TO_CONTINUE = BASE_DIR + 'ep_6_valacc_0.87869.h5'


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


if __name__ == '__main__':
    main_training()
