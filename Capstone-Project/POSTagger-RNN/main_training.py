import tensorflow as tf

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

from labeled_data_loader import LabeledDataLoader
from model_creator import LstmModelCreator
from nn_input_preparer import NNInputPreparer
from vocab_util import VocabUtil


EMBEDDING_DIM = 128
LSTM_DIM = 256

EXPERIMENT_NAME = f'03_bi_LSTM_{EMBEDDING_DIM}_{LSTM_DIM}'
MAX_EPOCHS = 100

BASE_DIR = f'models/{EXPERIMENT_NAME}/'

CONTINUE_TRAINING = True
INITIAL_EPOCH = 50 if CONTINUE_TRAINING else -1

TRAINING_MODEL_FILENAME_TO_CONTINUE = BASE_DIR + 'ep_50_valacc_0.93833.h5'


def main_training():
    print(f'Using TensorFlow version {tf.__version__}')
    loader = LabeledDataLoader('../data/pos/train.conll')
    tweets = loader.parse_tokens_and_labels(loader.load_lines())

    unique_input_tokens = set([item[0] for tweet in tweets for item in tweet])
    sorted_input_tokens = sorted(unique_input_tokens)

    print(len(sorted_input_tokens))

    vu = VocabUtil(sorted_input_tokens)

    irregular_inputs = [[vu.nn_input_token_to_int[item[0]] for item in tweet] for tweet in tweets]
    irregular_targets = [[vu.nn_pos_to_int[item[1]] for item in tweet] for tweet in tweets]

    nn_input_preparer = NNInputPreparer(vu)

    rectangular_inputs = nn_input_preparer.rectangularize_inputs(irregular_inputs)
    rectangular_targets = nn_input_preparer.rectangularize_targets(irregular_targets)

    targets_one_hot_encoded = nn_input_preparer.rectangular_targets_to_one_hot(rectangular_targets)

    if CONTINUE_TRAINING:
        print('Continuing training from', TRAINING_MODEL_FILENAME_TO_CONTINUE)
        model = load_model(TRAINING_MODEL_FILENAME_TO_CONTINUE)
        model.summary()
    else:
        print("Commencing new training run")
        model_creator = LstmModelCreator(vu, embedding_dim=EMBEDDING_DIM, lstm_dim=LSTM_DIM)
        model = model_creator.create_bi_lstm_model()

    cp_filepath = BASE_DIR + 'ep_{epoch}_valacc_{val_accuracy:.5f}.h5'

    checkpoint = ModelCheckpoint(cp_filepath, monitor='val_accuracy', verbose=1,
                                 save_best_only=False)

    model.fit(rectangular_inputs, targets_one_hot_encoded, batch_size=32,
              initial_epoch=INITIAL_EPOCH,
              epochs=MAX_EPOCHS,
              validation_split=0.1, callbacks=[checkpoint])


if __name__ == '__main__':
    main_training()
