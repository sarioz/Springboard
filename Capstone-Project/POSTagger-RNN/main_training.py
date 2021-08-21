import tensorflow as tf

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

from labeled_data_loader import LabeledDataLoader
from model_creator import LstmModelCreator
from nn_input_preparer import NNInputPreparer
from vocab_util import VocabUtil


MAX_SEQ_LEN = 128
EMBEDDING_DIM = 256
LSTM_DIM = 256

EXPERIMENT_NAME = f'05_filtered_bi_LSTM_{EMBEDDING_DIM}_{LSTM_DIM}'
MAX_EPOCHS = 100

BASE_DIR = f'models/{EXPERIMENT_NAME}/'

CONTINUE_TRAINING = True
INITIAL_EPOCH = 61 if CONTINUE_TRAINING else 0
TRAINING_MODEL_FILENAME_TO_CONTINUE = BASE_DIR + 'ep_61_valacc_0.95301.h5'


def main_training():
    print(f'Using TensorFlow version {tf.__version__}')
    loader = LabeledDataLoader('../data/pos/train.conll')
    tr_tweets = loader.parse_tokens_and_labels(loader.load_lines())
    print(f'Loaded {len(tr_tweets)} training tweets')

    unique_input_tokens = set([item[0] for tweet in tr_tweets for item in tweet])
    sorted_input_tokens = sorted(unique_input_tokens)

    vu = VocabUtil(sorted_input_tokens)
    nn_input_preparer = NNInputPreparer(vu, max_seq_len=MAX_SEQ_LEN)
    tr_tweets = nn_input_preparer.filter_out_long_sequences(tr_tweets)
    print(f'Training on {len(tr_tweets)} tweets, each no longer than {MAX_SEQ_LEN} tokens')
    tr_irregular_inputs = [[vu.nn_input_token_to_int[item[0]] for item in tweet] for tweet in tr_tweets]
    tr_irregular_targets = [[vu.nn_pos_to_int[item[1]] for item in tweet] for tweet in tr_tweets]
    tr_rectangular_inputs = nn_input_preparer.rectangularize_inputs(tr_irregular_inputs)
    tr_rectangular_targets = nn_input_preparer.rectangularize_targets(tr_irregular_targets)
    tr_targets_one_hot_encoded = nn_input_preparer.rectangular_targets_to_one_hot(tr_rectangular_targets)

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

    dev_loader = LabeledDataLoader('../data/pos/dev.conll')
    dev_tweets = dev_loader.parse_tokens_and_labels(dev_loader.load_lines())
    print(f'Loaded {len(dev_tweets)} dev tweets')
    dev_tweets = nn_input_preparer.filter_out_long_sequences(dev_tweets)
    print(f'Validating on {len(dev_tweets)} dev tweets, each no longer than {MAX_SEQ_LEN} tokens')
    dev_irregular_inputs = [[vu.nn_input_token_to_int[item[0]]
                            if item[0] in vu.nn_input_token_to_int
                            else vu.nn_input_token_to_int['<OOV>']
                            for item in tweet]
                            for tweet in dev_tweets]
    dev_irregular_targets = [[vu.nn_pos_to_int[item[1]] for item in tweet] for tweet in dev_tweets]
    dev_rectangular_inputs = nn_input_preparer.rectangularize_inputs(dev_irregular_inputs)
    dev_rectangular_targets = nn_input_preparer.rectangularize_targets(dev_irregular_targets)
    dev_targets_one_hot_encoded = nn_input_preparer.rectangular_targets_to_one_hot(dev_rectangular_targets)

    model.fit(x=tr_rectangular_inputs, y=tr_targets_one_hot_encoded, batch_size=32,
              initial_epoch=INITIAL_EPOCH,
              epochs=MAX_EPOCHS,
              validation_data=(dev_rectangular_inputs, dev_targets_one_hot_encoded),
              callbacks=[checkpoint])


if __name__ == '__main__':
    main_training()
